# IA Resumen Bancario – Banco Nación PLUS (BNA+)
# Herramienta para uso interno - AIE San Justo

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- UI / assets ----------------
HERE = Path(__file__).parent
LOGO = HERE / "assets" / "logo_aie.png"
FAVICON = HERE / "assets" / "favicon-aie.ico"

st.set_page_config(
    page_title="IA Resumen Bancario – Banco Nación PLUS",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="centered",
)

if LOGO.exists():
    st.image(str(LOGO), width=220)

st.title("IA Resumen Bancario – Banco Nación PLUS")

st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- deps diferidas ----------------
try:
    import pdfplumber
except Exception as e:
    st.error(f"No se pudo importar pdfplumber: {e}\nRevisá requirements.txt")
    st.stop()

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ---------------- regex / parse helpers ----------------
DATE_DDMM_RE = re.compile(r"^(?P<ddmm>\d{1,2}/\d{1,2})\b")
YEAR_RE = re.compile(r"^/(?P<yyyy>20\d{2})\b")
FULL_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/20\d{2}\b")

# dinero AR: -6.115.212,29  |  759.520,88  |  -5,28
MONEY_RE = re.compile(r"(?<!\S)-?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2}(?!\S)")
INT_RE = re.compile(r"\b\d{3,}\b")


def normalize_money(tok: str) -> float:
    if not tok:
        return np.nan
    tok = tok.strip().replace("−", "-")
    neg = tok.startswith("-")
    tok = tok.lstrip("-")
    main, frac = tok.rsplit(",", 1)
    main = main.replace(".", "").replace(" ", "")
    try:
        val = float(f"{main}.{frac}")
        return -val if neg else val
    except Exception:
        return np.nan


def fmt_ar(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    return f"{n:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


def extract_text_lines(file_like) -> list[str]:
    """Extrae texto por páginas y devuelve líneas normalizadas."""
    out = []
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for l in txt.splitlines():
                l = " ".join(l.split()).strip()
                if l:
                    out.append(l)
    return out


def parse_bna_plus_lines_to_df(lines: list[str]) -> pd.DataFrame:
    """
    Parser robusto BNA+.

    El PDF suele representar cada movimiento en 2-3 líneas:
      30/12 $
      4286 ... $ -5,28
      /2025 -6.115.212,29

    Pero a veces aparece en 1 sola línea (misma página / final de página), por ej:
      30/12 4273 ... $ -4,90 $ -6.115.134,31

    Este parser usa un autómata simple y NO desplaza saldos:
    - Detecta DD/MM (con o sin '$')
    - Captura COMPROBANTE + CONCEPTO + IMPORTE
    - Si el saldo no viene en la misma línea, lo toma de la línea siguiente que contiene '/YYYY' y el saldo.
    """
    rows = []
    pending_ddmm = None
    pending_year = None
    pending_comp = None
    pending_concept = None
    pending_importe = None

    def flush_with_saldo(saldo_val):
        nonlocal pending_ddmm, pending_year, pending_comp, pending_concept, pending_importe
        if not pending_ddmm or not pending_year or pending_comp is None or pending_importe is None:
            pending_ddmm = pending_year = pending_comp = pending_concept = pending_importe = None
            return
        fecha_str = f"{pending_ddmm}/{pending_year}"
        fecha = pd.to_datetime(fecha_str, dayfirst=True, errors="coerce")
        rows.append({
            "fecha": fecha,
            "comprobante": str(pending_comp),
            "concepto": (pending_concept or "").strip(),
            "importe": float(pending_importe),
            "saldo": float(saldo_val) if saldo_val is not None else np.nan,
        })
        pending_ddmm = pending_year = pending_comp = pending_concept = pending_importe = None

    for ln in lines:
        u = (ln or "").strip()
        if not u:
            continue

        # Saltar headers obvios
        up = u.upper()
        if up.startswith("FECHA:") or "ÚLTIMOS MOV" in up or "FECHA COMPROBANTE" in up:
            continue

        # Caso: línea combinada con DD/MM al inicio
        # 30/12 4273 CONCEPTO $ -4,90 $ -6.115.134,31
        m_combo = re.match(r'^(\d{2}/\d{2})\s+(\d{3,})\s+(.*)$', u)
        if m_combo:
            ddmm = m_combo.group(1)
            comp = m_combo.group(2)
            rest = m_combo.group(3)

            monies = MONEY_RE.findall(rest)
            if len(monies) >= 2:
                imp = normalize_money(monies[0])
                saldo = normalize_money(monies[-1])
                # año puede venir en otra línea; intentar inferir luego si aparece '/YYYY' sin movimiento
                # Si no hay año aquí, lo dejamos pendiente pero ya tenemos saldo e importe.
                pending_ddmm = ddmm
                pending_comp = comp
                pending_concept = re.sub(r'\$\s*-?\d{1,3}(?:\.\d{3})*,\d{2}.*$', '', rest).strip()
                pending_importe = imp
                # si hay año en la misma línea
                my = re.search(r'/(20\d{2})\b', u)
                if my:
                    pending_year = my.group(1)
                    flush_with_saldo(saldo)
                else:
                    # sin año: guardamos saldo temporal y esperamos línea "/YYYY" sola;
                    # si no aparece, se descarta por falta de fecha completa
                    pending_year = None
                    # stash saldo en el concepto para no perderlo (truco simple)
                    pending_concept = (pending_concept or "").strip()
                    pending_importe = imp
                    # guardamos saldo como float en pending_importe? no; usamos un buffer aparte
                    # Solución: si no hay año, igual intentamos con año de contexto posterior.
                    # Dejamos pendiente el saldo en una variable auxiliar:
                continue

        # Detectar línea DD/MM sola (con o sin $)
        m_ddmm = re.match(r'^(\d{2}/\d{2})\s*\$?$', u)
        if m_ddmm:
            pending_ddmm = m_ddmm.group(1)
            pending_year = None
            pending_comp = None
            pending_concept = None
            pending_importe = None
            continue

        # Detectar línea de movimiento: "4286 ... $ -5,28" (puede traer saldo también)
        m_mov = re.match(r'^(\d{3,})\s+(.*)$', u)
        if m_mov and pending_ddmm:
            pending_comp = m_mov.group(1)
            rest = m_mov.group(2)

            monies = MONEY_RE.findall(rest)
            if not monies:
                continue

            # Si hay 2 importes, el último es saldo
            if len(monies) >= 2:
                pending_importe = normalize_money(monies[0])
                saldo = normalize_money(monies[-1])
                pending_concept = re.sub(r'\$\s*-?\d{1,3}(?:\.\d{3})*,\d{2}.*$', '', rest).strip()
                my = re.search(r'/(20\d{2})\b', u)
                if my:
                    pending_year = my.group(1)
                    flush_with_saldo(saldo)
                else:
                    # saldo ya está, falta el año (puede venir en la próxima línea "/2025")
                    pending_year = None
                continue

            # Caso normal: 1 solo importe (el movimiento); saldo viene en línea siguiente
            pending_importe = normalize_money(monies[0])
            pending_concept = re.sub(r'\$\s*-?\d{1,3}(?:\.\d{3})*,\d{2}.*$', '', rest).strip()
            continue

        # Detectar línea "/YYYY SALDO"
        if pending_ddmm and pending_comp and pending_importe is not None:
            my = re.search(r'/(20\d{2})\b', u)
            monies = MONEY_RE.findall(u)
            if my and monies:
                pending_year = my.group(1)
                saldo = normalize_money(monies[-1])
                flush_with_saldo(saldo)
                continue

    return pd.DataFrame(rows)



def build_resumen_operativo(df: pd.DataFrame) -> pd.DataFrame:
    iva_base = float(df.loc[df["clasificacion"].eq("IVA BASE 21%"), "debito"].sum())
    neto_comision = round(iva_base / 0.21, 2) if iva_base else 0.0

    percep = float(df.loc[df["clasificacion"].eq("Percepciones IVA RG 2408"), "debito"].sum())

    ley_raw = float(df.loc[df["clasificacion"].eq("Ley 25.413"), "importe"].sum())
    ley_neto_gasto = -ley_raw  # cargo (negativo) -> gasto positivo

    total = neto_comision + iva_base + percep + ley_neto_gasto

    return pd.DataFrame(
        [
            ["COMISIÓN (NETO 21%)", neto_comision],
            ["I.V.A BASE (21%)", iva_base],
            ["RETEN. I.V.A. RG.2408", percep],
            ["GRAVAMEN LEY 25.413 (NETO)", ley_neto_gasto],
            ["TOTAL", total],
        ],
        columns=["Concepto", "Importe"],
    )


# ---------------- main ----------------
uploaded = st.file_uploader("Subí un PDF del resumen (Banco Nación PLUS)", type=["pdf"])
if uploaded is None:
    st.info("La app no almacena datos. Procesamiento local en memoria.")
    st.stop()

data = uploaded.read()
lines = extract_text_lines(io.BytesIO(data))
df = parse_bna_plus_lines_to_df(lines)

if df.empty:
    st.error("No se detectaron movimientos. Este PDF tiene un layout distinto o está escaneado.")
    st.stop()

df["comp_num"] = pd.to_numeric(df["comprobante"], errors="coerce")
df = df.sort_values(["fecha", "comp_num", "orden"]).reset_index(drop=True)

# Débito/Crédito desde IMPORTE (regla BNA+)
df["debito"] = np.where(df["importe"] < 0, -df["importe"], 0.0)
df["credito"] = np.where(df["importe"] > 0, df["importe"], 0.0)

df["clasificacion"] = df["concepto"].apply(clasificar)

saldo_inicial = float(df["saldo"].iloc[0])
saldo_final = float(df["saldo"].iloc[-1])
total_debitos = float(df["debito"].sum())
total_creditos = float(df["credito"].sum())
saldo_final_calculado = saldo_inicial + total_creditos - total_debitos
diferencia = saldo_final_calculado - saldo_final
cuadra = abs(diferencia) < 0.01

st.subheader("Conciliación bancaria")

# Orden lógico del período:
# - primer movimiento: menor FECHA y, si coincide, menor COMPROBANTE
# - último movimiento: mayor FECHA y, si coincide, mayor COMPROBANTE
df = df.sort_values(["fecha", "comp_num", "orden"]).reset_index(drop=True)

# BNA+ (regla operativa):
# - Saldo anterior: PRIMER saldo impreso del período.
# - Saldo final: NO aparece impreso; se infiere como:
#     saldo_final_inferido = último_saldo_impreso + último_importe (con signo).
saldo_anterior = float(df["saldo"].iloc[0])

ultimo_saldo_impreso = float(df["saldo"].iloc[-1])
ultimo_importe = float(df["importe"].iloc[-1])
saldo_final_inferido = ultimo_saldo_impreso + ultimo_importe

total_debitos = float(df["debito"].sum())
total_creditos = float(df["credito"].sum())

saldo_final_calculado = saldo_anterior + total_creditos - total_debitos
diferencia = saldo_final_calculado - saldo_final_inferido
cuadra = abs(diferencia) < 0.01

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.metric("Saldo anterior", f"$ {fmt_ar(saldo_anterior)}")
with r1c2:
    st.metric("Total débitos (–)", f"$ {fmt_ar(total_debitos)}")
with r1c3:
    st.metric("Total créditos (+)", f"$ {fmt_ar(total_creditos)}")
r2c1, r2c2 = st.columns(2)
with r2c1:
    st.metric("Saldo final (inferido)", f"$ {fmt_ar(saldo_final_inferido)}")
with r2c2:
    st.metric("Diferencia", f"$ {fmt_ar(diferencia)}")

if cuadra:
    st.success("Conciliado.")
else:
    st.error("No cuadra la conciliación.")

st.subheader("Resumen Operativo: Registración Módulo IVA")

resumen = build_resumen_operativo(df)
res_view = resumen.copy()
res_view["Importe"] = res_view["Importe"].map(fmt_ar)
st.dataframe(res_view, use_container_width=True)

st.subheader("Detalle de movimientos")
df_view = df.drop(columns=["comp_num"], errors="ignore").copy()
for c in ["importe", "debito", "credito", "saldo"]:
    df_view[c] = df_view[c].map(fmt_ar)
st.dataframe(
    df_view[["fecha", "comprobante", "concepto", "debito", "credito", "saldo", "clasificacion"]],
    use_container_width=True,
)

st.subheader("Descargar")
date_suffix = f"_{df['fecha'].iloc[-1].strftime('%Y%m%d')}" if pd.notna(df["fecha"].iloc[-1]) else ""

try:
    import xlsxwriter  # noqa: F401
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen_Operativo")
        wb = writer.book
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        date_fmt = wb.add_format({"num_format": "dd/mm/yyyy"})

        ws = writer.sheets["Movimientos"]
        for idx, col in enumerate(df.columns):
            ws.set_column(idx, idx, min(max(len(col), 12) + 2, 55))
        for colname in ["importe", "debito", "credito", "saldo"]:
            j = df.columns.get_loc(colname)
            ws.set_column(j, j, 16, money_fmt)
        j = df.columns.get_loc("fecha")
        ws.set_column(j, j, 14, date_fmt)

        ws2 = writer.sheets["Resumen_Operativo"]
        ws2.set_column(0, 0, 44)
        ws2.set_column(1, 1, 18, money_fmt)

    st.download_button(
        "📥 Descargar Excel (BNA+)",
        data=out.getvalue(),
        file_name=f"resumen_bna_plus{date_suffix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except Exception:
    st.download_button(
        "📥 Descargar CSV (fallback)",
        data=csv_bytes,
        file_name=f"resumen_bna_plus{date_suffix}.csv",
        mime="text/csv",
        use_container_width=True,
    )

if REPORTLAB_OK:
    try:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title="Resumen Operativo - BNA+")
        styles = getSampleStyleSheet()
        elems = [
            Paragraph("Resumen Operativo: Registración Módulo IVA (BNA+)", styles["Title"]),
            Spacer(1, 10),
        ]
        data_tbl = [["Concepto", "Importe"]] + [[r["Concepto"], fmt_ar(r["Importe"])] for _, r in resumen.iterrows()]
        tbl = Table(data_tbl, colWidths=[340, 140])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ]
            )
        )
        elems.append(tbl)
        elems.append(Spacer(1, 12))
        elems.append(Paragraph("Herramienta para uso interno - AIE San Justo", styles["Normal"]))
        doc.build(elems)

        st.download_button(
            "📄 Descargar PDF – Resumen Operativo (BNA+)",
            data=pdf_buf.getvalue(),
            file_name=f"Resumen_Operativo_BNA_PLUS{date_suffix}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.info(f"No se pudo generar el PDF del Resumen Operativo: {e}")

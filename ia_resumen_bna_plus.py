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


def merge_bna_plus_rows(lines: list[str]) -> list[str]:
    """
    En BNA+ suele venir en 2-3 líneas por fila:
      30/12 $
      4286 ... $ -5,28
      /2025 -6.115.212,29
    o:
      15/12
      3500 ... $ -36,00 $ 759.520,88
      /2025
    Esta función arma "bloques" por movimiento.
    """
    rows = []
    i = 0
    while i < len(lines):
        ln = lines[i]

        # saltar headers obvios
        if ln.upper().startswith("FECHA:") or "ÚLTIMOS MOV" in ln.upper() or "FECHA COMPROBANTE" in ln.upper():
            i += 1
            continue

        mdd = DATE_DDMM_RE.match(ln)
        if not mdd:
            i += 1
            continue

        ddmm = mdd.group("ddmm")
        chunk = [ln]
        i += 1

        yyyy = None
        # acumular hasta encontrar /yyyy o una fecha completa
        while i < len(lines):
            ln2 = lines[i]
            if DATE_DDMM_RE.match(ln2) and yyyy is not None:
                break

            chunk.append(ln2)

            my = YEAR_RE.match(ln2)
            if my:
                yyyy = my.group("yyyy")
                i += 1
                break

            if FULL_DATE_RE.search(ln2):
                yyyy = FULL_DATE_RE.search(ln2).group(0).split("/")[-1]
                i += 1
                break

            i += 1

        if yyyy is None:
            continue

        rows.append(" ".join(chunk))

    return rows


def parse_rows_to_df(rows: list[str]) -> pd.DataFrame:
    out = []
    seq = 0
    for r in rows:
        mdd = re.search(r"\b(\d{1,2}/\d{1,2})\b", r)
        my = re.search(r"/(20\d{2})\b", r)
        if not mdd or not my:
            continue
        fecha_str = f"{mdd.group(1)}/{my.group(1)}"
        fecha = pd.to_datetime(fecha_str, dayfirst=True, errors="coerce")
        if pd.isna(fecha):
            continue

        monies = list(MONEY_RE.finditer(r))
        if len(monies) < 2:
            continue

        saldo = normalize_money(monies[-1].group(0))
        importe = normalize_money(monies[-2].group(0))

        tail = r.split(mdd.group(1), 1)[-1]
        mint = INT_RE.search(tail)
        comprobante = mint.group(0) if mint else ""

        idx_imp_start = monies[-2].start()
        if comprobante:
            pos = r.find(comprobante)
            concept_slice = r[pos + len(comprobante): idx_imp_start]
        else:
            pos = r.find(mdd.group(1))
            concept_slice = r[pos + len(mdd.group(1)): idx_imp_start]

        concepto = concept_slice.replace("$", " ").replace("/" + my.group(1), " ").strip()
        concepto = " ".join(concepto.split())

        seq += 1
        out.append(
            {
                "fecha": fecha,
                "comprobante": comprobante,
                "concepto": concepto,
                "importe": float(importe),
                "saldo": float(saldo),
                "orden": seq,
            }
        )

    return pd.DataFrame(out)


# ---------------- clasificación ----------------
RE_IVA_BASE = re.compile(r"\bI\.?V\.?A\.?\s*BASE\b|\bIVA\s*BASE\b", re.IGNORECASE)
RE_RG2408 = re.compile(r"RG\.?\s*2408", re.IGNORECASE)
RE_LEY25413 = re.compile(r"LEY\s*25413|GRAVAMEN\s+LEY\s*25413|IMPTRANS|IMP\.\s*S/(DEB|CRED)", re.IGNORECASE)


def clasificar(concepto: str) -> str:
    c = (concepto or "")
    if RE_IVA_BASE.search(c):
        return "IVA BASE 21%"
    if RE_RG2408.search(c) or "RETEN" in c.upper():
        return "Percepciones IVA RG 2408"
    if RE_LEY25413.search(c):
        return "Ley 25.413"
    return "Otros"


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
rows = merge_bna_plus_rows(lines)
df = parse_rows_to_df(rows)

if df.empty:
    st.error("No se detectaron movimientos. Este PDF tiene un layout distinto o está escaneado.")
    st.stop()

df["comp_num"] = pd.to_numeric(df["comprobante"], errors="coerce")
df = df.sort_values(["fecha","comp_num","orden"]).reset_index(drop=True)

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
# - último movimiento: mayor FECHA y, si coincide, mayor COMPROBANTE
# - primer movimiento: menor FECHA y, si coincide, menor COMPROBANTE
df = df.sort_values(["fecha","comp_num","orden"]).reset_index(drop=True)

# BNA+ (según tu regla):
# - El PRIMER saldo que aparece (fecha más antigua) ES el "saldo anterior" del período.
# - El saldo final NO aparece en el PDF: se infiere desde el ÚLTIMO saldo que aparece
#   (previo al último movimiento) aplicando el ÚLTIMO importe.
saldo_anterior = float(df["saldo"].iloc[0])

ultimo_saldo_aparece = float(df["saldo"].iloc[-1])
ultimo_importe = float(df["importe"].iloc[-1])

saldo_final_inferido = ultimo_saldo_aparece + ultimo_importe  # aplica +/-

total_debitos = float(df["debito"].sum())
total_creditos = float(df["credito"].sum())

# Conciliación: saldo_final_calculado desde saldo anterior + sum(importes)
saldo_final_calculado = saldo_anterior + float(df["importe"].sum())
diferencia = saldo_final_calculado - saldo_final_inferido
cuadra = abs(diferencia) < 0.01

# Mostrar sin truncar
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.metric("Saldo anterior (1er saldo)", f"$ {fmt_ar(saldo_anterior)}")
with r1c2:
    st.metric("Total débitos (–)", f"$ {fmt_ar(total_debitos)}")
with r1c3:
    st.metric("Total créditos (+)", f"$ {fmt_ar(total_creditos)}")

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.metric("Último saldo que aparece", f"$ {fmt_ar(ultimo_saldo_aparece)}")
with r2c2:
    st.metric("Último importe", f"$ {fmt_ar(ultimo_importe)}")
with r2c3:
    st.metric("Saldo final inferido", f"$ {fmt_ar(saldo_final_inferido)}")

r3c1, r3c2 = st.columns(2)
with r3c1:
    st.metric("Saldo final calculado", f"$ {fmt_ar(saldo_final_calculado)}")
with r3c2:
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
        df.drop(columns=["comp_num"], errors="ignore").to_excel(writer, index=False, sheet_name="Movimientos")
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
    csv_bytes = df.drop(columns=["comp_num"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
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

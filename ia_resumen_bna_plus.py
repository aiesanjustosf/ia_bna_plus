# ia_resumen_bna_plus.py
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
ASSETS = HERE / "assets"
LOGO = ASSETS / "logo_aie.png"
FAVICON = ASSETS / "favicon-aie.ico"

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
      .block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 2rem; }
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


# ---------------- regex ----------------
DATE_DDMM_RE = re.compile(r"^(\d{1,2})/(\d{1,2})\b")          # dd/mm al inicio de bloque
YEAR_RE = re.compile(r"\b(20\d{2})\b")                        # 20xx
COMP_RE = re.compile(r"\b(\d{3,10})\b")                       # comprobante

# Importe argentino. Importante: NO usar "-" final como signo, porque BNA pega saldos:
# ejemplo real: -11.220.141,18-18.920.898,65
MONEY_RE = re.compile(r"-?(?:\d{1,3}(?:\.\d{3})+|\d+),\d{2}")

# Clasificación mínima para resumen operativo
RE_IVA_BASE = re.compile(r"\bI\.?V\.?A\.?\s*BASE\b", re.IGNORECASE)
RE_RET_IVA_2408 = re.compile(r"\bRETEN\.?\s*I\.?V\.?A\.?\s*RG\.?\s*2408\b", re.IGNORECASE)
RE_LEY_25413 = re.compile(r"\bLEY\s*25\.?413\b|\bGRAVAMEN\s+LEY\s*25\.?413\b", re.IGNORECASE)
RE_COMISION = re.compile(r"\bCOMISI[ÓO]N\b", re.IGNORECASE)
RE_SIRCREB = re.compile(r"\bREG\.?\s*REC\.?\s*SIRCREB\b", re.IGNORECASE)


# ---------------- utils ----------------
def normalize_money(tok: str) -> float:
    """Normaliza importes argentinos: -5,28 ó 1.234,56"""
    if not tok:
        return np.nan
    tok = tok.strip().replace("−", "-")
    neg = tok.startswith("-")
    tok = tok.strip("-").strip()
    if "," not in tok:
        return np.nan
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


def _text_from_pdf(file_like) -> str:
    try:
        with pdfplumber.open(file_like) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        return ""


def extract_lines(file_like) -> list[str]:
    out: list[str] = []
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for raw in txt.splitlines():
                ln = " ".join(raw.split())
                if not ln.strip():
                    continue
                # Sacamos encabezados/pie para que no contaminen el parser por bloques
                if ln.startswith("Jueves ") or ln.startswith("Últimos movimientos"):
                    continue
                if ln.startswith("Fecha Comprobante Concepto Monto Saldo"):
                    continue
                if ln.startswith("Página "):
                    continue
                out.append(ln)
    return out


def parse_bna_plus_movs(lines: list[str]) -> pd.DataFrame:
    """
    Parser BNA+ basado en bloques reales del PDF.

    Regla clave de Banco Nación PLUS:
    - La columna "Saldo" impresa es el saldo ANTES de aplicar el movimiento de esa línea.
    - El PDF viene del movimiento más nuevo al más viejo.
    - En algunos movimientos BNA/PDF pega o invierte Monto/Saldo visualmente.
      Por eso se elige cuál de los dos importes es saldo usando continuidad:
          saldo_de_esta_linea + importe_de_esta_linea = saldo_de_la_linea_anterior
    """
    blocks: list[list[str]] = []
    cur: list[str] | None = None

    for ln in lines:
        if DATE_DDMM_RE.match(ln):
            if cur:
                blocks.append(cur)
            cur = [ln]
        elif cur:
            cur.append(ln)

    if cur:
        blocks.append(cur)

    raw_rows = []
    for orden, block in enumerate(blocks, start=1):
        txt = " ".join(block)
        d = DATE_DDMM_RE.match(block[0])
        if not d:
            continue

        dd = int(d.group(1))
        mm = int(d.group(2))
        y = YEAR_RE.search(txt)
        year = int(y.group(1)) if y else 2026

        monies = [normalize_money(m.group(0)) for m in MONEY_RE.finditer(txt)]
        if len(monies) < 2:
            continue

        # Los dos últimos importes del bloque son Monto y Saldo, pero a veces el PDF los invierte.
        cand1 = float(monies[-2])
        cand2 = float(monies[-1])

        clean = txt
        clean = re.sub(r"^\d{1,2}/\d{1,2}\s*", "", clean)
        clean = re.sub(r"/20\d{2}", "", clean)
        clean = clean.replace("$", " ")
        clean = MONEY_RE.sub(" ", clean)
        clean = " ".join(clean.split())

        mc = COMP_RE.search(clean)
        comp = mc.group(1) if mc else ""
        concept = clean
        if mc:
            concept = (clean[: mc.start()] + clean[mc.end():]).strip()

        # Quita CUIT sueltos si quedaron en el concepto; el dato no hace falta para el resumen.
        concept = re.sub(r"\b\d{11}\b", "", concept)
        concept = " ".join(concept.split())

        raw_rows.append(
            {
                "fecha": pd.to_datetime(f"{year:04d}-{mm:02d}-{dd:02d}", errors="coerce"),
                "comprobante": comp,
                "concepto": concept,
                "cand1": cand1,
                "cand2": cand2,
                "orden": orden,
            }
        )

    if not raw_rows:
        return pd.DataFrame(columns=["fecha", "comprobante", "concepto", "importe", "saldo", "orden"])

    raw = pd.DataFrame(raw_rows)

    saldos = []
    importes = []

    for i, r in raw.iterrows():
        c1 = float(r["cand1"])
        c2 = float(r["cand2"])

        if i < len(raw) - 1:
            # El saldo de esta línea debe ser igual al resultado de la línea siguiente:
            # saldo_siguiente + importe_siguiente. Como en el crudo no sabemos cuál es cuál,
            # la suma cand1+cand2 de la línea siguiente da ese resultado.
            target = float(raw.loc[i + 1, "cand1"] + raw.loc[i + 1, "cand2"])

            c1_es_saldo = abs(c1 - target) <= 0.02
            c2_es_saldo = abs(c2 - target) <= 0.02

            if c1_es_saldo and not c2_es_saldo:
                saldo, importe = c1, c2
            elif c2_es_saldo and not c1_es_saldo:
                saldo, importe = c2, c1
            else:
                # Fallback: en la mayoría de las líneas el último importe es saldo.
                saldo, importe = c2, c1
        else:
            # Último movimiento del período: no hay línea siguiente para validar.
            # En BNA+ normalmente el último importe impreso es el saldo.
            saldo, importe = c2, c1

        saldos.append(float(saldo))
        importes.append(float(importe))

    raw["saldo"] = saldos
    raw["importe"] = importes

    df = raw[["fecha", "comprobante", "concepto", "importe", "saldo", "orden"]].copy()

    df["comprobante"] = df["comprobante"].astype(str).str.strip()
    df["concepto"] = df["concepto"].astype(str).str.strip()
    df["comp_num"] = pd.to_numeric(df["comprobante"], errors="coerce")

    # NO ordenar por fecha/comprobante: el orden del PDF es imprescindible para conciliar.
    return df.reset_index(drop=True)


def resumen_operativo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen Operativo BNA+ (según tus reglas):
    - I.V.A BASE (21%): se toma como IVA 21% y el neto (COMISIÓN) se calcula como IVA/0.21
    - RETEN. I.V.A. RG.2408: percepciones IVA (valor absoluto de débitos)
    - SIRCREB: recaudación bancaria de IIBB (valor absoluto de débitos)
    - LEY 25413: S/DEB + S/CRED neto
    """
    if df.empty:
        return pd.DataFrame(columns=["Concepto", "Importe"])

    iva_base = float(df.loc[df["concepto"].str.contains(RE_IVA_BASE, na=False), "importe"].sum() or 0.0)
    iva_base_abs = abs(iva_base)

    neto_com_21 = round(iva_base_abs / 0.21, 2) if iva_base_abs else 0.0

    ret_iva = float(df.loc[df["concepto"].str.contains(RE_RET_IVA_2408, na=False), "importe"].sum() or 0.0)
    ret_iva_abs = abs(ret_iva)

    sircreb = float(df.loc[df["concepto"].str.contains(RE_SIRCREB, na=False), "importe"].sum() or 0.0)
    sircreb_abs = abs(sircreb)

    ley = float(df.loc[df["concepto"].str.contains(RE_LEY_25413, na=False), "importe"].sum() or 0.0)
    ley_neto_gasto = -ley  # gasto positivo si el neto fue débito

    out = [
        ["COMISIÓN (NETO 21%)", neto_com_21],
        ["I.V.A BASE (21%)", iva_base_abs],
        ["RETEN. I.V.A. RG.2408", ret_iva_abs],
        ["SIRCREB", sircreb_abs],
        ["GRAVAMEN LEY 25.413 (NETO)", float(ley_neto_gasto)],
    ]
    total = sum(x[1] for x in out)
    out.append(["TOTAL", float(total)])

    return pd.DataFrame(out, columns=["Concepto", "Importe"])


# ---------------- UI principal ----------------
uploaded = st.file_uploader("Subí un PDF del resumen (Banco Nación PLUS)", type=["pdf"])
if uploaded is None:
    st.info("La app no almacena datos. Procesamiento local en memoria.")
    st.stop()

data = uploaded.read()
txt_full = _text_from_pdf(io.BytesIO(data)).strip()
if not txt_full:
    st.error(
        "No se pudo leer texto del PDF. "
        "Este resumen parece estar escaneado (solo imagen). "
        "La herramienta solo funciona con PDFs descargados del home banking, "
        "donde el texto sea seleccionable."
    )
    st.stop()

lines = extract_lines(io.BytesIO(data))
df = parse_bna_plus_movs(lines)

if df.empty:
    st.error("No se detectaron movimientos.")
    st.stop()

# Débito/Crédito desde IMPORTE (regla BNA+)
df["debito"] = np.where(df["importe"] < 0, -df["importe"], 0.0)
df["credito"] = np.where(df["importe"] > 0, df["importe"], 0.0)

# ---------------- Conciliación ----------------
st.subheader("Conciliación bancaria")

# Reglas BNA+:
# - El PDF viene del movimiento más nuevo al más viejo.
# - La columna "Saldo" es el saldo ANTES del movimiento.
# - Saldo anterior del período: saldo del último movimiento impreso.
# - Saldo final inferido: saldo del primer movimiento + importe del primer movimiento.
saldo_anterior = float(df["saldo"].iloc[-1])
saldo_final_inferido = float(df["saldo"].iloc[0] + df["importe"].iloc[0])

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

# ---------------- Resumen Operativo ----------------
st.subheader("Resumen Operativo: Registración Módulo IVA")
df_ro = resumen_operativo(df)
df_ro_view = df_ro.copy()
df_ro_view["Importe"] = df_ro_view["Importe"].map(fmt_ar)
st.dataframe(df_ro_view, use_container_width=True, hide_index=True)

# ---------------- Detalle de movimientos ----------------
st.subheader("Detalle de movimientos")

df_view = df.drop(columns=["comp_num"], errors="ignore").copy()
for c in ["importe", "saldo", "debito", "credito"]:
    if c in df_view.columns:
        df_view[c] = df_view[c].map(fmt_ar)

st.dataframe(df_view, use_container_width=True, hide_index=True)

# ---------------- Descargas ----------------
st.subheader("Descargas")

# Sufijos
first_date = df["fecha"].dropna().min()
last_date = df["fecha"].dropna().max()
date_suffix = ""
if pd.notna(first_date) and pd.notna(last_date):
    date_suffix = f"_{first_date.strftime('%Y%m%d')}_{last_date.strftime('%Y%m%d')}"

# Excel
try:
    import xlsxwriter  # noqa: F401

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.drop(columns=["comp_num"], errors="ignore").to_excel(writer, index=False, sheet_name="Movimientos")
        df_ro.to_excel(writer, index=False, sheet_name="Resumen_Operativo")

        wb = writer.book
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        date_fmt = wb.add_format({"num_format": "dd/mm/yyyy"})

        ws = writer.sheets["Movimientos"]
        for idx, col in enumerate(df.drop(columns=["comp_num"], errors="ignore").columns):
            width = min(max(len(str(col)), 12) + 2, 48)
            ws.set_column(idx, idx, width)

        # formatos
        cols_num = ["importe", "saldo", "debito", "credito"]
        for colname in cols_num:
            if colname in df.columns:
                j = df.drop(columns=["comp_num"], errors="ignore").columns.get_loc(colname)
                ws.set_column(j, j, 18, money_fmt)

        if "fecha" in df.columns:
            j = df.drop(columns=["comp_num"], errors="ignore").columns.get_loc("fecha")
            ws.set_column(j, j, 14, date_fmt)

        ws2 = writer.sheets["Resumen_Operativo"]
        ws2.set_column(0, 0, 40)
        ws2.set_column(1, 1, 18, money_fmt)

    st.download_button(
        "📥 Descargar Excel",
        data=output.getvalue(),
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

# PDF Resumen Operativo
if REPORTLAB_OK:
    try:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title="Resumen Operativo - BNA PLUS")
        styles = getSampleStyleSheet()

        elems = [
            Paragraph("Resumen Operativo: Registración Módulo IVA (BNA+)", styles["Title"]),
            Spacer(1, 10),
        ]

        datos = [["Concepto", "Importe"]]
        for _, r in df_ro.iterrows():
            datos.append([str(r["Concepto"]), fmt_ar(float(r["Importe"]))])

        tbl = Table(datos, colWidths=[360, 140])
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
        elems.append(Spacer(1, 14))
        elems.append(Paragraph("Herramienta para uso interno - AIE San Justo", styles["Normal"]))

        doc.build(elems)

        st.download_button(
            "📄 Descargar PDF – Resumen Operativo",
            data=pdf_buf.getvalue(),
            file_name=f"Resumen_Operativo_BNA_PLUS{date_suffix}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.info(f"No se pudo generar el PDF del Resumen Operativo: {e}")

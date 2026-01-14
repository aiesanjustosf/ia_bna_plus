# IA Resumen Bancario – Banco Nación PLUS
# Herramienta para uso interno - AIE San Justo

import io
import re
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# ---------------- UI ----------------
HERE = Path(__file__).parent
LOGO = HERE / "assets/logo_aie.png"
FAVICON = HERE / "assets/favicon-aie.ico"

st.set_page_config(
    page_title="IA Resumen Bancario – BNA PLUS",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="centered",
)

if LOGO.exists():
    st.image(str(LOGO), width=200)

st.title("IA Resumen Bancario – Banco Nación PLUS")

st.markdown(
    """
    <style>
    .block-container { max-width: 900px; padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- deps ----------------
try:
    import pdfplumber
except Exception as e:
    st.error(f"No se pudo importar pdfplumber: {e}")
    st.stop()

try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------------- regex ----------------
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
MONEY_RE = re.compile(r"-?\d{1,3}(?:\.\d{3})*,\d{2}")

# ---------------- utils ----------------
def parse_money(txt):
    return float(txt.replace(".", "").replace(",", "."))

def fmt_ar(n):
    if n is None or np.isnan(n):
        return "—"
    return f"{n:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")

# ---------------- extracción ----------------
def extract_lines(file_like):
    lines = []
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for l in txt.splitlines():
                l = " ".join(l.split())
                if l:
                    lines.append(l)
    return lines

def parse_movimientos(lines):
    rows = []
    for ln in lines:
        d = DATE_RE.search(ln)
        m = MONEY_RE.findall(ln)
        if not d or not m:
            continue

        importe = parse_money(m[-1])
        concepto = ln[d.end(): ln.rfind(m[-1])].strip()

        rows.append({
            "fecha": pd.to_datetime(d.group(), dayfirst=True),
            "concepto": concepto,
            "importe": importe,
        })
    return pd.DataFrame(rows)

# ---------------- clasificación ----------------
def clasificar(desc):
    u = desc.upper()

    if "IVA BASE" in u or "I.V.A BASE" in u:
        return "IVA BASE 21%"
    if "RG 2408" in u or "RETEN" in u:
        return "Percepciones IVA RG 2408"
    if "25413" in u or "GRAVAMEN" in u or "IMPTRANS" in u:
        return "Ley 25.413"

    return "Otros"

# ---------------- saldos inversos ----------------
def reconstruir_saldos(df, saldo_final):
    df = df.sort_values("fecha").reset_index(drop=True)
    saldos = [saldo_final]

    for imp in reversed(df["importe"].iloc[1:].tolist()):
        saldos.append(saldos[-1] - imp)

    df["saldo"] = list(reversed(saldos))
    return df

# ---------------- resumen operativo ----------------
def resumen_operativo(df):
    iva_base = df.loc[df["clasificacion"] == "IVA BASE 21%", "importe"].sum()
    neto_comision = round(iva_base / 0.21, 2) if iva_base else 0.0

    percep_iva = df.loc[df["clasificacion"] == "Percepciones IVA RG 2408", "importe"].sum()
    ley_25413 = df.loc[df["clasificacion"] == "Ley 25.413", "importe"].sum()

    return pd.DataFrame([
        ["Comisión (Neto 21%)", neto_comision],
        ["IVA BASE 21%", iva_base],
        ["Percepciones IVA RG 2408", percep_iva],
        ["Gravamen Ley 25.413 (neto)", ley_25413],
        ["TOTAL", neto_comision + iva_base + percep_iva + ley_25413],
    ], columns=["Concepto", "Importe"])

# ---------------- UI principal ----------------
uploaded = st.file_uploader("Subí el PDF BNA PLUS", type=["pdf"])
if not uploaded:
    st.stop()

data = uploaded.read()
lines = extract_lines(io.BytesIO(data))
df = parse_movimientos(lines)

if df.empty:
    st.error("No se detectaron movimientos.")
    st.stop()

df["clasificacion"] = df["concepto"].apply(clasificar)

# saldo final calculado por acumulado
saldo_final = df["importe"].cumsum().iloc[-1]
df = reconstruir_saldos(df, saldo_final)

df["debito"] = df["importe"].where(df["importe"] < 0, 0).abs()
df["credito"] = df["importe"].where(df["importe"] > 0, 0)

# ---------------- conciliación ----------------
saldo_inicial = df["saldo"].iloc[0]
total_deb = df["debito"].sum()
total_cred = df["credito"].sum()
saldo_calc = saldo_inicial + total_cred - total_deb
dif = saldo_calc - saldo_final

st.subheader("Conciliación bancaria")
c1, c2, c3 = st.columns(3)
c1.metric("Saldo inicial (calc.)", f"$ {fmt_ar(saldo_inicial)}")
c2.metric("Saldo final", f"$ {fmt_ar(saldo_final)}")
c3.metric("Diferencia", f"$ {fmt_ar(dif)}")

# ---------------- resumen operativo ----------------
st.subheader("Resumen Operativo – Módulo IVA")
resumen = resumen_operativo(df)
resumen["Importe_fmt"] = resumen["Importe"].map(fmt_ar)
st.dataframe(resumen[["Concepto", "Importe_fmt"]], use_container_width=True)

# ---------------- grilla movimientos ----------------
st.subheader("Detalle de movimientos")
df_view = df.copy()
for c in ["importe", "debito", "credito", "saldo"]:
    df_view[c] = df_view[c].map(fmt_ar)

st.dataframe(
    df_view[["fecha", "concepto", "debito", "credito", "saldo", "clasificacion"]],
    use_container_width=True,
)

# ---------------- export Excel ----------------
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Movimientos")
    resumen.to_excel(writer, index=False, sheet_name="Resumen_Operativo")

st.download_button(
    "📥 Descargar Excel",
    data=buf.getvalue(),
    file_name="resumen_bna_plus.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ---------------- PDF resumen ----------------
if REPORTLAB_OK:
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4)
    styles = getSampleStyleSheet()

    elems = [
        Paragraph("Resumen Operativo – Banco Nación PLUS", styles["Title"]),
        Spacer(1, 12),
    ]

    tbl_data = [["Concepto", "Importe"]] + [
        [r["Concepto"], fmt_ar(r["Importe"])] for _, r in resumen.iterrows()
    ]

    tbl = Table(tbl_data, colWidths=[340, 120])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]))

    elems.append(tbl)
    doc.build(elems)

    st.download_button(
        "📄 Descargar PDF Resumen Operativo",
        data=pdf_buf.getvalue(),
        file_name="Resumen_Operativo_BNA_PLUS.pdf",
        mime="application/pdf",
    )

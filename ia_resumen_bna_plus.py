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
DATE_DDMM_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")          # dd/mm (sin año)
YEAR_RE = re.compile(r"\b(20\d{2})\b")                        # 20xx
COMP_RE = re.compile(r"\b(\d{3,10})\b")                       # comprobante (número)
# Importe con signo: -5,28 o 5,28-
MONEY_RE = re.compile(r"(?<!\S)-?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2}-?(?!\S)")

# Clasificación mínima para resumen operativo
RE_IVA_BASE = re.compile(r"\bI\.?V\.?A\.?\s*BASE\b", re.IGNORECASE)
RE_RET_IVA_2408 = re.compile(r"\bRETEN\.?\s*I\.?V\.?A\.?\s*RG\.?\s*2408\b", re.IGNORECASE)
RE_LEY_25413 = re.compile(r"\bLEY\s*25\.?413\b|\bGRAVAMEN\s+LEY\s*25\.?413\b", re.IGNORECASE)
RE_COMISION = re.compile(r"\bCOMISI[ÓO]N\b", re.IGNORECASE)


# ---------------- utils ----------------
def normalize_money(tok: str) -> float:
    """Normaliza importes argentinos, aceptando: -5,28   ó   5,28-   ó  1.234,56"""
    if not tok:
        return np.nan
    tok = tok.strip().replace("−", "-")
    neg = tok.startswith("-") or tok.endswith("-")
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
                if ln.strip():
                    out.append(ln)
    return out


def parse_bna_plus_movs(lines: list[str]) -> pd.DataFrame:
    """
    Parser robusto BNA+:
    - Movimientos pueden venir en 1 línea o partidos en 2-3 líneas.
    - Columnas resultantes: fecha, comprobante, concepto, importe, saldo, orden
    - FECHA: dd/mm + año detectado en el bloque. Si no se detecta, se usa el primer 20xx cercano.
    """
    rows = []
    orden = 0

    # Año "vigente" dentro del bloque de movimientos
    current_year = None
    for ln in lines:
        y = YEAR_RE.search(ln)
        if y:
            current_year = int(y.group(1))
            break

    def build_fecha(dd: int, mm: int, year: int | None) -> pd.Timestamp:
        yy = year if year else 2000
        return pd.to_datetime(f"{yy:04d}-{mm:02d}-{dd:02d}", errors="coerce")

    # Helpers para detectar movimiento en una sola línea:
    # Esperamos: dd/mm ... comprobante ... concepto ... importe ... saldo
    def try_parse_one_line(ln: str) -> dict | None:
        d = DATE_DDMM_RE.search(ln)
        if not d:
            return None
        monies = list(MONEY_RE.finditer(ln))
        if len(monies) < 2:
            return None

        dd = int(d.group(1))
        mm = int(d.group(2))

        # saldo: último money; importe: penúltimo money
        saldo = normalize_money(monies[-1].group(0))
        importe = normalize_money(monies[-2].group(0))

        # recorte para extraer comprobante y concepto
        left = ln[d.end(): monies[-2].start()].strip()
        # comprobante: primer número grande que aparezca
        mc = COMP_RE.search(left)
        comp = mc.group(1) if mc else ""
        concept = left
        # si encontramos comprobante, lo sacamos del concepto textual
        if mc:
            concept = (left[: mc.start()] + left[mc.end():]).strip()
            # limpia restos
            concept = " ".join(concept.split())

        return {
            "fecha": build_fecha(dd, mm, current_year),
            "comprobante": comp,
            "concepto": concept,
            "importe": float(importe),
            "saldo": float(saldo),
        }

    # Parser multi-línea (patrón típico):
    # L1: "30/12 $"  (o "30/12")
    # L2: "{comp} {concepto} $ {importe}"
    # L3: "/2025 $ {saldo}"  (a veces con espacios)
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]

        parsed = try_parse_one_line(ln)
        if parsed:
            orden += 1
            parsed["orden"] = orden
            rows.append(parsed)
            i += 1
            continue

        d = DATE_DDMM_RE.search(ln)
        if not d:
            # refresca año si aparece
            y = YEAR_RE.search(ln)
            if y:
                current_year = int(y.group(1))
            i += 1
            continue

        # Si hay fecha pero no es one-line, intentamos multi-line
        dd = int(d.group(1))
        mm = int(d.group(2))

        # buscar línea de detalle (i+1) y línea de saldo (i+2)
        if i + 2 >= n:
            i += 1
            continue

        ln2 = lines[i + 1]
        ln3 = lines[i + 2]

        # ln2 debe tener al menos 1 money (importe)
        m2 = list(MONEY_RE.finditer(ln2))
        m3 = list(MONEY_RE.finditer(ln3))
        y3 = YEAR_RE.search(ln3) or YEAR_RE.search(ln2) or YEAR_RE.search(ln)

        if y3:
            current_year = int(y3.group(1))

        if len(m2) >= 1 and len(m3) >= 1:
            # importe: último money de ln2
            importe = normalize_money(m2[-1].group(0))
            # saldo: último money de ln3
            saldo = normalize_money(m3[-1].group(0))

            # comprobante + concepto: parte izquierda de ln2 hasta el money
            left2 = ln2[: m2[-1].start()].strip()
            mc = COMP_RE.search(left2)
            comp = mc.group(1) if mc else ""
            concept = left2
            if mc:
                concept = (left2[: mc.start()] + left2[mc.end():]).strip()
                concept = " ".join(concept.split())

            # Validación básica
            if not np.isnan(importe) and not np.isnan(saldo):
                orden += 1
                rows.append(
                    {
                        "fecha": build_fecha(dd, mm, current_year),
                        "comprobante": comp,
                        "concepto": concept,
                        "importe": float(importe),
                        "saldo": float(saldo),
                        "orden": orden,
                    }
                )
                i += 3
                continue

        # si falló el multi-line, avanzamos
        i += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalizaciones mínimas
    df["comprobante"] = df["comprobante"].astype(str).str.strip()
    df["concepto"] = df["concepto"].astype(str).str.strip()
    df["comp_num"] = pd.to_numeric(df["comprobante"], errors="coerce")

    # Orden final (clave para "último movimiento")
    df = df.sort_values(["fecha", "comp_num", "orden"]).reset_index(drop=True)
    return df


def resumen_operativo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen Operativo BNA+ (según tus reglas):
    - I.V.A BASE (21%): se toma como IVA 21% y el neto (COMISIÓN) se calcula como IVA/0.21
      (solo si existe COMISIÓN asociable; práctica: neto = IVA/0.21)
    - RETEN. I.V.A. RG.2408: percepciones IVA (se suman en valor absoluto de débitos)
    - LEY 25413: sumar S/DEB y S/CRED en un mismo neto:
      se suma por signo del IMPORTE (si hay acreditaciones/devoluciones, restan)
    """
    if df.empty:
        return pd.DataFrame(columns=["Concepto", "Importe"])

    # IVA BASE: en BNA+ normalmente aparece como débito (importe negativo)
    iva_base = float(df.loc[df["concepto"].str.contains(RE_IVA_BASE), "importe"].sum() or 0.0)
    iva_base_abs = abs(iva_base)  # se muestra como gasto/IVA

    # Neto comisión 21: inversa IVA/0.21
    neto_com_21 = round(iva_base_abs / 0.21, 2) if iva_base_abs else 0.0

    # Reten IVA RG2408: suma en valor absoluto si vienen negativos
    ret_iva = float(df.loc[df["concepto"].str.contains(RE_RET_IVA_2408), "importe"].sum() or 0.0)
    ret_iva_abs = abs(ret_iva)

    # Ley 25413 neto (S/DEB + S/CRED, devoluciones restan por signo)
    ley = float(df.loc[df["concepto"].str.contains(RE_LEY_25413), "importe"].sum() or 0.0)
    ley_abs = abs(ley) if ley < 0 else ley  # si queda neto negativo, lo mostramos como negativo? preferimos neto tal cual
    # Para registración como gasto neto: normalmente debita (negativo). Si hay devoluciones (positivas), restan.
    ley_neto_gasto = -ley  # gasto positivo si ley fue negativo neto

    out = [
        ["COMISIÓN (NETO 21%)", neto_com_21],
        ["I.V.A BASE (21%)", iva_base_abs],
        ["RETEN. I.V.A. RG.2408", ret_iva_abs],
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

# Reglas BNA+ (según tu criterio operativo):
# - Saldo anterior: primer saldo impreso del período.
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

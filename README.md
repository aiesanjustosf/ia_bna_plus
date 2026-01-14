# IA Resumen Bancario – Banco Nación PLUS

Aplicación Streamlit para procesar resúmenes bancarios del **Banco Nación (formato PLUS)**.

## Características
- Importe con signo (una sola columna)
- Reconstrucción de saldos en inversa (sin saldo anterior)
- Conciliación bancaria automática
- Resumen Operativo (IVA / RG 2408 / Ley 25.413 neta)
- Grilla completa de movimientos
- Exportación a Excel
- PDF del Resumen Operativo

## Uso
```bash
pip install -r requirements.txt
streamlit run ia_resumen_bna_plus.py

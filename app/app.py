"""Aplicación web: ¿a quién llamo a tutoría esta semana?

    streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.datos import ErrorValidacion, leer_archivo, validar_datos  # noqa: E402
from src.prediccion import alumnos_a_citar, cargar_modelo, predecir  # noqa: E402

TAMANO_LOTE = 200  # filas por lote, para mostrar avance en archivos grandes

st.set_page_config(page_title="Alumnos en riesgo", page_icon="🎓", layout="wide")


@st.cache_resource
def _modelo():
    return cargar_modelo()


def _excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


st.title("🎓 ¿A quién llamo a tutoría esta semana?")
st.write(
    "Suba el archivo semanal de alumnos (**.xlsx** o **.csv**). Debe tener las columnas "
    f"`{config.COLUMNA_ID}`, " + ", ".join(f"`{c}`" for c in config.VARIABLES) + "."
)

try:
    modelo, metadata = _modelo()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

archivo = st.file_uploader("Archivo de alumnos", type=["xlsx", "csv"])
if archivo is None:
    st.info("Estado: esperando archivo.")
    st.stop()

barra = st.progress(0, text="Cargando archivo…")
with st.status("Cargando archivo…", expanded=True) as estado:
    try:
        st.write("📂 Leyendo el archivo…")
        df = leer_archivo(archivo, archivo.name)
        barra.progress(15, text="Validando datos…")

        estado.update(label="Validando datos…")
        st.write(f"🔎 Validando {len(df):,} filas…")
        df = validar_datos(df)
        barra.progress(30, text="Calculando riesgo…")

        estado.update(label="Calculando riesgo…")
        st.write("🧮 Calculando la probabilidad de riesgo de cada alumno…")
        partes = []
        for inicio in range(0, len(df), TAMANO_LOTE):
            partes.append(predecir(df.iloc[inicio:inicio + TAMANO_LOTE], modelo, metadata, validar=False))
            avance = min(1.0, (inicio + TAMANO_LOTE) / len(df))
            barra.progress(30 + int(65 * avance),
                           text=f"Calculando riesgo… {min(inicio + TAMANO_LOTE, len(df)):,} de {len(df):,}")
        resultado = (pd.concat(partes)
                     .sort_values("probabilidad_riesgo", ascending=False, kind="stable")
                     .reset_index(drop=True))
        barra.progress(100, text="Listo")
        estado.update(label="Listo ✅", state="complete", expanded=False)
    except ErrorValidacion as exc:
        barra.progress(100, text="Error")
        estado.update(label="Error: el archivo tiene problemas", state="error", expanded=True)
        st.error("No se pudo procesar el archivo. Corrija lo siguiente y vuelva a subirlo:")
        for err in exc.errores:
            st.write(f"- {err}")
        st.stop()

citar = alumnos_a_citar(resultado)
c1, c2, c3 = st.columns(3)
c1.metric("Alumnos analizados", f"{len(resultado):,}")
c2.metric("Citar a tutoría", f"{len(citar):,}")
c3.metric("Umbral de riesgo", f"{metadata['umbral']:.0%}")

st.subheader("Alumnos ordenados por riesgo")
solo_citar = st.toggle("Mostrar solo alumnos a citar", value=False)
tabla = citar if solo_citar else resultado
st.dataframe(
    tabla,
    hide_index=True,
    width="stretch",
    column_config={
        config.COLUMNA_ID: "Alumno",
        "probabilidad_riesgo": st.column_config.ProgressColumn(
            "Probabilidad de riesgo", min_value=0.0, max_value=1.0, format="percent"),
        "citar_tutoria": "Citar a tutoría",
        "motivo_principal": "Motivo principal",
    },
)

st.subheader("Descargar lista de alumnos a citar")
d1, d2 = st.columns(2)
d1.download_button("⬇️ Descargar Excel", _excel(citar), "alumnos_a_citar.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   disabled=citar.empty)
d2.download_button("⬇️ Descargar CSV", citar.to_csv(index=False).encode("utf-8-sig"),
                   "alumnos_a_citar.csv", "text/csv", disabled=citar.empty)

with st.expander("Acerca del modelo"):
    st.write(
        f"Modelo: **{metadata['nombre']}**, entrenado con {sum(metadata['tamanos'].values())} alumnos "
        f"históricos. Se cita a todo alumno con probabilidad ≥ {metadata['umbral']:.0%}. "
        "El motivo principal es el factor que más aumenta el riesgo de ese alumno."
    )

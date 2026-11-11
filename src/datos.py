"""Carga de archivos (.csv / .xlsx), validación y división de datos."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


class ErrorValidacion(ValueError):
    """Error de validación con una lista de mensajes claros para el usuario."""

    def __init__(self, errores: list[str]):
        self.errores = errores
        super().__init__("\n".join(errores))


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def leer_archivo(origen: str | Path | BinaryIO, nombre: str | None = None) -> pd.DataFrame:
    """Lee un .csv o .xlsx. `origen` puede ser una ruta o un archivo abierto
    (p. ej. el que entrega Streamlit); en ese caso `nombre` da la extensión."""
    nombre = nombre or getattr(origen, "name", None) or str(origen)
    extension = Path(nombre).suffix.lower()
    try:
        if extension == ".csv":
            return pd.read_csv(origen)
        if extension in (".xlsx", ".xlsm"):
            return pd.read_excel(origen, engine="openpyxl")
    except Exception as exc:  # archivo dañado o ilegible
        raise ErrorValidacion([f"No se pudo leer el archivo: {exc}"]) from exc
    raise ErrorValidacion(
        [f"Formato no soportado ('{extension or 'sin extensión'}'). Use un archivo .xlsx o .csv."]
    )


def cargar_dataset(ruta: str | Path = config.RUTA_DATASET) -> pd.DataFrame:
    """Carga el dataset de entrenamiento (variables + objetivo)."""
    df = pd.read_csv(ruta)
    faltantes = [c for c in config.VARIABLES + [config.COLUMNA_OBJETIVO] if c not in df.columns]
    if faltantes:
        raise ErrorValidacion([f"Al dataset le faltan columnas: {', '.join(faltantes)}"])
    return df


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def _filas(indices) -> str:
    """Convierte índices de pandas en números de fila de Excel (encabezado = fila 1)."""
    filas = [str(i + 2) for i in indices]
    if len(filas) > 10:
        return ", ".join(filas[:10]) + f" … (+{len(filas) - 10} más)"
    return ", ".join(filas)


def validar_datos(df: pd.DataFrame, requiere_id: bool = True) -> pd.DataFrame:
    """Valida columnas y valores. Devuelve una copia limpia (tipos numéricos)
    o lanza ErrorValidacion con todos los problemas encontrados."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    errores: list[str] = []

    requeridas = ([config.COLUMNA_ID] if requiere_id else []) + config.VARIABLES
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        errores.append(f"Faltan columnas requeridas: {', '.join(faltantes)}.")
        raise ErrorValidacion(errores)

    if df.empty:
        raise ErrorValidacion(["El archivo no contiene alumnos (no hay filas de datos)."])

    if requiere_id:
        ids = df[config.COLUMNA_ID]
        vacios = ids.isna() | (ids.astype(str).str.strip() == "")
        if vacios.any():
            errores.append(f"'{config.COLUMNA_ID}' vacío en las filas: {_filas(df.index[vacios])}.")
        duplicados = ids[~vacios].duplicated(keep=False)
        if duplicados.any():
            errores.append(
                f"'{config.COLUMNA_ID}' repetido en las filas: {_filas(ids[~vacios].index[duplicados])}."
            )

    for col in config.VARIABLES:
        minimo, maximo, entero = config.RANGOS_VALIDOS[col]
        original = df[col]
        numerico = pd.to_numeric(original, errors="coerce")

        vacio = original.isna() | (original.astype(str).str.strip() == "")
        texto = numerico.isna() & ~vacio
        if vacio.any():
            errores.append(f"'{col}' tiene celdas vacías en las filas: {_filas(df.index[vacio])}.")
        if texto.any():
            errores.append(
                f"'{col}' tiene texto en lugar de números en las filas: {_filas(df.index[texto])}."
            )

        validos = numerico.notna()
        fuera = validos & ((numerico < minimo) | (numerico > maximo))
        if fuera.any():
            errores.append(
                f"'{col}' tiene valores fuera de rango ({minimo}–{maximo}) en las filas: "
                f"{_filas(df.index[fuera])}."
            )
        if entero:
            no_entero = validos & ~fuera & (numerico != numerico.round())
            if no_entero.any():
                errores.append(
                    f"'{col}' debe ser un número entero; revise las filas: {_filas(df.index[no_entero])}."
                )
        df[col] = numerico

    if errores:
        raise ErrorValidacion(errores)
    return df


# ---------------------------------------------------------------------------
# División
# ---------------------------------------------------------------------------
def dividir_datos(df: pd.DataFrame, semilla: int = config.SEMILLA):
    """División 80/20 estratificada con semilla fija.
    Devuelve X_entrenamiento, X_prueba, y_entrenamiento, y_prueba."""
    X = df[config.VARIABLES]
    y = df[config.COLUMNA_OBJETIVO]
    return train_test_split(
        X, y, test_size=config.TAMANO_PRUEBA, stratify=y, random_state=semilla
    )

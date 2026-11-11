"""Predicción con el modelo guardado (la app no reentrena)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.datos import validar_datos
from src.explicacion import SIN_MOTIVO, motivo_principal


def cargar_modelo(ruta_modelo: Path = config.RUTA_MODELO, ruta_metadata: Path = config.RUTA_METADATA):
    if not Path(ruta_modelo).exists():
        raise FileNotFoundError(
            "No hay un modelo entrenado. Ejecute primero: python -m src.entrenamiento"
        )
    modelo = joblib.load(ruta_modelo)
    metadata = json.loads(Path(ruta_metadata).read_text(encoding="utf-8"))
    return modelo, metadata


def predecir(df: pd.DataFrame, modelo, metadata: dict, validar: bool = True) -> pd.DataFrame:
    """Tabla de salida ordenada por probabilidad de riesgo (mayor a menor):
    id_alumno, probabilidad_riesgo, citar_tutoria, motivo_principal."""
    if validar:
        df = validar_datos(df)
    X = df[config.VARIABLES]
    probabilidad = modelo.predict_proba(X)[:, 1]
    citar = probabilidad >= metadata["umbral"]
    motivos = motivo_principal(modelo, X, metadata["algoritmo"])

    salida = pd.DataFrame({
        config.COLUMNA_ID: df[config.COLUMNA_ID].to_numpy(),
        "probabilidad_riesgo": probabilidad.round(3),
        "citar_tutoria": ["Sí" if c else "No" for c in citar],
        "motivo_principal": [m if c else SIN_MOTIVO for m, c in zip(motivos, citar)],
    })
    return salida.sort_values("probabilidad_riesgo", ascending=False, kind="stable").reset_index(drop=True)


def alumnos_a_citar(resultado: pd.DataFrame) -> pd.DataFrame:
    return resultado[resultado["citar_tutoria"] == "Sí"].reset_index(drop=True)

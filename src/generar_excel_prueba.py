"""Genera los Excel sintéticos de pruebas de producción (sin columna en_riesgo).

    python -m src.generar_excel_prueba                 # cantidades por defecto
    python -m src.generar_excel_prueba --normal 80 --grande 2000

Los alumnos sintéticos se crean remuestreando filas del dataset original y
agregando un pequeño ruido; así se conservan las distribuciones y las
relaciones entre variables. Los valores se recortan a los rangos observados.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.datos import cargar_dataset

# ruido relativo a la desviación estándar de cada variable
_RUIDO = 0.15


def _ids(n: int, inicio: int = 1) -> list[str]:
    return [f"A-{i:04d}" for i in range(inicio, inicio + n)]


def generar_alumnos(n: int, semilla: int = config.SEMILLA, inicio_id: int = 1) -> pd.DataFrame:
    """`n` alumnos sintéticos válidos con distribuciones similares al dataset."""
    rng = np.random.default_rng(semilla)
    base = cargar_dataset()[config.VARIABLES]
    muestra = base.sample(n=n, replace=True, random_state=semilla).reset_index(drop=True)
    for col in config.VARIABLES:
        _, _, entero = config.RANGOS_VALIDOS[col]
        ruido = rng.normal(0, _RUIDO * base[col].std(), size=n)
        valores = (muestra[col] + ruido).clip(base[col].min(), base[col].max())
        muestra[col] = valores.round().astype(int) if entero else valores.round(1)
    muestra.insert(0, config.COLUMNA_ID, _ids(n, inicio_id))
    return muestra


def casos_extremos() -> pd.DataFrame:
    """Valores en los límites del dataset original (todos válidos)."""
    filas = [
        # horas, asistencia, promedio, tareas, participación
        (0.0, 42.0, 5.1, 0, 0),      # todo en el mínimo observado
        (11.9, 100.0, 10.0, 10, 5),  # todo en el máximo observado
        (0.0, 100.0, 10.0, 10, 5),   # 0 horas, el resto perfecto
        (11.9, 100.0, 10.0, 0, 5),   # 0 tareas
        (11.9, 42.0, 10.0, 10, 5),   # asistencia mínima, el resto perfecto
        (11.9, 100.0, 5.1, 10, 5),   # promedio mínimo, el resto perfecto
        (6.0, 100.0, 7.5, 5, 2),     # 100 % de asistencia
        (6.0, 70.0, 7.5, 5, 0),      # participación 0
        (6.0, 70.0, 7.5, 5, 5),      # participación 5
        (0.0, 42.0, 5.1, 0, 5),      # todo mínimo salvo participación
        (5.5, 75.0, 10.0, 10, 3),    # promedio 10, todas las tareas
        (6.0, 70.0, 7.5, 5, 3),      # alumno promedio
    ]
    df = pd.DataFrame(filas, columns=config.VARIABLES)
    df.insert(0, config.COLUMNA_ID, _ids(len(df), 9001))
    return df


def errores_columnas(n: int = 20) -> pd.DataFrame:
    """Archivo válido al que le falta la columna `asistencia`."""
    return generar_alumnos(n, semilla=config.SEMILLA + 3).drop(columns=["asistencia"])


def errores_valores(n: int = 20) -> pd.DataFrame:
    """Valores fuera de rango, vacíos, texto en columnas numéricas e ID repetido."""
    df = generar_alumnos(n, semilla=config.SEMILLA + 4).astype(object)
    df.loc[0, "asistencia"] = 130          # fuera de rango
    df.loc[1, "promedio_anterior"] = -2    # fuera de rango
    df.loc[2, "tareas_entregadas"] = None  # vacío
    df.loc[3, "horas_estudio"] = "cinco"   # texto
    df.loc[4, "participacion_clase"] = 2.5  # no entero
    df.loc[5, "tareas_entregadas"] = 15    # fuera de rango
    df.loc[6, config.COLUMNA_ID] = df.loc[7, config.COLUMNA_ID]  # ID repetido
    return df


def generar_todos(destino: Path = config.DIR_PRUEBAS_PRODUCCION, n_normal: int = 60,
                  n_grande: int = 1200) -> dict[str, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    archivos = {
        "semana_normal.xlsx": generar_alumnos(n_normal, semilla=config.SEMILLA + 1),
        "casos_extremos.xlsx": casos_extremos(),
        "errores_columnas.xlsx": errores_columnas(),
        "errores_valores.xlsx": errores_valores(),
        "archivo_grande.xlsx": generar_alumnos(n_grande, semilla=config.SEMILLA + 2),
    }
    rutas = {}
    for nombre, df in archivos.items():
        ruta = destino / nombre
        df.to_excel(ruta, index=False, engine="openpyxl")
        rutas[nombre] = ruta
    return rutas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal", type=int, default=60, help="alumnos en semana_normal.xlsx")
    parser.add_argument("--grande", type=int, default=1200, help="alumnos en archivo_grande.xlsx")
    args = parser.parse_args()
    for nombre, ruta in generar_todos(n_normal=args.normal, n_grande=args.grande).items():
        print(f"{nombre}: {ruta}")

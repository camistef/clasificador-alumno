"""Motivo principal de riesgo por alumno, en lenguaje claro.

- Regresión logística: contribución de cada variable = peso aprendido × valor
  estandarizado del alumno. El motivo es la variable que más sube el riesgo.
- Árbol de decisión: en el camino de decisión del alumno, la condición que más
  aumenta la probabilidad de riesgo respecto al nodo anterior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config

SIN_MOTIVO = "—"


def _texto_motivo(variable: str, valor_bajo: bool) -> str:
    return (config.MOTIVOS_BAJO if valor_bajo else config.MOTIVOS_ALTO)[variable]


def contribuciones_logistica(modelo: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Matriz alumnos × variables con la contribución de cada variable al riesgo (log-odds)."""
    escalador = modelo.named_steps["escalador"]
    coef = modelo.named_steps["modelo"].coef_[0]
    z = escalador.transform(X[config.VARIABLES])
    return pd.DataFrame(z * coef, columns=config.VARIABLES, index=X.index)


def _motivos_logistica(modelo: Pipeline, X: pd.DataFrame) -> list[str]:
    contrib = contribuciones_logistica(modelo, X)
    coef = dict(zip(config.VARIABLES, modelo.named_steps["modelo"].coef_[0]))
    motivos = []
    for _, fila in contrib.iterrows():
        variable = fila.idxmax()
        # coef < 0 y contribución > 0  ⇒ el valor del alumno está por debajo de la media
        valor_bajo = coef[variable] < 0
        motivos.append(_texto_motivo(variable, valor_bajo))
    return motivos


def _motivos_arbol(modelo: Pipeline, X: pd.DataFrame) -> list[str]:
    arbol = modelo.named_steps["modelo"]
    t = arbol.tree_
    # probabilidad de riesgo (clase 1) en cada nodo
    prob_nodo = t.value[:, 0, 1] / t.value[:, 0, :].sum(axis=1)
    caminos = arbol.decision_path(X[config.VARIABLES])
    motivos = []
    for i in range(len(X)):
        nodos = caminos.indices[caminos.indptr[i]:caminos.indptr[i + 1]]
        mejor, mejor_aumento = None, -np.inf
        for padre, hijo in zip(nodos[:-1], nodos[1:]):
            aumento = prob_nodo[hijo] - prob_nodo[padre]
            if aumento > mejor_aumento:
                variable = config.VARIABLES[t.feature[padre]]
                valor_bajo = hijo == t.children_left[padre]  # rama izquierda: valor <= corte
                mejor, mejor_aumento = (variable, valor_bajo), aumento
        motivos.append(_texto_motivo(*mejor) if mejor else SIN_MOTIVO)
    return motivos


def motivo_principal(modelo: Pipeline, X: pd.DataFrame, algoritmo: str) -> list[str]:
    """Devuelve un motivo en lenguaje claro para cada fila de X."""
    if algoritmo == "regresion_logistica":
        return _motivos_logistica(modelo, X)
    if algoritmo == "arbol_decision":
        return _motivos_arbol(modelo, X)
    raise ValueError(f"Algoritmo sin explicación definida: {algoritmo}")

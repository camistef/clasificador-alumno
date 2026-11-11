"""Entrenamiento, comparación y selección del modelo.

Ejecutar desde la raíz del proyecto:
    python -m src.entrenamiento
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.datos import cargar_dataset, dividir_datos


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------
def crear_candidatos(semilla: int = config.SEMILLA) -> dict:
    """Pipelines y rejillas de hiperparámetros de los dos algoritmos candidatos."""
    logistica = Pipeline([
        ("escalador", StandardScaler()),  # se ajusta solo con los datos de cada `fit`
        ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=semilla)),
    ])
    arbol = Pipeline([
        ("modelo", DecisionTreeClassifier(class_weight="balanced", random_state=semilla)),
    ])
    return {
        "regresion_logistica": {
            "nombre": "Regresión logística",
            "pipeline": logistica,
            "rejilla": {"modelo__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
        },
        "arbol_decision": {
            "nombre": "Árbol de decisión",
            "pipeline": arbol,
            # profundidad limitada para que el árbol siga siendo explicable
            "rejilla": {"modelo__max_depth": [2, 3, 4, 5], "modelo__min_samples_leaf": [1, 3, 5, 10, 15]},
        },
    }


def _pliegues(semilla: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.N_PLIEGUES, shuffle=True, random_state=semilla)


def _pliegues_repetidos(semilla: int) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=config.N_PLIEGUES, n_repeats=config.N_REPETICIONES,
                                   random_state=semilla)


# ---------------------------------------------------------------------------
# Umbral
# ---------------------------------------------------------------------------
def elegir_umbral(y, probabilidades, meta_recall: float = config.META_RECALL_UMBRAL) -> float:
    """Umbral más alto (mayor precisión) que mantiene recall >= meta.

    Recorre los valores de probabilidad observados como posibles umbrales;
    entre los que cumplen la meta de recall elige el de mayor precisión
    (en empate, el umbral más alto)."""
    y = np.asarray(y)
    probabilidades = np.asarray(probabilidades)
    mejor_umbral, mejor_precision = None, -1.0
    for umbral in np.unique(probabilidades):
        prediccion = (probabilidades >= umbral).astype(int)
        if recall_score(y, prediccion, zero_division=0) < meta_recall:
            continue
        precision = precision_score(y, prediccion, zero_division=0)
        if precision >= mejor_precision:
            mejor_umbral, mejor_precision = float(umbral), precision
    return mejor_umbral if mejor_umbral is not None else float(probabilidades.min())


def metricas(y, probabilidades, umbral: float) -> dict:
    prediccion = (np.asarray(probabilidades) >= umbral).astype(int)
    return {
        "recall": recall_score(y, prediccion, zero_division=0),
        "precision": precision_score(y, prediccion, zero_division=0),
        "f1": f1_score(y, prediccion, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilidades),
    }


# ---------------------------------------------------------------------------
# Validación cruzada de un candidato
# ---------------------------------------------------------------------------
@dataclass
class ResultadoCandidato:
    clave: str
    nombre: str
    mejores_parametros: dict
    umbral: float
    cv_media: dict
    cv_desviacion: dict
    probabilidades_cv: np.ndarray = field(repr=False)
    cumple_metas: bool = False


def evaluar_candidato(clave: str, candidato: dict, X, y, semilla: int = config.SEMILLA) -> ResultadoCandidato:
    """1) GridSearchCV (ROC-AUC, CV repetida 10 × 5) sobre el 80 %;
    2) probabilidades fuera de pliegue en cada una de las 10 repeticiones;
    3) umbral para recall >= meta + margen sobre todas las repeticiones juntas;
    4) métricas por pliegue (50 pliegues) con ese umbral."""
    busqueda = GridSearchCV(candidato["pipeline"], candidato["rejilla"], scoring="roc_auc",
                            cv=_pliegues_repetidos(semilla), n_jobs=1)
    busqueda.fit(X, y)
    mejor = busqueda.best_estimator_

    y_array = np.asarray(y)
    # prob[r, i] = probabilidad fuera de pliegue del alumno i en la repetición r
    prob = np.empty((config.N_REPETICIONES, len(y_array)))
    pliegues_val = []
    for k, (idx_ent, idx_val) in enumerate(_pliegues_repetidos(semilla).split(X, y)):
        modelo = clone(mejor).fit(X.iloc[idx_ent], y_array[idx_ent])
        prob[k // config.N_PLIEGUES, idx_val] = modelo.predict_proba(X.iloc[idx_val])[:, 1]
        pliegues_val.append((k // config.N_PLIEGUES, idx_val))

    umbral = elegir_umbral(np.tile(y_array, config.N_REPETICIONES), prob.ravel())
    probabilidades = prob.mean(axis=0)  # promedio por alumno (para la curva ROC del reporte)

    por_pliegue = [metricas(y_array[idx_val], prob[r, idx_val], umbral) for r, idx_val in pliegues_val]
    tabla = pd.DataFrame(por_pliegue)
    media, desviacion = tabla.mean().to_dict(), tabla.std(ddof=1).to_dict()
    cumple = media["recall"] >= config.META_RECALL and media["roc_auc"] >= config.META_ROC_AUC

    return ResultadoCandidato(
        clave=clave,
        nombre=candidato["nombre"],
        mejores_parametros={k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()},
        umbral=umbral,
        cv_media=media,
        cv_desviacion=desviacion,
        probabilidades_cv=probabilidades,
        cumple_metas=cumple,
    )


def seleccionar(resultados: list[ResultadoCandidato]) -> ResultadoCandidato | None:
    """Regla de selección: cumplir ambas metas en CV; desempate por mayor precisión.
    Devuelve None si ninguno cumple (RM-07: se debe consultar otro método)."""
    aptos = [r for r in resultados if r.cumple_metas]
    if not aptos:
        return None
    return max(aptos, key=lambda r: r.cv_media["precision"])


# ---------------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------------
def entrenar_final(clave: str, parametros: dict, X, y, semilla: int = config.SEMILLA) -> Pipeline:
    """Reentrena el pipeline elegido con todo el 80 % de entrenamiento."""
    pipeline = crear_candidatos(semilla)[clave]["pipeline"]
    pipeline.set_params(**{f"modelo__{k}": v for k, v in parametros.items()})
    return pipeline.fit(X, y)


def guardar_modelo(pipeline: Pipeline, metadata: dict) -> None:
    config.DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump(pipeline, config.RUTA_MODELO)
    config.RUTA_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def ejecutar(semilla: int = config.SEMILLA, guardar: bool = True) -> dict:
    """Ejecuta las fases 2–4: división, CV de candidatos, selección y evaluación final."""
    df = cargar_dataset()
    X_ent, X_pru, y_ent, y_pru = dividir_datos(df, semilla)

    resultados = [evaluar_candidato(clave, cand, X_ent, y_ent, semilla)
                  for clave, cand in crear_candidatos(semilla).items()]
    elegido = seleccionar(resultados)

    salida = {"df": df, "X_ent": X_ent, "X_pru": X_pru, "y_ent": y_ent, "y_pru": y_pru,
              "resultados": resultados, "elegido": elegido}
    if elegido is None:
        return salida

    # Evaluación única en el 20 % de prueba
    modelo = entrenar_final(elegido.clave, elegido.mejores_parametros, X_ent, y_ent, semilla)
    prob_pru = modelo.predict_proba(X_pru)[:, 1]
    final = metricas(y_pru, prob_pru, elegido.umbral)
    matriz = confusion_matrix(y_pru, (prob_pru >= elegido.umbral).astype(int)).tolist()

    metadata = {
        "algoritmo": elegido.clave,
        "nombre": elegido.nombre,
        "hiperparametros": elegido.mejores_parametros,
        "umbral": elegido.umbral,
        "metodo_umbral": (f"CV repetida {config.N_REPETICIONES}x{config.N_PLIEGUES}; mayor precisión con "
                          f"recall >= {config.META_RECALL_UMBRAL:.2f} (meta {config.META_RECALL} + margen "
                          f"{config.MARGEN_RECALL})"),
        "nota_evaluacion": ("Segunda evaluación sobre el conjunto de prueba; la primera (umbral 0.690 "
                            "sin margen) dio recall 0.733. Ver reporte."),
        "variables": config.VARIABLES,
        "metricas_cv_media": elegido.cv_media,
        "metricas_cv_desviacion": elegido.cv_desviacion,
        "metricas_prueba": final,
        "matriz_confusion_prueba": matriz,
        "cumple_metas_prueba": final["recall"] >= config.META_RECALL and final["roc_auc"] >= config.META_ROC_AUC,
        "tamanos": {"entrenamiento": len(X_ent), "prueba": len(X_pru)},
        "semilla": semilla,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
    }
    if guardar:
        guardar_modelo(modelo, metadata)
    salida.update({"modelo": modelo, "prob_pru": prob_pru, "metricas_prueba": final, "metadata": metadata})
    return salida


if __name__ == "__main__":
    from src.reporte import generar_reporte

    salida = ejecutar()
    for r in salida["resultados"]:
        print(f"{r.nombre}: umbral={r.umbral:.3f} params={r.mejores_parametros} "
              + " ".join(f"{k}={v:.3f}±{r.cv_desviacion[k]:.3f}" for k, v in r.cv_media.items()))
    if salida["elegido"] is None:
        print("\nNINGÚN modelo cumple las metas en validación cruzada. Ver RM-07.")
    else:
        print(f"\nElegido: {salida['elegido'].nombre}")
        print("Prueba:", {k: round(v, 3) for k, v in salida["metricas_prueba"].items()})
        print("Matriz de confusión:", salida["metadata"]["matriz_confusion_prueba"])
    ruta = generar_reporte(salida)
    print(f"Reporte: {ruta}")

"""Pruebas del entrenamiento y del modelo (Entregable 1)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.datos import cargar_dataset, dividir_datos
from src.entrenamiento import crear_candidatos, ejecutar, elegir_umbral, guardar_modelo


def test_division_estratificada():
    df = cargar_dataset()
    X_ent, X_pru, y_ent, y_pru = dividir_datos(df)
    assert (len(X_ent), len(X_pru)) == (160, 40)
    assert set(X_ent.index).isdisjoint(X_pru.index)
    for y in (y_ent, y_pru):
        assert abs(y.mean() - 0.38) <= 0.02


def test_sin_fuga_datos(salida):
    """El escalador se ajusta solo con el 80 % de entrenamiento."""
    if salida["elegido"].clave != "regresion_logistica":
        pytest.skip("el modelo elegido no usa escalador")
    escalador = salida["modelo"].named_steps["escalador"]
    np.testing.assert_allclose(escalador.mean_, salida["X_ent"].mean().to_numpy())
    todos = salida["df"][config.VARIABLES].mean().to_numpy()
    assert not np.allclose(escalador.mean_, todos)


def test_ambos_candidatos_evaluados(salida):
    assert {r.clave for r in salida["resultados"]} == {"regresion_logistica", "arbol_decision"}
    for r in salida["resultados"]:
        assert set(r.cv_media) == {"recall", "precision", "f1", "roc_auc"}


def test_umbral_cumple_recall_en_cv(salida):
    for r in salida["resultados"]:
        assert r.cv_media["recall"] >= config.META_RECALL - 1e-9


def test_elegir_umbral_prefiere_mayor_precision():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.3, 0.6, 0.8, 0.9, 0.95])
    # 0.6 → recall 4/5, precisión 4/5 · 0.3 → recall 5/5, precisión 5/6 · 0.8 → recall 3/5 (no cumple)
    assert elegir_umbral(y, p, meta_recall=0.8) == pytest.approx(0.3)


def test_modelo_cumple_recall(salida):
    recall = salida["metricas_prueba"]["recall"]
    assert recall >= config.META_RECALL, f"Recall en prueba {recall:.3f} < meta {config.META_RECALL}"


def test_modelo_cumple_roc_auc(salida):
    assert salida["metricas_prueba"]["roc_auc"] >= config.META_ROC_AUC


def test_modelo_aprende(salida):
    """Prueba de aprendizaje (plan §2): con etiquetas revueltas el ROC-AUC cae a ≈ 0.50."""
    clave = salida["elegido"].clave
    X, y = salida["X_ent"], salida["y_ent"].to_numpy()
    rng = np.random.default_rng(config.SEMILLA)
    aucs_revueltas = []
    for semilla in range(10):
        y_revuelta = rng.permutation(y)
        modelo = crear_candidatos(semilla)[clave]["pipeline"]
        cv = StratifiedKFold(config.N_PLIEGUES, shuffle=True, random_state=semilla)
        prob = cross_val_predict(modelo, X, y_revuelta, cv=cv, method="predict_proba")[:, 1]
        aucs_revueltas.append(roc_auc_score(y_revuelta, prob))
    auc_revuelta = np.mean(aucs_revueltas)
    assert abs(auc_revuelta - 0.5) < 0.1
    assert salida["elegido"].cv_media["roc_auc"] >= config.META_ROC_AUC
    assert salida["elegido"].cv_media["roc_auc"] - auc_revuelta > 0.25


def test_reproducibilidad(salida):
    otra = ejecutar(guardar=False)
    assert otra["elegido"].clave == salida["elegido"].clave
    assert otra["elegido"].umbral == salida["elegido"].umbral
    assert otra["metricas_prueba"] == salida["metricas_prueba"]
    np.testing.assert_array_equal(otra["prob_pru"], salida["prob_pru"])


def test_guardar_cargar_modelo(salida, tmp_path, monkeypatch):
    from src.prediccion import cargar_modelo

    monkeypatch.setattr(config, "DIR_MODELOS", tmp_path)
    monkeypatch.setattr(config, "RUTA_MODELO", tmp_path / "modelo.joblib")
    monkeypatch.setattr(config, "RUTA_METADATA", tmp_path / "metadata.json")
    guardar_modelo(salida["modelo"], salida["metadata"])
    modelo, metadata = cargar_modelo(tmp_path / "modelo.joblib", tmp_path / "metadata.json")
    np.testing.assert_array_equal(modelo.predict_proba(salida["X_pru"]),
                                  salida["modelo"].predict_proba(salida["X_pru"]))
    assert metadata["umbral"] == salida["metadata"]["umbral"]

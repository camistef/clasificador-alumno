"""Pruebas de predicción, motivo principal y Excel de producción."""
import numpy as np
import pytest

from src import config
from src.datos import ErrorValidacion, cargar_dataset, dividir_datos, leer_archivo, validar_datos
from src.entrenamiento import entrenar_final
from src.explicacion import SIN_MOTIVO, motivo_principal
from src.generar_excel_prueba import casos_extremos, generar_alumnos, generar_todos
from src.prediccion import alumnos_a_citar, predecir


@pytest.fixture(scope="module")
def datos():
    return generar_alumnos(300)


@pytest.fixture(scope="module")
def resultado(salida, datos):
    return predecir(datos, salida["modelo"], salida["metadata"])


def test_salida_ordenada_y_columnas(resultado):
    assert list(resultado.columns) == [config.COLUMNA_ID, "probabilidad_riesgo",
                                       "citar_tutoria", "motivo_principal"]
    assert resultado["probabilidad_riesgo"].is_monotonic_decreasing
    assert resultado["probabilidad_riesgo"].between(0, 1).all()


def test_citar_segun_umbral(salida, datos):
    prob = salida["modelo"].predict_proba(datos[config.VARIABLES])[:, 1]
    esperados = set(datos.loc[prob >= salida["metadata"]["umbral"], config.COLUMNA_ID])
    r = predecir(datos, salida["modelo"], salida["metadata"])
    assert set(alumnos_a_citar(r)[config.COLUMNA_ID]) == esperados


def test_motivo_presente(resultado):
    citados = alumnos_a_citar(resultado)
    assert len(citados) > 0
    assert (citados["motivo_principal"].str.len() > 0).all()
    assert (citados["motivo_principal"] != SIN_MOTIVO).all()
    assert (resultado.loc[resultado.citar_tutoria == "No", "motivo_principal"] == SIN_MOTIVO).all()


def test_motivo_coherente_con_dato(salida, datos, resultado):
    """'Baja X' / 'Pocas X' solo si el valor del alumno está por debajo de la media de entrenamiento."""
    unido = resultado.merge(datos, on=config.COLUMNA_ID)
    media = salida["X_ent"].mean()
    for variable, motivo in config.MOTIVOS_BAJO.items():
        filas = unido[unido["motivo_principal"] == motivo]
        assert (filas[variable] < media[variable]).all(), motivo


def test_motivo_arbol():
    """La explicación también funciona si el modelo elegido fuera el árbol."""
    X_ent, X_pru, y_ent, _ = dividir_datos(cargar_dataset())
    arbol = entrenar_final("arbol_decision", {"max_depth": 3, "min_samples_leaf": 5}, X_ent, y_ent)
    motivos = motivo_principal(arbol, X_pru, "arbol_decision")
    assert len(motivos) == len(X_pru) and all(m and m != SIN_MOTIVO for m in motivos)


def test_cualquier_cantidad_de_alumnos(salida):
    for n in (1, 7, 1500):
        assert len(predecir(generar_alumnos(n), salida["modelo"], salida["metadata"])) == n


def test_excel_produccion(tmp_path, salida):
    rutas = generar_todos(tmp_path, n_normal=50, n_grande=1000)
    assert len(rutas) == 5
    base = salida["df"]
    for nombre in ("semana_normal.xlsx", "casos_extremos.xlsx", "archivo_grande.xlsx"):
        df = validar_datos(leer_archivo(rutas[nombre]))
        assert config.COLUMNA_OBJETIVO not in df.columns
        for col in config.VARIABLES:
            assert df[col].between(base[col].min(), base[col].max()).all(), (nombre, col)
    assert len(leer_archivo(rutas["archivo_grande.xlsx"])) >= 1000
    for nombre in ("errores_columnas.xlsx", "errores_valores.xlsx"):
        with pytest.raises(ErrorValidacion):
            validar_datos(leer_archivo(rutas[nombre]))


def test_casos_extremos_predicen(salida):
    r = predecir(casos_extremos(), salida["modelo"], salida["metadata"])
    assert np.isfinite(r["probabilidad_riesgo"]).all()

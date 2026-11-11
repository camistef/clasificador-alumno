"""Pruebas de carga y validación de archivos."""
import pandas as pd
import pytest

from src import config
from src.datos import ErrorValidacion, leer_archivo, validar_datos
from src.generar_excel_prueba import generar_alumnos


@pytest.fixture
def validos():
    return generar_alumnos(30)


def test_archivo_valido_pasa(validos):
    assert len(validar_datos(validos)) == 30


@pytest.mark.parametrize("columna", [config.COLUMNA_ID] + config.VARIABLES)
def test_validacion_columnas(validos, columna):
    with pytest.raises(ErrorValidacion, match=columna):
        validar_datos(validos.drop(columns=[columna]))


@pytest.mark.parametrize("columna, valor, texto", [
    ("asistencia", 130, "fuera de rango"),
    ("promedio_anterior", -1, "fuera de rango"),
    ("tareas_entregadas", 11, "fuera de rango"),
    ("tareas_entregadas", None, "vacías"),
    ("horas_estudio", "cinco", "texto"),
    ("participacion_clase", 2.5, "entero"),
])
def test_validacion_rangos(validos, columna, valor, texto):
    df = validos.astype(object)
    df.loc[3, columna] = valor  # índice 3 = fila 5 de Excel
    with pytest.raises(ErrorValidacion) as exc:
        validar_datos(df)
    assert any(columna in e and texto in e and "5" in e for e in exc.value.errores)


def test_validacion_id_repetido(validos):
    validos.loc[1, config.COLUMNA_ID] = validos.loc[0, config.COLUMNA_ID]
    with pytest.raises(ErrorValidacion, match="repetido"):
        validar_datos(validos)


def test_archivo_vacio(validos):
    with pytest.raises(ErrorValidacion, match="no contiene alumnos"):
        validar_datos(validos.iloc[0:0])


def test_formato_no_soportado(tmp_path):
    ruta = tmp_path / "datos.txt"
    ruta.write_text("x")
    with pytest.raises(ErrorValidacion, match="Formato no soportado"):
        leer_archivo(ruta)


def test_lectura_xlsx_y_csv(validos, tmp_path):
    validos.to_excel(tmp_path / "a.xlsx", index=False)
    validos.to_csv(tmp_path / "a.csv", index=False)
    desde_xlsx = validar_datos(leer_archivo(tmp_path / "a.xlsx"))
    desde_csv = validar_datos(leer_archivo(tmp_path / "a.csv"))
    pd.testing.assert_frame_equal(desde_xlsx, desde_csv, check_dtype=False)
    with open(tmp_path / "a.xlsx", "rb") as f:  # como lo entrega Streamlit
        pd.testing.assert_frame_equal(validar_datos(leer_archivo(f, "a.xlsx")), desde_xlsx)

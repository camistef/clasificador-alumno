import pytest

from src.entrenamiento import ejecutar


@pytest.fixture(scope="session")
def salida():
    """Ejecuta el flujo completo de entrenamiento una sola vez (sin sobrescribir modelos/)."""
    return ejecutar(guardar=False)

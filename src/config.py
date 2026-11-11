"""Configuración central del proyecto: columnas, rangos válidos, semilla y metas."""
from pathlib import Path

# --- Rutas ---------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "dataset" / "datos.csv"
DIR_MODELOS = RAIZ / "modelos"
RUTA_MODELO = DIR_MODELOS / "modelo.joblib"
RUTA_METADATA = DIR_MODELOS / "metadata.json"
DIR_REPORTES = RAIZ / "reportes"
DIR_PRUEBAS_PRODUCCION = RAIZ / "pruebas_produccion"

# --- Columnas ------------------------------------------------------------
COLUMNA_ID = "id_alumno"
COLUMNA_OBJETIVO = "en_riesgo"
VARIABLES = [
    "horas_estudio",
    "asistencia",
    "promedio_anterior",
    "tareas_entregadas",
    "participacion_clase",
]

# Rangos válidos (mínimo, máximo, ¿entero?). Son los límites lógicos de cada
# variable; un valor fuera de ellos es un error de captura.
RANGOS_VALIDOS = {
    "horas_estudio": (0.0, 80.0, False),
    "asistencia": (0.0, 100.0, False),
    "promedio_anterior": (0.0, 10.0, False),
    "tareas_entregadas": (0, 10, True),
    "participacion_clase": (0, 5, True),
}

# Nombres en lenguaje claro para mostrar al usuario
NOMBRES_CLAROS = {
    "horas_estudio": "horas de estudio",
    "asistencia": "asistencia",
    "promedio_anterior": "promedio anterior",
    "tareas_entregadas": "tareas entregadas",
    "participacion_clase": "participación en clase",
}

# Motivo cuando la variable empuja el riesgo hacia arriba por valor bajo / alto
MOTIVOS_BAJO = {
    "horas_estudio": "Pocas horas de estudio",
    "asistencia": "Baja asistencia",
    "promedio_anterior": "Promedio anterior bajo",
    "tareas_entregadas": "Pocas tareas entregadas",
    "participacion_clase": "Baja participación en clase",
}
MOTIVOS_ALTO = {
    "horas_estudio": "Muchas horas de estudio",
    "asistencia": "Asistencia alta",
    "promedio_anterior": "Promedio anterior alto",
    "tareas_entregadas": "Muchas tareas entregadas",
    "participacion_clase": "Alta participación en clase",
}

# --- Entrenamiento -------------------------------------------------------
SEMILLA = 42
TAMANO_PRUEBA = 0.20
N_PLIEGUES = 5
N_REPETICIONES = 10  # validación cruzada repetida 10 × 5 para un umbral más estable

# El umbral se elige para una meta interna de recall con margen sobre la meta
# oficial. Margen = 1 desviación estándar del recall entre pliegues observada
# en la validación cruzada (≈ 0.10), para no quedar justo en el límite.
MARGEN_RECALL = 0.10

# --- Metas ---------------------------------------------------------------
META_RECALL = 0.80
META_ROC_AUC = 0.80
META_RECALL_UMBRAL = META_RECALL + MARGEN_RECALL  # 0.90

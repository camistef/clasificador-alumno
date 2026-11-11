# Plan de implementación — Clasificador de alumnos en riesgo

> Basado en [`requirements.md`](requirements.md). Este documento es **solo el plan**: cada fase se implementará después, con tu aprobación.

---

## 1. Resumen de decisiones

| Tema | Decisión |
|---|---|
| Tipo de aprendizaje | **Supervisado**: el modelo aprende de 200 alumnos ya etiquetados (`en_riesgo` = 0/1) |
| Tipo de problema | **Clasificación binaria** |
| Algoritmos a validar | **Regresión logística** y **árbol de decisión** (ambos interpretables); se comparan y se elige uno |
| División de datos | **80 % entrenamiento / 20 % prueba**, estratificada, semilla fija; validación cruzada estratificada de **5 pliegues** dentro del 80 % |
| Meta principal | **Recall (clase en riesgo) ≥ 80 %** |
| Meta secundaria | **ROC-AUC ≥ 0.80** |
| Desempate | Si ambos cumplen las metas, gana el de **mayor precisión** |
| Stack | Python + pandas + scikit-learn + matplotlib + openpyxl |
| App | **Streamlit**; acepta **.xlsx y .csv** |
| Excel de producción | **Sintéticos realistas**, con ID ficticio, **sin** columna `en_riesgo` |
| Pruebas | **pytest** (automatizadas) + **reporte de evaluación** del modelo |
| Documentos | `docs/plan_implementacion.md` (este archivo) y `docs/explicacion_codigo.md` |

---

## 2. Validación del método: por qué es aprendizaje y no fórmulas

**Requisito:** el sistema tiene que **aprender de los datos y predecir**. No puede ser un conjunto de reglas o fórmulas escritas a mano.

| | Método de fórmulas / reglas fijas | Método de aprendizaje (el que usaremos) |
|---|---|---|
| ¿Quién define la lógica? | Una persona escribe, p. ej., `si asistencia < 60 → riesgo` | El algoritmo **descubre** la lógica a partir de los ejemplos etiquetados |
| ¿Usa los datos históricos? | No, o solo como referencia | Sí: el modelo se **entrena** (`fit`) con ellos |
| ¿Cambia si cambian los datos? | No; hay que reescribir el código | Sí; al reentrenar con datos nuevos, el modelo se ajusta solo |
| ¿Puede medirse su error? | No de forma sistemática | Sí: recall, precisión, ROC-AUC sobre datos que nunca vio |

**Cómo aprende cada algoritmo candidato:**

- **Regresión logística.** Durante el entrenamiento **aprende un peso por cada variable** (p. ej. cuánto sube el riesgo por cada punto menos de asistencia), optimizando una función de error. Para predecir, combina esos pesos aprendidos y da una **probabilidad de riesgo** entre 0 y 1. Los pesos no se escriben a mano: los calcula el algoritmo.
- **Árbol de decisión.** Durante el entrenamiento **aprende las reglas**: elige qué variable y qué valor de corte separan mejor a los alumnos en riesgo de los que no lo están (p. ej. "asistencia < 63.5"). Nadie define ni las preguntas ni los cortes.

**Prueba objetiva de que el modelo aprende** (se incluirá como test automatizado en la Fase 6):
1. Entrenar el modelo con las etiquetas reales: debe alcanzar ROC-AUC ≥ 0.80.
2. Entrenar el mismo modelo con las etiquetas **revueltas al azar**: el ROC-AUC debe caer a ≈ 0.50, es decir, al azar.
3. Si el resultado depende de las etiquetas, el modelo **aprendió de los datos**. Una fórmula fija daría el mismo resultado en ambos casos.

---

## 3. Estructura del proyecto propuesta

```
clasificador-alumno/
├── dataset/
│   └── datos.csv                    # datos de entrenamiento (200 alumnos)
├── docs/
│   ├── requirements.md
│   ├── plan_implementacion.md       # este documento
│   └── explicacion_codigo.md        # Entregable 3
├── src/
│   ├── config.py                    # columnas, rangos válidos, semilla, metas
│   ├── datos.py                     # carga (.csv/.xlsx) y validación
│   ├── entrenamiento.py             # entrenamiento, comparación y selección
│   ├── explicacion.py               # motivo principal por alumno
│   ├── prediccion.py                # predicción con el modelo guardado
│   └── generar_excel_prueba.py      # genera los Excel de producción
├── app/
│   └── app.py                       # aplicación Streamlit
├── modelos/
│   ├── modelo.joblib                # modelo entrenado
│   └── metadata.json                # algoritmo, umbral, métricas, fecha
├── reportes/
│   ├── reporte_entrenamiento.md     # Entregable 1 (reporte)
│   └── *.png                        # matrices de confusión, curvas ROC
├── pruebas_produccion/              # Entregable 2
│   └── *.xlsx
├── tests/                           # Entregable 1 (pytest)
└── requirements.txt
```

---

## 4. Fases

Cada fase tiene un **objetivo**, **actividades**, un **criterio de alcance medible** y su **entregable**. Una fase se da por terminada solo cuando cumple su criterio.

### Fase 0 — Preparación del entorno

- **Objetivo:** tener un proyecto reproducible y listo para desarrollar.
- **Actividades:**
  - Crear la estructura de carpetas.
  - Crear `requirements.txt` con versiones fijas (pandas, scikit-learn, matplotlib, openpyxl, streamlit, joblib, pytest).
  - Crear `src/config.py` con las columnas, los rangos válidos de cada variable, la semilla aleatoria y las metas (recall 0.80, ROC-AUC 0.80).
- **Criterio de alcance medible:**
  - `pip install -r requirements.txt` termina sin errores.
  - `pytest` se ejecuta (0 tests fallidos).
- **Entregable:** estructura del proyecto y `requirements.txt`.

### Fase 1 — Análisis exploratorio de datos (EDA)

- **Objetivo:** entender los datos antes de entrenar y confirmar que sirven para aprendizaje supervisado.
- **Actividades:**
  - Verificar tipos de dato, valores faltantes, duplicados y rangos.
  - Ver la distribución de la clase objetivo (62 % / 38 %).
  - Comparar cada variable entre alumnos en riesgo y sin riesgo, y calcular correlaciones.
- **Criterio de alcance medible:**
  - Se documenta el 100 % de las 6 columnas (tipo, rango, faltantes).
  - Hay al menos 1 gráfica por variable que compare las dos clases.
  - Se confirma 0 valores faltantes y se registra el número de duplicados.
- **Entregable:** sección "Análisis de datos" en `reportes/reporte_entrenamiento.md`.

### Fase 2 — Preparación de datos

- **Objetivo:** dividir los datos sin fuga de información, para que la evaluación final sea honesta.
- **Actividades:**
  - Dividir 80/20 de forma estratificada con semilla fija: 160 alumnos para entrenamiento y 40 para prueba.
  - Crear un `Pipeline` de scikit-learn:
    - **Regresión logística:** `StandardScaler` + modelo.
    - **Árbol de decisión:** sin escalado, porque no lo necesita.
  - El escalado se ajusta **solo** con los datos de entrenamiento (dentro del pipeline).
- **Criterio de alcance medible:**
  - Tamaños exactos: 160 / 40.
  - La proporción de la clase en riesgo en ambos conjuntos es 38 % ± 2 puntos.
  - El conjunto de prueba no se usa en ninguna decisión hasta la Fase 4.
- **Entregable:** funciones de carga y división en `src/datos.py` y `src/entrenamiento.py`.

### Fase 3 — Entrenamiento y validación de los modelos

- **Objetivo:** entrenar los dos algoritmos candidatos y medir su desempeño con validación cruzada.
- **Actividades:**
  - **Regresión logística:** `class_weight="balanced"`. Búsqueda del parámetro de regularización `C` con `GridSearchCV`.
  - **Árbol de decisión:** `class_weight="balanced"`. Búsqueda de `max_depth` y `min_samples_leaf` con `GridSearchCV`; la profundidad se limita para que el árbol siga siendo explicable.
  - Validación cruzada **estratificada de 5 pliegues** sobre el 80 % de entrenamiento.
  - **Ajuste del umbral de decisión:** con las probabilidades de la validación cruzada, buscar el umbral que logre recall ≥ 0.80 con la mayor precisión posible.
- **Criterio de alcance medible:**
  - Tabla comparativa de **ambos** modelos con media y desviación estándar en validación cruzada de: recall, precisión, F1 y ROC-AUC.
  - Umbral elegido y justificado para cada modelo.
- **Entregable:** `src/entrenamiento.py` y la sección "Comparación de modelos" del reporte.

### Fase 4 — Selección y evaluación final

- **Objetivo:** elegir el modelo definitivo y confirmar en datos nunca vistos que cumple las metas.
- **Actividades:**
  - Aplicar la regla de selección:
    1. Descartar los modelos que no cumplan **recall ≥ 0.80** y **ROC-AUC ≥ 0.80** en validación cruzada.
    2. Si los dos cumplen, gana el de **mayor precisión**.
    3. **Si ninguno cumple:** se analizan los resultados según el contexto (qué métrica falla, por cuánto y en qué tipo de alumnos) y se proponen **otros métodos de aprendizaje** (p. ej. Random Forest, Gradient Boosting, SVM o KNN), con la justificación de cada uno. **El método se elige contigo antes de implementarlo** (RM-07).
  - Reentrenar el modelo elegido con el 80 % completo y evaluarlo **una sola vez** en el 20 % de prueba.
  - Guardar `modelos/modelo.joblib` y `modelos/metadata.json` (algoritmo, hiperparámetros, umbral, métricas, fecha y versiones).
- **Criterio de alcance medible:**
  - En el conjunto de prueba: **recall ≥ 0.80** y **ROC-AUC ≥ 0.80**.
  - Se reportan la matriz de confusión, la curva ROC y la precisión.
  - El modelo se puede volver a cargar desde disco y produce predicciones idénticas.
- **Entregable:** modelo guardado y sección "Evaluación final" del reporte.

### Fase 5 — Explicabilidad (motivo principal)

- **Objetivo:** que cada alumno marcado tenga un motivo claro y sin tecnicismos (RF-05, RNF-03).
- **Actividades:**
  - **Si gana la regresión logística:** el motivo es la variable con mayor contribución al riesgo (peso aprendido × valor estandarizado del alumno).
  - **Si gana el árbol:** el motivo es la condición más determinante en el camino de decisión del alumno.
  - Traducir cada variable a lenguaje claro, p. ej. `asistencia` → "Baja asistencia" y `tareas_entregadas` → "Pocas tareas entregadas".
- **Criterio de alcance medible:**
  - El 100 % de los alumnos marcados "citar" tiene un motivo no vacío.
  - En 5 casos revisados a mano, el motivo coincide con el dato del alumno (p. ej. "Baja asistencia" solo si su asistencia está por debajo de la media).
- **Entregable:** `src/explicacion.py`.

### Fase 6 — Pruebas de entrenamiento · **Entregable 1**

- **Objetivo:** verificar automáticamente que el entrenamiento, el modelo y las validaciones funcionan, y documentar los resultados.
- **Tests automatizados (pytest):**

| Test | Qué verifica |
|---|---|
| `test_division_estratificada` | 160/40 y proporción de clases 38 % ± 2 |
| `test_sin_fuga_datos` | El escalador solo se ajusta con datos de entrenamiento |
| `test_modelo_cumple_recall` | Recall ≥ 0.80 en prueba |
| `test_modelo_cumple_roc_auc` | ROC-AUC ≥ 0.80 en prueba |
| `test_modelo_aprende` | Con etiquetas revueltas, ROC-AUC ≈ 0.50 (prueba de aprendizaje, ver §2) |
| `test_reproducibilidad` | Misma semilla → mismo modelo y mismas métricas |
| `test_guardar_cargar_modelo` | El modelo cargado de disco predice igual |
| `test_validacion_columnas` | Rechaza archivos sin columnas requeridas |
| `test_validacion_rangos` | Rechaza valores fuera de rango o vacíos |
| `test_lectura_xlsx_y_csv` | Lee ambos formatos con el mismo resultado |
| `test_motivo_presente` | Todo alumno citado tiene motivo |

- **Reporte de evaluación:** `reportes/reporte_entrenamiento.md`, con EDA, comparación de modelos, umbral, métricas finales, matriz de confusión y curva ROC.
- **Criterio de alcance medible:**
  - **100 % de los tests pasan.**
  - El reporte contiene las 4 secciones: análisis de datos, comparación, evaluación final y conclusión.
- **Entregable:** carpeta `tests/` y `reportes/reporte_entrenamiento.md`.

### Fase 7 — Excel para pruebas de producción · **Entregable 2**

- **Objetivo:** contar con archivos realistas para probar la app como si fuera una semana real.
- **Actividades:**
  - Crear `src/generar_excel_prueba.py`, que genera alumnos **sintéticos** con distribuciones similares al dataset original, con semilla fija e ID ficticio (`A-0001`, …).
  - Como no hay un mínimo ni un máximo de alumnos por archivo, la **cantidad es un parámetro** del script; la app debe aceptar cualquier cantidad.
  - Los archivos **no** incluyen la columna `en_riesgo`.

| Archivo | Escenario | Contenido |
|---|---|---|
| `semana_normal.xlsx` | Semana normal | Archivo válido con mezcla realista de alumnos |
| `casos_extremos.xlsx` | Casos extremos | Valores en los límites: 0 tareas, 100 % de asistencia, promedio 10, 0 horas, etc. |
| `errores_columnas.xlsx` | Errores | Falta una columna requerida |
| `errores_valores.xlsx` | Errores | Valores fuera de rango, vacíos y texto en columnas numéricas |
| `archivo_grande.xlsx` | Volumen | 1,000+ alumnos válidos |

- **Criterio de alcance medible:**
  - Se generan los 5 archivos y se pueden abrir en Excel.
  - Los archivos válidos pasan la validación al 100 %.
  - Los archivos con errores son rechazados con un mensaje específico.
  - Las variables numéricas de los archivos sintéticos quedan dentro de los rangos del dataset original.
- **Entregable:** carpeta `pruebas_produccion/` y el script generador.

### Fase 8 — Aplicación web (Streamlit)

- **Objetivo:** que la coordinación responda *"¿a quién llamo a tutoría esta semana?"* sin conocimientos técnicos.
- **Actividades:**
  - Carga de archivo `.xlsx` o `.csv`, sin límite de filas.
  - **Estado visible en todo momento** (cargando → validando → prediciendo → listo / error) y **barra de avance** durante el procesamiento (RF-08). No hay tiempo máximo.
  - Validación con mensajes claros (RF-02).
  - Tabla ordenada con `id_alumno`, probabilidad, "Citar: Sí/No" y motivo (RF-03 a RF-06).
  - Botón para descargar la lista de alumnos a citar (RF-07).
  - La app usa el modelo guardado; **no reentrena** (RNF-04).
- **Criterio de alcance medible:**
  - Los 5 Excel de la Fase 7 producen el resultado esperado: los válidos muestran el ranking y los inválidos muestran un error claro.
  - El archivo descargado contiene exactamente los alumnos marcados "Sí".
  - En los 5 escenarios, el estado y la barra de avance se muestran desde la carga hasta el resultado final.
- **Entregable:** `app/app.py`.

### Fase 9 — Documentación del código · **Entregable 3**

- **Objetivo:** explicar todo el código y justificar las decisiones de método.
- **Contenido de `docs/explicacion_codigo.md`:**
  1. Visión general y flujo: datos → entrenamiento → modelo → predicción → app.
  2. Por qué aprendizaje supervisado y no reglas o fórmulas (ver §2).
  3. Por qué regresión logística y árbol de decisión; por qué se eligió el ganador, con sus métricas.
  4. Por qué recall como métrica principal, por qué el ajuste de umbral y por qué `class_weight="balanced"`.
  5. Por qué la división 80/20 con validación cruzada y cómo se evita la fuga de datos.
  6. Explicación archivo por archivo y función por función.
  7. Cómo se calcula el motivo principal.
  8. Cómo ejecutar: instalar, entrenar, correr tests, generar los Excel y abrir la app.
  9. Limitaciones: dataset pequeño (200), datos sintéticos de prueba, y cuándo conviene reentrenar.
- **Criterio de alcance medible:**
  - El 100 % de los archivos de `src/` y `app/` están explicados.
  - Cada decisión de la tabla §1 tiene su justificación.
  - Una persona puede ejecutar el proyecto completo siguiendo solo este documento.
- **Entregable:** `docs/explicacion_codigo.md`.

---

## 5. Relación entregables ↔ fases

| # | Entregable solicitado | Fase | Ubicación |
|---|---|---|---|
| 1 | Test de entrenamiento | 6 (y apoyado en 1–4) | `tests/`, `reportes/reporte_entrenamiento.md` |
| 2 | Excel para pruebas de producción | 7 | `pruebas_produccion/*.xlsx` |
| 3 | MD con la explicación del código y del método | 9 | `docs/explicacion_codigo.md` |
| 4 | Método de aprendizaje, no de fórmulas | 2–4 (validado en §2 y `test_modelo_aprende`) | `src/entrenamiento.py`, `modelos/` |

---

## 6. Riesgos

| Riesgo | Mitigación |
|---|---|
| Dataset pequeño (200 alumnos): métricas inestables | Validación cruzada estratificada; reportar media ± desviación |
| Ningún modelo alcanza las metas | Analizar según el contexto y proponer otro método de aprendizaje; se elige contigo (RM-07) |
| Datos de producción distintos a los de entrenamiento | Validación estricta de columnas y rangos en la app |
| Árbol demasiado profundo (sobreajuste, difícil de explicar) | Limitar `max_depth` en la búsqueda de hiperparámetros |

---

## 7. Puntos confirmados

| Punto | Decisión |
|---|---|
| Actualizar `requirements.md` | Hecho: .xlsx/.csv, ROC-AUC ≥ 0.80, 80/20 + validación cruzada, RF-08 y RM-07 |
| Alumnos por archivo | Sin mínimo ni máximo; la cantidad es un parámetro del generador |
| Tiempo de respuesta | Sin límite, siempre mostrando el estado y el avance |
| Si ningún modelo cumple | Evaluar según el contexto y proponer otro método de aprendizaje, a elegir contigo |

# Requerimientos — Clasificador de alumnos en riesgo

> Plan de desarrollo: [`plan_implementacion.md`](plan_implementacion.md)

## 1. Contexto del caso

**Caso A — Coordinación académica: "Detectar alumnos en riesgo"**

| Campo | Descripción |
|---|---|
| **Cliente** | Coordinación académica de una universidad |
| **Problema** | Se pierden alumnos por reprobación y la coordinación se entera demasiado tarde |
| **Objetivo** | Anticipar, con los datos del semestre, qué alumnos están en riesgo para enviarlos a tutoría a tiempo |
| **Decisión a tomar** | *"¿A quién llamo a tutoría esta semana?"* |

## 2. Datos

**Archivo de entrenamiento:** `dataset/datos.csv` (`caso1_riesgo_alumnos.csv`) — 200 registros, sin valores faltantes.

| Variable | Tipo | Rango | Descripción |
|---|---|---|---|
| `horas_estudio` | Numérica | 0 – 11.9 | Horas de estudio semanales |
| `asistencia` | Numérica (%) | 42 – 100 | Porcentaje de asistencia |
| `promedio_anterior` | Numérica | 5.1 – 10 | Promedio del periodo anterior |
| `tareas_entregadas` | Entera | 0 – 10 | Número de tareas entregadas |
| `participacion_clase` | Entera (escala) | 0 – 5 | Nivel de participación en clase |
| `en_riesgo` | Binaria (**objetivo**) | 0 / 1 | 1 = el alumno terminó en riesgo |

**Distribución de la clase objetivo:** 124 sin riesgo (62 %) · 76 en riesgo (38 %). Desbalance moderado.

**Archivos semanales (producción):**
- Formato **.xlsx o .csv**.
- Deben incluir una columna de identificación del alumno (`id_alumno` / matrícula). El modelo **no** la usa para predecir, pero se conserva en la salida para saber a quién llamar.
- **No** incluyen la columna `en_riesgo`, porque es lo que el modelo predice.
- **No hay un número mínimo ni máximo de alumnos por archivo**; la cantidad varía cada semana.

## 3. Requerimientos funcionales

| ID | Requerimiento |
|---|---|
| RF-01 | La aplicación permite **cargar un archivo .xlsx o .csv** con los datos de los alumnos de la semana, sin límite de filas. |
| RF-02 | Valida que el archivo tenga las columnas requeridas (`id_alumno` + las 5 variables) y avisa si falta alguna o hay valores inválidos (vacíos, fuera de rango, texto en columnas numéricas). |
| RF-03 | Calcula para cada alumno la **probabilidad de estar en riesgo**. |
| RF-04 | Muestra una **lista ordenada** de mayor a menor probabilidad de riesgo. |
| RF-05 | Para cada alumno en riesgo, indica el **motivo principal** (el factor que más contribuye, p. ej. "baja asistencia" o "pocas tareas entregadas"). |
| RF-06 | Marca como "citar a tutoría" a **todos** los alumnos que superen el umbral de riesgo (no hay cupo máximo). |
| RF-07 | Permite **descargar** la lista de alumnos a citar. |
| RF-08 | Muestra en todo momento el **estado del proceso** (cargando, validando, prediciendo, listo o error) y una **barra de avance** mientras procesa. No hay tiempo máximo de respuesta. |

## 4. Requerimientos del modelo

| ID | Requerimiento |
|---|---|
| RM-01 | Tipo de problema: **clasificación binaria con aprendizaje supervisado** (`en_riesgo`). |
| RM-02 | Debe ser un **método de aprendizaje**: el modelo aprende sus parámetros a partir de los datos (`fit`) y luego predice. **No** se permiten reglas ni fórmulas definidas a mano. |
| RM-03 | **Prioridad: minimizar falsos negativos.** Es preferible citar de más a un alumno que no detectar a uno en riesgo. |
| RM-04 | **Metas:** recall de la clase "en riesgo" **≥ 80 %** y **ROC-AUC ≥ 0.80**. |
| RM-05 | Métricas adicionales a reportar: precisión, F1-score y matriz de confusión. |
| RM-06 | Algoritmos candidatos (interpretables): **regresión logística** y **árbol de decisión**. Se comparan y gana el que cumpla las metas; si ambos las cumplen, gana el de **mayor precisión**. |
| RM-07 | Si **ningún** modelo cumple las metas, se evalúan los resultados según el contexto y se propone **otro método de aprendizaje** (p. ej. Random Forest, Gradient Boosting, SVM o KNN). La decisión se consulta antes de implementarla. |
| RM-08 | El umbral de decisión se ajusta (no necesariamente 0.5) para cumplir la meta de recall. |
| RM-09 | Se considera el desbalance de clases (`class_weight="balanced"` y validación estratificada). |
| RM-10 | Evaluación: división **80 % entrenamiento / 20 % prueba** estratificada con semilla fija, más **validación cruzada estratificada de 5 pliegues** dentro del 80 %. El 20 % de prueba se usa una sola vez, al final. |

## 5. Requerimientos no funcionales

| ID | Requerimiento |
|---|---|
| RNF-01 | **Stack:** Python + pandas + scikit-learn + matplotlib + openpyxl. |
| RNF-02 | **Interfaz:** aplicación web en **Streamlit**, usable por personal no técnico. |
| RNF-03 | Las explicaciones se muestran en lenguaje claro, sin términos técnicos. |
| RNF-04 | El modelo entrenado se guarda en archivo (`joblib`) para no reentrenar en cada uso. |
| RNF-05 | Los datos de los alumnos son confidenciales; la app se usa de forma local o interna. |
| RNF-06 | El tiempo de procesamiento no tiene límite, pero el usuario siempre debe ver el estado y el avance (ver RF-08). |

## 6. Entregable semanal (salida)

Tabla ordenada por probabilidad de riesgo:

| id_alumno | probabilidad_riesgo | citar_tutoria | motivo_principal |
|---|---|---|---|
| A-0123 | 0.91 | Sí | Baja asistencia |
| A-0456 | 0.84 | Sí | Pocas tareas entregadas |
| A-0789 | 0.22 | No | — |

## 7. Entregables del proyecto

| # | Entregable | Ubicación |
|---|---|---|
| 1 | Pruebas de entrenamiento (pytest + reporte de evaluación) | `tests/`, `reportes/reporte_entrenamiento.md` |
| 2 | Excel para pruebas de producción (alumnos sintéticos, sin `en_riesgo`) | `pruebas_produccion/*.xlsx` |
| 3 | Explicación de todo el código y del porqué del método | `docs/explicacion_codigo.md` |
| 4 | Método de aprendizaje (no fórmulas), validado con prueba de etiquetas revueltas | `src/entrenamiento.py`, `modelos/` |

## 8. Criterios de aceptación

- [ ] El modelo alcanza **recall ≥ 80 %** y **ROC-AUC ≥ 0.80** en el conjunto de prueba.
- [ ] El modelo demuestra que aprende: entrenado con etiquetas revueltas, su ROC-AUC cae a ≈ 0.50.
- [ ] La app carga archivos .xlsx y .csv válidos y muestra la lista ordenada con probabilidad y motivo.
- [ ] La app rechaza con un mensaje claro los archivos con columnas faltantes o valores inválidos.
- [ ] La app muestra el estado y el avance durante todo el procesamiento.
- [ ] Se puede descargar la lista de alumnos a citar.
- [ ] El motivo mostrado es coherente con los pesos o reglas aprendidos por el modelo.
- [ ] El 100 % de los tests automatizados pasan.

## 9. Supuestos y fuera de alcance

**Supuestos**
- Los datos semanales tienen las mismas columnas y escalas que el dataset de entrenamiento.
- La columna `en_riesgo` histórica es una etiqueta confiable.

**Fuera de alcance**
- Integración con sistemas escolares, envío automático de citas o notificaciones.
- Gestión de usuarios / autenticación.
- Modelos de "caja negra" (Random Forest, XGBoost, redes neuronales), **salvo** que los modelos interpretables no cumplan las metas y se apruebe su uso (RM-07).

## 10. Decisiones tomadas (resumen de las entrevistas)

| Pregunta | Decisión |
|---|---|
| ¿Qué error cuesta más? | Falsos negativos → priorizar **recall** |
| Capacidad de tutoría | **Sin límite fijo**: se cita a todos los que superen el umbral |
| Salida esperada | **Lista ordenada + motivo** |
| Explicabilidad | **Muy importante** → modelos interpretables |
| Metas | **Recall ≥ 80 %** y **ROC-AUC ≥ 0.80** |
| Algoritmos | **Comparar** regresión logística y árbol de decisión |
| Desempate | **Mayor precisión** |
| Si ningún modelo cumple | Evaluar según el contexto y **proponer otro método** (con consulta) |
| División de datos | **80/20** + validación cruzada de 5 pliegues |
| Forma de uso | **App web en Streamlit** |
| Formato de archivos | **.xlsx y .csv** |
| Alumnos por archivo | **Sin mínimo ni máximo** |
| Tiempo de respuesta | **Sin límite**, mostrando estado y avance |
| Stack | **Python + scikit-learn** |
| Identificación | Columna **`id_alumno`** en el archivo semanal |
| Pruebas | **pytest + reporte** de evaluación |
| Excel de prueba | **Sintéticos**, sin `en_riesgo`: semana normal, casos extremos, errores, archivo grande |

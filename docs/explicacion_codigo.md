# Explicación del código y del método — Clasificador de alumnos en riesgo

> Relacionado: [`requirements.md`](requirements.md) · [`plan_implementacion.md`](plan_implementacion.md) · [`../reportes/reporte_entrenamiento.md`](../reportes/reporte_entrenamiento.md)

---

## 1. Visión general y flujo

```
dataset/datos.csv ──► src/entrenamiento.py ──► modelos/modelo.joblib + metadata.json
   (200 alumnos          (división 80/20, CV,         (modelo + umbral + métricas)
    etiquetados)          selección, evaluación)                │
                                │                               ▼
                                ▼               archivo semanal (.xlsx/.csv)
              reportes/reporte_entrenamiento.md         │
                                                        ▼
                                           app/app.py (Streamlit)
                                    validar ► predecir ► motivo ► descargar
```

1. **Entrenamiento** (una vez, o cuando haya datos nuevos): `python -m src.entrenamiento`.
2. **Uso semanal**: la coordinación sube el archivo a la app; la app **no reentrena**, solo carga el modelo guardado (RNF-04).

---

## 2. Por qué aprendizaje supervisado y no reglas o fórmulas

Contamos con 200 alumnos **ya etiquetados** (`en_riesgo` = 0/1). Eso permite aprendizaje **supervisado**: el algoritmo recibe ejemplos (variables → resultado conocido) y **ajusta sus parámetros** (`fit`) para reproducir ese resultado en alumnos nuevos.

- Nadie escribe en el código reglas como `si asistencia < 60 → riesgo`. Los **pesos** de la regresión logística y los **cortes** del árbol los calcula el algoritmo.
- Si mañana llegan datos nuevos, basta con reentrenar: el modelo cambia solo.
- **Prueba objetiva** (`tests/test_entrenamiento.py::test_modelo_aprende`): se entrena el mismo modelo con las etiquetas **revueltas al azar** 10 veces. El ROC-AUC cae a ≈ 0.50 (azar), mientras que con las etiquetas reales es ≈ 0.95. Como el resultado depende de las etiquetas, el modelo **aprendió de los datos**; una fórmula fija daría lo mismo en ambos casos.

---

## 3. Algoritmos candidatos y elección

| | Regresión logística | Árbol de decisión |
|---|---|---|
| Qué aprende | Un peso por variable | Preguntas y cortes (p. ej. "horas ≤ 4.2") |
| Por qué es candidato | Interpretable: el peso dice cuánto sube o baja el riesgo | Interpretable: se puede seguir el camino de decisión |
| Hiperparámetros buscados | `C` (regularización) | `max_depth` ≤ 5, `min_samples_leaf` |

Resultado (validación cruzada repetida 10 × 5 sobre el 80 %, umbral con margen; ver reporte):

| Modelo | Umbral | Recall | Precisión | ROC-AUC | ¿Cumple? |
|---|---|---|---|---|---|
| Regresión logística (C = 100) | 0.429 | 0.908 ± 0.095 | **0.769** ± 0.076 | 0.945 ± 0.031 | Sí |
| Árbol (profundidad 4, hoja ≥ 15) | 0.187 | 0.905 ± 0.089 | 0.533 ± 0.069 | 0.814 ± 0.077 | Sí |

Ambos cumplen; según la regla de desempate gana **regresión logística**, por mayor precisión (cita menos alumnos de más para el mismo recall). Además tiene mucho mayor ROC-AUC.

**Evaluación final en el 20 % de prueba:** recall **0.800** (12 de 15 detectados), ROC-AUC **0.923**, precisión 0.667, F1 0.727. Cumple ambas metas.

> **Historia del umbral (transparencia).** En una primera versión el umbral se eligió con una sola validación cruzada de 5 pliegues y **sin margen** (recall ≥ 0.80 exacto, umbral 0.690). En la prueba eso dio recall 0.733. Como el umbral quedaba justo en el límite, se cambió el procedimiento (§4) y se evaluó por segunda vez. El umbral **no** se ajustó con los datos de prueba, pero como la prueba ya se había visto, el resultado debe leerse con cautela (§10).

---

## 4. Decisiones de métricas

- **Recall como métrica principal (RM-03).** El error más caro es no detectar a un alumno en riesgo (falso negativo). Recall = proporción de alumnos en riesgo que el modelo detecta.
- **ROC-AUC como meta secundaria.** Mide si el modelo ordena bien a los alumnos (los de riesgo arriba) sin depender del umbral; asegura que el ranking de la app tiene sentido.
- **Ajuste del umbral (RM-08).** El modelo da una probabilidad; el umbral decide a partir de cuánto se cita. 0.5 no es especial: se busca, con las probabilidades **fuera de pliegue** de las 10 repeticiones de la validación cruzada, el umbral con **mayor precisión entre los que mantienen recall ≥ 0.90** (`elegir_umbral`). Así el umbral se decide sin tocar el conjunto de prueba.
- **Por qué 0.90 y no 0.80.** Si el umbral se elige para cumplir 0.80 exacto, en datos nuevos el recall cae por debajo de la meta aproximadamente la mitad de las veces. El recall varía ≈ 0.10 (una desviación estándar) entre pliegues, así que la meta interna es 0.80 + 0.10 (`MARGEN_RECALL` en `config.py`). El costo es menor precisión: se cita a más alumnos que no lo necesitaban, lo cual es aceptable según RM-03.
- **Por qué validación cruzada repetida.** Con 160 alumnos, el umbral elegido con una sola partición en 5 pliegues depende mucho de cómo cayeron los alumnos en cada pliegue. Repetirla 10 veces con particiones distintas (50 pliegues) da un umbral y unas métricas más estables.
- **`class_weight="balanced"` (RM-09).** Hay 62 % / 38 %. Sin compensación, el modelo tiende a favorecer la clase mayoritaria ("sin riesgo"). `balanced` da más peso a cada error en la clase minoritaria.

---

## 5. División 80/20 + validación cruzada y cómo se evita la fuga de datos

- **80/20 estratificado, semilla 42** → 160 / 40 alumnos, ambos con ≈ 38 % en riesgo (`dividir_datos`).
- El **20 % de prueba** no participa en ninguna decisión (hiperparámetros, umbral ni elección de modelo). Se usa **una sola vez**, al final.
- **Validación cruzada estratificada de 5 pliegues, repetida 10 veces**, dentro del 80 %: con solo 160 alumnos, una sola partición sería muy inestable; los 50 pliegues dan media ± desviación.
- **Sin fuga:** el `StandardScaler` está **dentro del `Pipeline`**, por lo que en cada `fit` (cada pliegue y el entrenamiento final) se ajusta solo con los datos de entrenamiento de ese momento. `test_sin_fuga_datos` verifica que la media del escalador es la del 80 % y no la del dataset completo.

---

## 6. Archivo por archivo

### `src/config.py`
Constantes centrales: rutas, nombres de columnas (`COLUMNA_ID`, `VARIABLES`, `COLUMNA_OBJETIVO`), **rangos válidos** para la validación (`RANGOS_VALIDOS`: mínimo, máximo y si debe ser entero), textos en lenguaje claro (`NOMBRES_CLAROS`, `MOTIVOS_BAJO`, `MOTIVOS_ALTO`), semilla, tamaño de prueba, número de pliegues y metas. Cambiar una meta o un rango se hace solo aquí.

Los rangos válidos son los **límites lógicos** de cada variable (asistencia 0–100, promedio 0–10, tareas 0–10, participación 0–5, horas 0–80), no los observados en el dataset: un alumno con 30 % de asistencia es posible, mientras que 130 % es un error de captura.

### `src/datos.py`
| Función | Qué hace |
|---|---|
| `ErrorValidacion` | Excepción que guarda una **lista** de mensajes, para mostrar todos los problemas a la vez. |
| `leer_archivo(origen, nombre)` | Lee `.csv` o `.xlsx` desde una ruta o un archivo abierto (el de Streamlit). Rechaza otros formatos. |
| `cargar_dataset()` | Lee `dataset/datos.csv` y comprueba que estén las 6 columnas. |
| `validar_datos(df)` | Revisa columnas requeridas, archivo vacío, `id_alumno` vacío o repetido y, por variable: celdas vacías, texto, fuera de rango y no enteros. Indica las **filas de Excel** afectadas (encabezado = fila 1). Devuelve los datos con tipos numéricos. |
| `dividir_datos(df)` | División 80/20 estratificada con semilla fija. |

### `src/entrenamiento.py`
| Función | Qué hace |
|---|---|
| `crear_candidatos()` | Define los dos pipelines (logística con escalador; árbol sin escalador) y sus rejillas de hiperparámetros. |
| `elegir_umbral(y, prob)` | Umbral de mayor precisión con recall ≥ meta interna (0.90 por defecto). |
| `metricas(y, prob, umbral)` | Recall, precisión, F1 y ROC-AUC. |
| `evaluar_candidato()` | `GridSearchCV` (ROC-AUC, CV repetida 10 × 5) → probabilidades fuera de pliegue en cada repetición con los mejores hiperparámetros → umbral sobre todas las repeticiones → métricas en los 50 pliegues (media y desviación). |
| `seleccionar()` | Regla del plan: descartar los que no cumplen; desempate por precisión; `None` si ninguno cumple (RM-07). |
| `entrenar_final()` | Reentrena el elegido con todo el 80 %. |
| `guardar_modelo()` | Guarda `modelo.joblib` y `metadata.json` (algoritmo, hiperparámetros, umbral, métricas CV y de prueba, matriz de confusión, fecha y versiones). |
| `ejecutar()` | Orquesta las fases 2–4. Con `python -m src.entrenamiento` además imprime el resumen y genera el reporte. |

### `src/reporte.py`
Genera `reportes/reporte_entrenamiento.md` con 4 secciones (análisis de datos, comparación, evaluación final, conclusión) y las gráficas: diagrama de caja por variable y clase, mapa de correlaciones, curvas ROC (validación cruzada y prueba) y matriz de confusión.

### `src/explicacion.py`
Calcula el **motivo principal** (ver §7). `motivo_principal(modelo, X, algoritmo)` devuelve un texto por alumno.

### `src/prediccion.py`
| Función | Qué hace |
|---|---|
| `cargar_modelo()` | Carga el modelo y su metadata; si no existen, pide entrenar primero. |
| `predecir(df, modelo, metadata)` | Valida, calcula probabilidad, marca "Sí/No" según el umbral guardado y pone el motivo a los citados (— a los demás). Devuelve la tabla ordenada de mayor a menor riesgo. |
| `alumnos_a_citar(resultado)` | Filtra los marcados "Sí" (lo que se descarga). |

### `src/generar_excel_prueba.py`
Genera los 5 Excel de `pruebas_produccion/`. Los alumnos sintéticos se crean **remuestreando filas reales** y agregando un ruido pequeño (15 % de la desviación estándar), recortado a los rangos observados; así se conservan las distribuciones **y** las relaciones entre variables. IDs ficticios `A-0001…`; nunca incluye `en_riesgo`. Las cantidades son parámetros (`--normal`, `--grande`).

| Archivo | Contenido | Resultado esperado en la app |
|---|---|---|
| `semana_normal.xlsx` | 60 alumnos válidos | Ranking |
| `casos_extremos.xlsx` | 12 alumnos en los límites (0 horas, 0 tareas, 100 % asistencia, promedio 10, todo mínimo, todo máximo…) | Ranking |
| `errores_columnas.xlsx` | Falta `asistencia` | Error: columna faltante |
| `errores_valores.xlsx` | Fuera de rango, vacío, texto, no entero, ID repetido | Error con cada problema y su fila |
| `archivo_grande.xlsx` | 1,200 alumnos válidos | Ranking |

### `app/app.py`
Aplicación Streamlit:
1. Carga el modelo una sola vez (`st.cache_resource`).
2. Recibe un `.xlsx` o `.csv` sin límite de filas.
3. Muestra el **estado** (`st.status`: cargando → validando → calculando → listo / error) y una **barra de avance**; la predicción se hace en lotes de 200 filas para que el avance se vea en archivos grandes.
4. Si hay errores de validación, los lista en lenguaje claro y se detiene.
5. Muestra métricas resumen, la tabla ordenada (con opción de ver solo los citados) y botones para descargar la lista de citados en Excel o CSV.

### `tests/`
| Archivo | Cubre |
|---|---|
| `conftest.py` | Ejecuta el entrenamiento una vez por sesión (sin sobrescribir `modelos/`). |
| `test_entrenamiento.py` | División 160/40 estratificada, sin fuga, ambos candidatos, umbral, metas de recall y ROC-AUC en prueba, prueba de etiquetas revueltas, reproducibilidad, guardar/cargar. |
| `test_datos.py` | Columnas faltantes, rangos, vacíos, texto, enteros, ID repetido, archivo vacío, formato no soportado, misma lectura de `.xlsx` y `.csv`. |
| `test_prediccion.py` | Orden y columnas de salida, citados = los que superan el umbral, motivo presente y coherente, explicación del árbol, cualquier cantidad de alumnos, Excel de producción. |

---

## 7. Cómo se calcula el motivo principal

**Regresión logística (modelo actual).** El modelo calcula el riesgo sumando `peso × valor estandarizado` de cada variable. Esa suma por variable es su **contribución** para ese alumno. El motivo es la variable con **mayor contribución positiva** (la que más sube su riesgo).

Pesos aprendidos (negativo = a menor valor, mayor riesgo): horas de estudio −3.25 · promedio anterior −2.74 · tareas −1.90 · asistencia −1.69 · participación +0.31.

Ejemplo: un alumno con 0.4 horas de estudio (media 6) y promedio 5.1 (media 7.6) tiene contribuciones grandes en ambas; gana la mayor y el motivo es "Promedio anterior bajo" o "Pocas horas de estudio". Como el peso es negativo, una contribución positiva implica que el valor está **por debajo de la media**, así que el texto "Baja…/Pocas…" siempre es coherente con el dato (`test_motivo_coherente_con_dato`).

**Árbol de decisión (si se eligiera).** Se recorre el camino del alumno desde la raíz hasta su hoja; en cada paso se mide cuánto **aumenta** la proporción de alumnos en riesgo. La condición con el mayor aumento es el motivo ("horas ≤ corte" → "Pocas horas de estudio").

Solo los alumnos marcados "Citar: Sí" muestran motivo; el resto muestra "—".

---

## 8. Cómo ejecutar

Desde la carpeta del proyecto (`clasificador-alumno/`), con Python 3.11 o superior:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar (genera modelos/ y reportes/)
python -m src.entrenamiento

# 3. Correr las pruebas
python -m pytest

# 4. Generar los Excel de prueba de producción
python -m src.generar_excel_prueba            # opcional: --normal 80 --grande 2000

# 5. Abrir la app (se abre en el navegador, http://localhost:8501)
streamlit run app/app.py
```

En Windows, si la consola muestra caracteres raros, ejecute antes `set PYTHONIOENCODING=utf-8` (cmd) o `$env:PYTHONIOENCODING="utf-8"` (PowerShell).

---

## 9. Justificación de cada decisión del plan (§1)

| Decisión | Justificación |
|---|---|
| Aprendizaje supervisado | Hay etiquetas históricas; RM-02 exige aprender de los datos (§2). |
| Clasificación binaria | La pregunta es "¿citar o no?"; la probabilidad sirve para ordenar. |
| Logística y árbol | Interpretables: permiten dar el motivo a la coordinación (§3). |
| 80/20 + CV de 5 pliegues | Evaluación honesta con pocos datos (§5). |
| Recall ≥ 0.80 | Falsos negativos son el error más caro (§4). |
| ROC-AUC ≥ 0.80 | Garantiza un buen ordenamiento, independiente del umbral (§4). |
| Desempate por precisión | Con igual recall, citar a menos alumnos de más ahorra tutorías. |
| Python + scikit-learn + Streamlit | Stack estándar, librerías probadas, app web sin desarrollo frontend. |
| `.xlsx` y `.csv` | Formatos que la coordinación ya usa. |
| Excel sintéticos sin `en_riesgo` | Simulan una semana real sin exponer datos de alumnos (RNF-05). |
| pytest + reporte | Verificación automática y repetible + evidencia legible. |

---

## 10. Limitaciones

- **Dataset pequeño (200 alumnos).** La prueba tiene 15 alumnos en riesgo: cada uno vale 6.7 puntos de recall. Las métricas tienen mucha variación (recall en CV: 0.80 ± 0.10).
- **Recall en prueba justo en la meta (0.800).** Un alumno en riesgo menos detectado lo dejaría en 0.733. Además, esta es la segunda evaluación sobre la prueba (ver §3), por lo que no es una evaluación completamente "a ciegas". Conviene confirmar el desempeño con los alumnos reales del siguiente periodo.
- **Más citas de más.** El margen del umbral baja la precisión (0.667 en prueba): de cada 3 alumnos citados, ~1 no estaba en riesgo. En la semana de ejemplo (60 alumnos) se citan 26.
- **Datos de prueba sintéticos.** Los Excel de producción imitan la distribución original pero no son alumnos reales.
- **Probabilidades muy extremas.** Con `C = 100` (poca regularización) muchos alumnos aparecen con probabilidad 0.99–1.00; el ranking sigue siendo útil, pero la probabilidad no debe leerse como certeza.
- **Cuándo reentrenar.** Cada semestre, o cuando cambie la forma de medir alguna variable, o si los alumnos citados que realmente estaban en riesgo bajan notablemente. Para reentrenar basta reemplazar `dataset/datos.csv` y ejecutar `python -m src.entrenamiento`.

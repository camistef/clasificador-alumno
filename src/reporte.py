"""Genera reportes/reporte_entrenamiento.md y sus gráficas (EDA, comparación, evaluación)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src import config

COLOR_SIN = "#4C78A8"
COLOR_RIESGO = "#E45756"


def _guardar(fig, nombre: str) -> str:
    config.DIR_REPORTES.mkdir(exist_ok=True)
    fig.savefig(config.DIR_REPORTES / nombre, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return nombre


# ---------------------------------------------------------------------------
# Fase 1 — EDA
# ---------------------------------------------------------------------------
def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in config.VARIABLES + [config.COLUMNA_OBJETIVO]:
        s = df[col]
        filas.append({
            "columna": col,
            "tipo": str(s.dtype),
            "mínimo": s.min(),
            "máximo": s.max(),
            "media": round(s.mean(), 3),
            "faltantes": int(s.isna().sum()),
            "valores únicos": int(s.nunique()),
        })
    return pd.DataFrame(filas)


def comparacion_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    tabla = df.groupby(config.COLUMNA_OBJETIVO)[config.VARIABLES].mean().T
    tabla.columns = ["media sin riesgo (0)", "media en riesgo (1)"]
    tabla["diferencia"] = tabla["media en riesgo (1)"] - tabla["media sin riesgo (0)"]
    tabla["correlación con en_riesgo"] = df[config.VARIABLES].corrwith(df[config.COLUMNA_OBJETIVO])
    return tabla.round(3)


def graficas_eda(df: pd.DataFrame) -> list[str]:
    nombres = []
    for col in config.VARIABLES:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        grupos = [df.loc[df[config.COLUMNA_OBJETIVO] == c, col] for c in (0, 1)]
        caja = ax.boxplot(grupos, patch_artist=True, widths=0.5)
        ax.set_xticks([1, 2], ["Sin riesgo (0)", "En riesgo (1)"])
        for parche, color in zip(caja["boxes"], (COLOR_SIN, COLOR_RIESGO)):
            parche.set_facecolor(color)
            parche.set_alpha(0.6)
        ax.set_title(f"{config.NOMBRES_CLAROS[col].capitalize()} por clase")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)
        nombres.append(_guardar(fig, f"eda_{col}.png"))

    corr = df[config.VARIABLES + [config.COLUMNA_OBJETIVO]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlaciones")
    nombres.append(_guardar(fig, "eda_correlaciones.png"))
    return nombres


# ---------------------------------------------------------------------------
# Fases 3–4 — gráficas de modelos
# ---------------------------------------------------------------------------
def grafica_roc_cv(resultados, y_ent) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for r, color in zip(resultados, (COLOR_SIN, COLOR_RIESGO)):
        fpr, tpr, _ = roc_curve(y_ent, r.probabilidades_cv)
        ax.plot(fpr, tpr, color=color, label=f"{r.nombre} (AUC {r.cv_media['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Recall (tasa de verdaderos positivos)")
    ax.set_title("Curva ROC — validación cruzada (80 %)")
    ax.legend(loc="lower right", fontsize=8)
    return _guardar(fig, "roc_validacion_cruzada.png")


def grafica_roc_prueba(y_pru, prob_pru, auc: float) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fpr, tpr, _ = roc_curve(y_pru, prob_pru)
    ax.plot(fpr, tpr, color=COLOR_RIESGO, label=f"Modelo elegido (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Recall")
    ax.set_title("Curva ROC — conjunto de prueba (20 %)")
    ax.legend(loc="lower right", fontsize=8)
    return _guardar(fig, "roc_prueba.png")


def grafica_matriz(matriz) -> str:
    matriz = np.asarray(matriz)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(matriz, cmap="Blues")
    etiquetas = ["Sin riesgo", "En riesgo"]
    ax.set_xticks([0, 1], etiquetas)
    ax.set_yticks([0, 1], etiquetas)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matriz[i, j], ha="center", va="center", fontsize=14,
                    color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    ax.set_title("Matriz de confusión — prueba")
    return _guardar(fig, "matriz_confusion_prueba.png")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _tabla_md(df: pd.DataFrame, indice: bool = False) -> str:
    if indice:
        df = df.rename_axis("variable").reset_index()
    cabecera = "| " + " | ".join(map(str, df.columns)) + " |"
    separador = "|" + "---|" * len(df.columns)
    filas = ["| " + " | ".join(map(str, fila)) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cabecera, separador, *filas])


def _pct(v: float) -> str:
    return f"{v:.3f}"


def generar_reporte(salida: dict) -> str:
    df = salida["df"]
    y_ent, y_pru = salida["y_ent"], salida["y_pru"]
    resultados, elegido = salida["resultados"], salida["elegido"]
    m = []

    m.append("# Reporte de entrenamiento — Clasificador de alumnos en riesgo\n")
    m.append("> Generado automáticamente por `python -m src.entrenamiento`.\n")

    # --- 1. Análisis de datos -------------------------------------------
    conteo = df[config.COLUMNA_OBJETIVO].value_counts()
    m.append("## 1. Análisis de datos\n")
    m.append(f"- Registros: **{len(df)}** · columnas: **{df.shape[1]}**")
    m.append(f"- Valores faltantes: **{int(df.isna().sum().sum())}**")
    m.append(f"- Filas duplicadas: **{int(df.duplicated().sum())}**")
    m.append(f"- Clase objetivo: {conteo.get(0, 0)} sin riesgo ({conteo.get(0, 0) / len(df):.0%}) · "
             f"{conteo.get(1, 0)} en riesgo ({conteo.get(1, 0) / len(df):.0%}) → desbalance moderado.\n")
    m.append("### Columnas\n")
    m.append(_tabla_md(resumen_columnas(df)) + "\n")
    m.append("### Comparación entre clases\n")
    m.append(_tabla_md(comparacion_por_clase(df), indice=True) + "\n")
    m.append("Lectura: los alumnos en riesgo estudian menos horas, tienen menor asistencia y menor "
             "promedio anterior. La participación en clase casi no se relaciona con el riesgo.\n")
    for nombre in graficas_eda(df):
        m.append(f"![{nombre}]({nombre})")
    m.append("")

    # --- 2. Comparación de modelos --------------------------------------
    m.append("## 2. Comparación de modelos\n")
    m.append(f"- División estratificada 80/20 (semilla {config.SEMILLA}): **{len(y_ent)}** entrenamiento "
             f"({y_ent.mean():.1%} en riesgo) · **{len(y_pru)}** prueba ({y_pru.mean():.1%} en riesgo).")
    m.append(f"- Validación cruzada estratificada **repetida {config.N_REPETICIONES} × {config.N_PLIEGUES}** "
             f"({config.N_REPETICIONES * config.N_PLIEGUES} pliegues) dentro del 80 %. "
             "Hiperparámetros elegidos con `GridSearchCV` (ROC-AUC). Ambos con `class_weight=\"balanced\"`.")
    m.append("- **Umbral:** con las probabilidades fuera de pliegue de las "
             f"{config.N_REPETICIONES} repeticiones se elige el umbral de mayor precisión que mantiene "
             f"recall ≥ {config.META_RECALL_UMBRAL:.2f} = meta {config.META_RECALL} + margen "
             f"{config.MARGEN_RECALL} (≈ 1 desviación estándar del recall entre pliegues), para no "
             "quedar justo en el límite de la meta.")
    m.append("- Las métricas de la tabla son media ± desviación de los "
             f"{config.N_REPETICIONES * config.N_PLIEGUES} pliegues con ese umbral.\n")
    filas = []
    for r in resultados:
        fila = {"modelo": r.nombre, "hiperparámetros": r.mejores_parametros, "umbral": f"{r.umbral:.3f}"}
        for k in ("recall", "precision", "f1", "roc_auc"):
            fila[k] = f"{r.cv_media[k]:.3f} ± {r.cv_desviacion[k]:.3f}"
        fila["cumple metas"] = "Sí" if r.cumple_metas else "No"
        filas.append(fila)
    m.append(_tabla_md(pd.DataFrame(filas)) + "\n")
    m.append(f"![roc_cv]({grafica_roc_cv(resultados, y_ent)})\n")

    if elegido is None:
        m.append("**Ningún modelo cumple las metas en validación cruzada.** Según RM-07 se debe "
                 "analizar y proponer otro método de aprendizaje antes de continuar.\n")
        ruta = config.DIR_REPORTES / "reporte_entrenamiento.md"
        ruta.write_text("\n".join(m), encoding="utf-8")
        return str(ruta)

    m.append(f"**Selección:** cumplen las metas: "
             f"{', '.join(r.nombre for r in resultados if r.cumple_metas)}. "
             f"Gana **{elegido.nombre}** por mayor precisión en validación cruzada "
             f"({elegido.cv_media['precision']:.3f}).\n")

    # --- 3. Evaluación final --------------------------------------------
    final = salida["metricas_prueba"]
    meta = salida["metadata"]
    m.append("## 3. Evaluación final (20 % de prueba)\n")
    m.append("> **Transparencia:** esta es la **segunda** evaluación sobre el conjunto de prueba. En la "
             "primera, con el umbral elegido en el límite de la meta (CV simple de 5 pliegues, umbral "
             "0.690), el modelo obtuvo recall 0.733, precisión 0.846 y ROC-AUC 0.923. Después se cambió "
             "el **procedimiento** de elección de umbral (CV repetida + margen fijo justificado por la "
             "variación entre pliegues), sin ajustar el umbral a los datos de prueba. Aun así, como la "
             "prueba ya se había visto, el resultado debe leerse con cautela.\n")
    m.append(f"Modelo: **{elegido.nombre}** · hiperparámetros {elegido.mejores_parametros} · "
             f"umbral **{elegido.umbral:.3f}**\n")
    m.append("| métrica | prueba | meta | ¿cumple? |\n|---|---|---|---|")
    m.append(f"| Recall | {_pct(final['recall'])} | ≥ {config.META_RECALL} | "
             f"{'Sí' if final['recall'] >= config.META_RECALL else '**No**'} |")
    m.append(f"| ROC-AUC | {_pct(final['roc_auc'])} | ≥ {config.META_ROC_AUC} | "
             f"{'Sí' if final['roc_auc'] >= config.META_ROC_AUC else '**No**'} |")
    m.append(f"| Precisión | {_pct(final['precision'])} | — | — |")
    m.append(f"| F1 | {_pct(final['f1'])} | — | — |\n")
    (vn, fp), (fn, vp) = meta["matriz_confusion_prueba"]
    m.append(f"Matriz de confusión: {vp} en riesgo detectados, **{fn} en riesgo no detectados** "
             f"(falsos negativos), {fp} citados de más (falsos positivos), {vn} correctamente sin riesgo.\n")
    m.append(f"![matriz]({grafica_matriz(meta['matriz_confusion_prueba'])}) "
             f"![roc_prueba]({grafica_roc_prueba(y_pru, salida['prob_pru'], final['roc_auc'])})\n")

    if elegido.clave == "regresion_logistica":
        coef = salida["modelo"].named_steps["modelo"].coef_[0]
        pesos = pd.DataFrame({"variable": config.VARIABLES, "peso aprendido (estandarizado)": coef.round(3)})
        pesos = pesos.reindex(pesos["peso aprendido (estandarizado)"].abs().sort_values(ascending=False).index)
        m.append("### Pesos aprendidos\n")
        m.append("Peso negativo = a menor valor, mayor riesgo.\n")
        m.append(_tabla_md(pesos) + "\n")

    # --- 4. Conclusión ---------------------------------------------------
    m.append("## 4. Conclusión\n")
    if meta["cumple_metas_prueba"]:
        m.append(f"El modelo **{elegido.nombre}** cumple las metas en el conjunto de prueba "
                 f"(recall {final['recall']:.3f}, ROC-AUC {final['roc_auc']:.3f}) y queda guardado en "
                 "`modelos/modelo.joblib`. "
                 f"El recall queda justo en la meta ({vp} de {fn + vp} alumnos en riesgo detectados); con "
                 f"solo {fn + vp} alumnos en riesgo en la prueba, un alumno más o menos lo mueve "
                 f"~{1 / (fn + vp):.1%}. El margen del umbral sacrifica precisión ({final['precision']:.3f}: "
                 f"{fp} alumnos citados de más) a favor de no dejar de detectar alumnos en riesgo, "
                 "en línea con RM-03. Conviene confirmar el desempeño con datos reales del siguiente "
                 "periodo.")
    else:
        m.append(f"El modelo **{elegido.nombre}** cumple ambas metas en validación cruzada y tiene un "
                 f"ROC-AUC de prueba alto ({final['roc_auc']:.3f}), pero su **recall en prueba "
                 f"({final['recall']:.3f}) queda por debajo de la meta {config.META_RECALL}**: "
                 f"{fn} de {fn + vp} alumnos en riesgo no se detectaron. Con solo {fn + vp} alumnos en "
                 "riesgo en la prueba, cada uno vale ~"
                 f"{1 / (fn + vp):.1%} de recall, y el umbral se eligió justo en el límite de la meta "
                 "en validación cruzada, por lo que el margen era nulo. "
                 "**Se requiere decidir cómo proceder** (no se reajusta el umbral con los datos de prueba "
                 "para no invalidar la evaluación).")
    m.append("")

    ruta = config.DIR_REPORTES / "reporte_entrenamiento.md"
    ruta.write_text("\n".join(m), encoding="utf-8")
    return str(ruta)

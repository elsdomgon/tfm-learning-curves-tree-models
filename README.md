<p align="center">
<h1 align="center">Análisis y predicción de curvas de aprendizaje de modelos basados en árboles</h1>
</p>
<p align="center">
    <em>Trabajo Fin de Máster · Máster Universitario en Lógica, Computación e Inteligencia Artificial · Universidad de Sevilla</em>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=default&logo=scikitlearn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/Optuna-4B8BBE.svg?style=default&logo=python&logoColor=white" alt="Optuna">
    <img src="https://img.shields.io/badge/LightGBM-9BCD5A.svg?style=default&logo=python&logoColor=white" alt="LightGBM">
</p>
<br>

## Índice

- [Resumen](#resumen)
- [Corpus](#corpus)
- [Estructura del repositorio](#estructura-del-repositorio)

---

## Resumen

La **optimización de hiperparámetros** es una etapa costosa pero necesaria en el desarrollo de modelos de aprendizaje automático. Completar el entrenamiento de cientos de configuraciones puede suponer un gasto computacional enorme, especialmente cuando se trabaja con grandes volúmenes de datos.

Una forma de reducir este coste consiste en descartar las configuraciones menos prometedoras antes de que finalicen su ejecución. La **predicción de curvas de aprendizaje** permite estimar el rendimiento final de un modelo observando exclusivamente el inicio de su entrenamiento. Si bien este enfoque ha sido ampliamente estudiado en el aprendizaje profundo, su aplicación en los algoritmos de **potenciación de gradiente** es un ámbito apenas explorado.

Este trabajo analiza si dicha estimación puede aprovecharse para construir un **mecanismo de interrupción temprana** durante el ajuste de hiperparámetros. Para ello, se recopilan curvas de aprendizaje de optimizaciones reales bajo dos estrategias de búsqueda distintas, se desarrolla y valida un modelo encargado de predecir el valor final de las curvas a partir de sus primeros puntos, y se evalúa el impacto de este descarte selectivo de configuraciones mediante simulación utilizando veinte problemas de clasificación binaria.

Los experimentos muestran que es posible anticipar con suficiente fiabilidad el comportamiento final de una combinación de hiperparámetros a partir de pocas iteraciones iniciales. El mecanismo propuesto logra **ahorros computacionales superiores al 60 %** en ambas estrategias de búsqueda, con una **degradación media del rendimiento en torno al 2 %**, lo que representa un compromiso altamente favorable entre la aceleración del proceso y la calidad predictiva resultante.

**Palabras clave:** Optimización de hiperparámetros, aprendizaje automático, potenciación de gradiente, predicción de curvas de aprendizaje, interrupción temprana.

---

## Corpus

Los datos provienen de la colección **AutoML Benchmark All Classification (_suite_ 271)** de OpenML, el estándar _de facto_ para la evaluación de sistemas de aprendizaje automático automatizado. Tras aplicar filtros de clasificación binaria y restricciones de tamaño (1.000–50.000 instancias, máximo 350 columnas), se han obtenido **20 conjuntos de datos** finales para la ejecución de los experimentos.

En cada conjunto, se han ejecutado dos estudios de Optuna independientes:

| Estrategia | Pruebas por dataset | Pliegues CV | Total de instancias |
|------------|--------------------:|:-----------:|--------------------:|
| Búsqueda aleatoria (`RandomSampler`) | 500 | 10 | 100.000 |
| Optimización bayesiana (`TPESampler`) | 200 | 10 | 40.000 |

Cada fila del metaconjunto resultante representa un pliegue individual e incluye los hiperparámetros evaluados, la pérdida final de validación y la serie temporal completa de la curva de aprendizaje (entropía cruzada binaria en entrenamiento y validación, iteración a iteración).

Los metaconjuntos generados (`dataset_curvas_pliegues_random.csv` y `dataset_curvas_pliegues_TPE_N200.csv`) no están incluidos en el repositorio por su tamaño, pero pueden reproducirse ejecutando los _scripts_ de `fase1_recogida_datos/`.

---

## Estructura del repositorio

```
tfm-learning-curves-tree-models/
│
├── fase1_recogida_datos/
│   ├── seleccion_datasets.ipynb              # Consulta a OpenML, filtrado y selección de los 20 conjuntos de datos
│   ├── generador_curvas_pliegues_random.py   # Entrenamiento de LightGBM con búsqueda aleatoria (500 pruebas × 10 pliegues × 20 datasets) y registro de curvas de aprendizaje
│   └── generador_curvas_pliegues_TPE.py      # Ídem con optimización bayesiana TPE (200 pruebas × 10 pliegues × 20 datasets)
│
├── fase2_predictor_curvas/
│   ├── analisisPreliminarT_random.py         # Optimización de hiperparámetros de la LSTM para cada T ∈ {5, 10, 15, 20} para el metaconjunto obtenido con búsqueda aleatoria; guarda resultados en resultados_analisisPreliminarT/
│   ├── analisisPreliminarT_TPE.py            # Ídem para TPE
│   ├── entrenar_predictor_random.py          # Entrena los 3 predictores finales (uno por pliegue de validación cruzada) para el metaconjunto obtenido con búsqueda aleatoria; guarda modelos en modelos_random/
│   ├── entrenar_predictor_TPE.py             # Ídem para TPE; guarda modelos en modelos_TPE/
│   │
│   ├── resultados_analisisPreliminarT/       # CSVs con las métricas (RMSE, MAE, MAPE, R²) de cada configuración T evaluada (en media y por pliegue)
│   │
│   ├── modelos_random/                       # Predictores LSTM entrenados para el metaconjunto obtenido con búsqueda aleatoria
│   │   ├── modelos_T5/
│   │   │   ├── modelo_T5_fold0.keras  |  scaler_T5_fold0.joblib
│   │   │   ├── modelo_T5_fold1.keras  |  scaler_T5_fold1.joblib
│   │   │   └── modelo_T5_fold2.keras  |  scaler_T5_fold2.joblib
│   │   ├── modelos_T10/   # (misma estructura)
│   │   ├── modelos_T15/   # (misma estructura)
│   │   └── modelos_T20/   # (misma estructura)
│   │
│   └── modelos_TPE/                          # Predictores LSTM entrenados para el metaconjunto obtenido con el algoritmo TPE (misma estructura que modelos_random/)
│       ├── modelos_T5/
│       ├── modelos_T10/
│       ├── modelos_T15/
│       └── modelos_T20/
│
├── fase3_simulacion_poda/
│   ├── analisisSensibilidad_random.py        # Simula la poda para todas las combinaciones (T, E) para el metaconjunto obtenido con búsqueda aleatoria; guarda resultados en resultados_analisisSensibilidad/
│   ├── analisisSensibilidad_TPE.py           # Ídem para TPE
│   └── resultados_analisisSensibilidad/      # CSVs con el ahorro de iteraciones y la degradación del rendimiento para cada par (T, E) evaluado (en media y por pliegue)
│
└── visualizacion/
    ├── curvas_random.ipynb                   # Visualización de curvas de aprendizaje bajo búsqueda aleatoria
    ├── curvas_TPE.ipynb                      # Visualización de curvas de aprendizaje bajo TPE
    ├── variable_objetivo.ipynb               # Distribución de la variable objetivo (logloss final) por dataset
    ├── analisisPreliminarT.ipynb             # Análisis preliminar del rendimiento del predictor según T (en media y por pliegue)
    ├── analisisSensibilidad_random.ipynb   # Espacio ahorro–degradación y efecto de E para el metaconjunto obtenido con búsqueda aleatoria 
    ├── analisisSensibilidad_TPE.ipynb      # Ídem para TPE
    └── simulacionPoda.ipynb                # Resultados finales: ahorro e impacto en calidad sobre los 20 conjuntos de datos
```

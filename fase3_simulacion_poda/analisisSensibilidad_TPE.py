# =============================================================================
# EVALUACIÓN DEL SISTEMA DE PODA CON LSTM
# Bucle completo sobre T = [5, 10, 15, 20] y E = [-0.2, -0.1, 0, 0.1]
# Guarda dos CSVs:
#   - resultados_detalle_TPE.csv   → una fila por (T, E, fold, task_id)
#   - resultados_agregados_TPE.csv → una fila por (T, E), medias sobre los 20 datasets
# =============================================================================


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import joblib
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

dataset = pd.read_csv('dataset_curvas_pliegues_TPE_N200.csv')

# =============================================================================
# 2. PARÁMETROS DEL EXPERIMENTO
# =============================================================================

N      = 20
TARGET = 'valid_logloss_fold'
T_vals = [5, 10, 15, 20]
E_vals = [-0.2, -0.1, 0.0, 0.1]

resultados_detalle = []

# =============================================================================
# 3. BUCLE PRINCIPAL: T × E
# =============================================================================

for T in T_vals:
    print(f"\n{'='*60}")
    print(f"  T = {T}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 3.1 Filtrado del dataset para este T
    # -------------------------------------------------------------------------
    cota_minima = T + 5
    df = dataset[dataset['curves_length'] >= cota_minima].copy()

    print(f"Filas originales:     {len(dataset)}")
    print(f"Filas tras el filtro: {len(df)}")

    # -------------------------------------------------------------------------
    # 3.2 Validación cruzada por grupos
    # -------------------------------------------------------------------------
    groups = df['task_id']
    gkf = GroupKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(gkf.split(df, groups=groups))

    # -------------------------------------------------------------------------
    # 3.3 Precarga de modelos y scalers para este T (una vez por T)
    # -------------------------------------------------------------------------
    modelos = {}
    scalers = {}
    for fold in range(3):
        scalers[fold] = joblib.load(f'scaler_T{T}_fold{fold}.joblib')
        modelos[fold] = tf.keras.models.load_model(f'modelo_T{T}_fold{fold}.keras')
        print(f"  Modelo y scaler cargados para fold {fold}")

    # =========================================================================
    # 4. BUCLE SOBRE E
    # =========================================================================

    for E in E_vals:
        print(f"\n  --- E = {E} ---")

        # =====================================================================
        # 5. BUCLE POR FOLD
        # =====================================================================

        for fold, (train_idx, test_idx) in enumerate(splits):

            df_test = df.iloc[test_idx]
            scaler_cargado = scalers[fold]
            modelo_cargado = modelos[fold]

            # =================================================================
            # 6. SIMULACIÓN POR TAREA
            # =================================================================

            for task_id in df_test['task_id'].unique():

                df_task = df_test[df_test['task_id'] == task_id].copy()
                trials_disponibles = sorted(df_task['trial_id'].unique())

                mejor_rendimiento   = float('inf')
                n_iter_poda         = 0
                n_iter_real         = 0
                errores_calibracion = []
                sesgo               = 0

                for i, t_id in enumerate(trials_disponibles):
                    trial_data = df_task[df_task['trial_id'] == t_id]

                    rendimiento_real_trial = trial_data[TARGET].mean()
                    coste_real_total       = trial_data['curves_length'].sum()
                    n_iter_real           += coste_real_total

                    # Preparación del input al LSTM
                    puntos_train = trial_data[[f'train_{j}' for j in range(1, T+1)]].mean().values
                    puntos_val   = trial_data[[f'val_{j}'   for j in range(1, T+1)]].mean().values

                    X_trial  = np.concatenate([puntos_train, puntos_val]).reshape(1, -1)
                    X_scaled = scaler_cargado.transform(X_trial.reshape(-1, 1)).reshape(X_trial.shape)
                    X_3d     = np.stack([X_scaled[:, :T], X_scaled[:, T:]], axis=-1)

                    pred_logloss = modelo_cargado.predict(X_3d, verbose=0).flatten()[0]

                    # Lógica de poda
                    if i < N:
                        # FASE INICIAL
                        n_iter_poda += coste_real_total

                        if rendimiento_real_trial < mejor_rendimiento:
                            mejor_rendimiento = rendimiento_real_trial

                        errores_calibracion.append(rendimiento_real_trial - pred_logloss)

                        if i == N - 1:
                            errores_sin_extremos = sorted(errores_calibracion)[1:-1]
                            sesgo = np.mean(errores_sin_extremos)

                    else:
                        # FASE DE PODA
                        pred_logloss_corregida = pred_logloss + sesgo

                        if pred_logloss_corregida < mejor_rendimiento * (1 + E):
                            n_iter_poda += coste_real_total
                            if rendimiento_real_trial < mejor_rendimiento:
                                mejor_rendimiento = rendimiento_real_trial
                        else:
                            n_iter_poda += (T * 10)

                # Registro de resultados
                mejor_logloss_real  = df_task.groupby('trial_id')[TARGET].mean().min()
                logloss_perdido_pct = round(
                    100 * (mejor_rendimiento - mejor_logloss_real) / mejor_logloss_real, 4
                )

                resultados_detalle.append({
                    'T':                  T,
                    'E':                  E,
                    'fold_cv':            fold,
                    'task_id':            task_id,
                    'logloss_poda':       round(mejor_rendimiento, 6),
                    'logloss_real':       round(mejor_logloss_real, 6),
                    'logloss_perdido(%)': logloss_perdido_pct,
                    'n_iter_poda':        n_iter_poda,
                    'n_iter_real':        n_iter_real,
                    'iter_ahorradas(%)':  round(100 * (1 - n_iter_poda / n_iter_real), 4),
                    'sesgo_calibracion':  round(float(sesgo), 6)
                })

    # Liberar memoria al terminar cada T
    for fold in range(3):
        del modelos[fold], scalers[fold]
    tf.keras.backend.clear_session()
    print(f"\n  Memoria liberada para T = {T}")

# =============================================================================
# 7. GUARDADO DE RESULTADOS
# =============================================================================

df_detalle = pd.DataFrame(resultados_detalle)
df_detalle.to_csv('resultados_detalle_TPE.csv', index=False)
print("\nGuardado: resultados_detalle_TPE.csv")

# Tabla agregada: media y desviación típica sobre los 20 datasets
df_agregado = (
    df_detalle
    .groupby(['T', 'E'])
    .agg(
        logloss_perdido_medio   = ('logloss_perdido(%)', 'mean'),
        logloss_perdido_std     = ('logloss_perdido(%)', 'std'),
        iter_ahorradas_media    = ('iter_ahorradas(%)',  'mean'),
        iter_ahorradas_std      = ('iter_ahorradas(%)',  'std'),
        n_datasets              = ('task_id',            'nunique')
    )
    .round(4)
    .reset_index()
)

df_agregado.to_csv('resultados_agregados_TPE.csv', index=False)
print("Guardado: resultados_agregados_TPE.csv")

print("\n" + "="*60)
print("  PROCESO COMPLETO FINALIZADO")
print("="*60)

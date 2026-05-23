import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import optuna
from optuna.importance import get_param_importances


# LECTURA DE LOS DATOS
dataset = pd.read_csv('dataset_curvas_pliegues_TPE_N200.csv')


# LSTM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


## --- OPTUNA ---
# Silenciamos los prints de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def lstm_model(X_data, T, df, y, seed=42):
    # Fijamos semillas
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Validación cruzada 
    gkf = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
    
    # Optuna
    def objective(trial):
        # Limpiar memoria de Keras al inicio de cada trial para evitar ralentización
        tf.keras.backend.clear_session()
        
        print(f"Ejecutando trial: {trial.number}")
        
        # Hiperparámetros
        lstm_units = trial.suggest_int('lstm_units', 12, 60, step=4)
        dense_units = trial.suggest_int('dense_units', 8, 60, step=4)
        dropout = trial.suggest_float('dropout', 0.05, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        
        # Cross-validation manual
        fold_rmses = []
        fold_maes = []
        fold_mapes = []
        fold_r2s = []
        
        for train_idx, test_idx in gkf.split(X_data, y, groups=df['task_id']):
            # 1. Dividir MUESTRAS
            X_cv_train, y_cv_train = X_data.iloc[train_idx], y.iloc[train_idx] # 2/3 datasets para entrenar modelo
            X_cv_test, y_cv_test = X_data.iloc[test_idx], y.iloc[test_idx]     # 1/3 datasets para evaluar modelo
            
            # 2. StandardScaler()
            # Escalar X (global para mantener tendencia temporal)
            scaler_x = StandardScaler()
            scaler_x.fit(X_cv_train.values.reshape(-1, 1))
            
            X_cv_train_scaled = scaler_x.transform(X_cv_train.values.reshape(-1, 1)).reshape(X_cv_train.shape)
            X_cv_test_scaled = scaler_x.transform(X_cv_test.values.reshape(-1, 1)).reshape(X_cv_test.shape)
            
            # 3. Convertir a 3D: (n_samples, T_iteraciones, 2_curvas)
            # Cada paso temporal tiene [logloss_train_iter_i, logloss_val_iter_i]
            X_cv_train_3d = np.stack([
                X_cv_train_scaled[:, :T],   # logloss en TRAIN iteraciones 1-T
                X_cv_train_scaled[:, T:]    # logloss en VAL iteraciones 1-T
            ], axis=-1)  # Shape: (n_train_cv, T, 2)
            
            X_cv_test_3d = np.stack([
                X_cv_test_scaled[:, :T],    # logloss en TRAIN iteraciones 1-T
                X_cv_test_scaled[:, T:]     # logloss en VAL iteraciones 1-T
            ], axis=-1)  # Shape: (n_test_cv, T, 2)
            
            # 4. Modelo LSTM
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(T, 2)), # T pasos de tiempo, 2 variables
                tf.keras.layers.LSTM(lstm_units, activation='tanh'),
                tf.keras.layers.Dropout(dropout),     
                tf.keras.layers.Dense(dense_units, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse')
            
            # 5. Entrenar con early stopping
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True, verbose=0
            )
            
            model.fit(
                X_cv_train_3d, y_cv_train, 
                validation_data=(X_cv_test_3d, y_cv_test), 
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # 6. Predicción 
            y_pred = model.predict(X_cv_test_3d, verbose=0).flatten() # flatten para evitar problemas de dimensiones
            
            # Calcular métricas 
            fold_rmses.append(np.sqrt(mean_squared_error(y_cv_test, y_pred)))
            fold_maes.append(mean_absolute_error(y_cv_test, y_pred))
            fold_mapes.append(mean_absolute_percentage_error(y_cv_test, y_pred))
            fold_r2s.append(r2_score(y_cv_test, y_pred))
            
            # Limpiar memoria de Keras después de cada fold
            tf.keras.backend.clear_session()
        
        # Guardar métricas en el trial
        trial.set_user_attr('mae_mean', np.mean(fold_maes))
        trial.set_user_attr('mae_std', np.std(fold_maes))
        trial.set_user_attr('mape_mean', np.mean(fold_mapes))
        trial.set_user_attr('mape_std', np.std(fold_mapes))
        trial.set_user_attr('r2_mean', np.mean(fold_r2s))
        trial.set_user_attr('r2_std', np.std(fold_r2s))
        trial.set_user_attr('rmse_std', np.std(fold_rmses))
        trial.set_user_attr('rmse_folds', fold_rmses)
        trial.set_user_attr('mae_folds', fold_maes)
        trial.set_user_attr('mape_folds', fold_mapes)
        trial.set_user_attr('r2_folds', fold_r2s)
        
        return np.mean(fold_maes)
    
    # Estudio
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=50, n_jobs=8) 
    
    # --- Cálculo de importancia de hiperparámetros ---
    try:
        importances = get_param_importances(study)
        importances_dict = {k: round(v, 4) for k, v in importances.items()}
    except Exception as e:
        importances_dict = f"Error: {e}"

    # Extraer resultados del mejor trial
    best_trial = study.best_trial
    
    return {
        'T': T,
        'RMSE': study.best_value,
        'RMSE_std': best_trial.user_attrs['rmse_std'],
        'MAE': best_trial.user_attrs['mae_mean'],
        'MAE_std': best_trial.user_attrs['mae_std'],
        'MAPE': best_trial.user_attrs['mape_mean'],
        'MAPE_std': best_trial.user_attrs['mape_std'],
        'R2': best_trial.user_attrs['r2_mean'],
        'R2_std': best_trial.user_attrs['r2_std'],
        'Importancia_Params': importances_dict,
        'Parámetros': study.best_params,
        'RMSE_folds': best_trial.user_attrs['rmse_folds'],
        'MAE_folds': best_trial.user_attrs['mae_folds'],
        'MAPE_folds': best_trial.user_attrs['mape_folds'],
        'R2_folds': best_trial.user_attrs['r2_folds'],
    }

# --- EJECUCIÓN DEL EXPERIMENTO ---
valores_T = [5, 10, 15, 20]

resultados_lista = []

for T in valores_T:
    # Filtramos el dataframe según T
    cota_minima = T + 5
    df = dataset[dataset['curves_length'] >= cota_minima].copy()
    y = df['valid_logloss_fold']

    # Bloques de variables predictoras
    X_train_puntos = df[[f'train_{i}' for i in range(1, T+1)]]
    X_val_puntos = df[[f'val_{i}' for i in range(1, T+1)]]
    X_A = pd.concat([X_train_puntos, X_val_puntos], axis=1)

    experimentos = [
        (X_A, "Curvas")
    ]

    for X_input, nombre in experimentos:
        print(f"Entrenando modelo con: {nombre}, T={T}...")
        res = lstm_model(X_input, T, df, y)
        resultados_lista.append(res)

# Convertimos los resultados a dataframe de pandas
df_resultados = pd.DataFrame(resultados_lista)
print("\nProceso finalizado")

# --- CSV PRINCIPAL (media + std) ---
df_resultados_LSTM = df_resultados.drop(columns=['RMSE_folds', 'MAE_folds', 'MAPE_folds', 'R2_folds'])
df_resultados_LSTM = df_resultados_LSTM.sort_values(by=['T', 'RMSE'])
df_resultados_LSTM.to_csv("resultados_LSTM_efectoT_TPE.csv", index=False)

# --- CSV POR FOLDS ---
filas_folds = []
for res in resultados_lista:
    for fold_idx, (rmse, mae, mape, r2) in enumerate(zip(
        res['RMSE_folds'], res['MAE_folds'], res['MAPE_folds'], res['R2_folds']
    )):
        filas_folds.append({
            'T': res['T'],
            'Fold': fold_idx,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

pd.DataFrame(filas_folds).to_csv("resultados_LSTM_folds_efectoT_TPE.csv", index=False)

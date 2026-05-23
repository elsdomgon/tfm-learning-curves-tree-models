import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import joblib


# LECTURA DE LOS DATOS
dataset = pd.read_csv('dataset_curvas_pliegues_random.csv')
resultados_optuna = pd.read_csv('resultados_LSTM_efectoT_random.csv')


# --- LISTA DE T A ENTRENAR ---
lista_T = [5, 10, 15, 20]

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def entrenar_lstm(X_data, y, groups, best_params, T, output_dir, seed=42):
    """
    Entrena por cada fold y guarda el modelo (.keras) y el scaler (.joblib)
    en la carpeta output_dir.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gkf = GroupKFold(n_splits=3, shuffle=True, random_state=seed)

    lstm_units = int(best_params['lstm_units'])
    dense_units = int(best_params['dense_units'])
    dropout = float(best_params['dropout'])
    lr = float(best_params['learning_rate'])
    bs = int(best_params['batch_size'])

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_data, y, groups=groups)):
        print(f"\n--- Procesando FOLD {fold} ---")

        X_train, y_train = X_data.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_data.iloc[val_idx], y.iloc[val_idx]

        scaler_x = StandardScaler()
        scaler_x.fit(X_train.values.reshape(-1, 1))

        X_train_s = scaler_x.transform(X_train.values.reshape(-1, 1)).reshape(X_train.shape)
        X_val_s = scaler_x.transform(X_val.values.reshape(-1, 1)).reshape(X_val.shape)

        X_train_3d = np.stack([X_train_s[:, :T], X_train_s[:, T:]], axis=-1)
        X_val_3d = np.stack([X_val_s[:, :T], X_val_s[:, T:]], axis=-1)

        tf.keras.backend.clear_session()

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(T, 2)),
            tf.keras.layers.LSTM(lstm_units, activation='tanh'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30, restore_best_weights=True, verbose=0
        )

        model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d, y_val),
            epochs=100,
            batch_size=bs,
            callbacks=[early_stop],
            verbose=1
        )

        # Guardar en la carpeta de output
        model_path = os.path.join(output_dir, f'modelo_T{T}_fold{fold}.keras')
        scaler_path = os.path.join(output_dir, f'scaler_T{T}_fold{fold}.joblib')

        model.save(model_path)
        joblib.dump(scaler_x, scaler_path)

        print(f"Guardados: {model_path} | {scaler_path}")


# BUCLE PRINCIPAL
for T in lista_T:
    print(f"\n{'='*50}")
    print(f"  ENTRENANDO MODELO PARA T = {T}")
    print(f"{'='*50}")

    # Buscar parámetros para este T
    fila = resultados_optuna[resultados_optuna['T'] == T]
    if fila.empty:
        print(f"[AVISO] No hay resultados para T={T} en el CSV. Saltando...")
        continue

    best_params = eval(fila.iloc[0]['Parámetros'])
    print(f"Parámetros cargados: {best_params}")

    # Filtrar dataset
    cota_minima = T + 5
    df = dataset[dataset['curves_length'] >= cota_minima].copy()
    y = df['valid_logloss_fold']

    X_train_puntos = df[[f'train_{i}' for i in range(1, T+1)]]
    X_val_puntos = df[[f'val_{i}' for i in range(1, T+1)]]
    X_A = pd.concat([X_train_puntos, X_val_puntos], axis=1)

    # Crear carpeta para este T
    output_dir = f'modelos_T{T}'
    os.makedirs(output_dir, exist_ok=True)

    entrenar_lstm(X_A, y, df['task_id'], best_params, T, output_dir)

print("\nEntrenamiento completado para todos los valores de T.")

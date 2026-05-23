import openml
import optuna

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


### PREPROCESADOR DE DATOS
def preprocesar_datos(df, null_threshold=0.75):
    df = df.copy()

    # 1. SUSTITUIR COLUMNAS CON MUCHOS NaN POR VARIABLES BINARIAS (nulo/no_nulo)
    null_percentages = df.isna().sum() / len(df)
    cols_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()

    if cols_to_drop:
        new_cols = {f"{col}_is_null": df[col].isna().astype(int) for col in cols_to_drop}
        df_indicators = pd.DataFrame(new_cols)
        df = pd.concat([df, df_indicators], axis=1)
        df = df.drop(cols_to_drop, axis=1)

    # 2. IDENTIFICACIÓN DE VARIABLES CATEGÓRICAS Y NUMÉRICAS
    # Variables categóricas
    categorical_cols = []
    numeric_disguised_as_categorical = []

    for col in df.select_dtypes(exclude=[np.number]).columns.tolist():
        try:
            non_null_values = df[col].dropna()
            pd.to_numeric(non_null_values)
            numeric_disguised_as_categorical.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)

    # Variables numéricas (sin contar las columnas _is_null)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.endswith('_is_null')]
    numeric_cols = numeric_cols + numeric_disguised_as_categorical

    for col in numeric_disguised_as_categorical:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. GESTIÓN DE LOS VALORES FALTANTES
    # Imputación de la mediana en variables numéricas
    for col in numeric_cols:
        if df[col].isna().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    # Imputación de la moda en variables categóricas
    for col in categorical_cols:
        if df[col].isna().any():
            moda = df[col].mode()[0] if len(df[col].mode()) > 0 else None
            df[col] = df[col].fillna(moda)

    # 4. CODIFICACIÓN DE CATEGÓRICAS
    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes.astype('category')

    return df


### GENERACIÓN DE CURVAS (CSV)
## Benchmark de OpenML
suite = openml.study.get_suite(271)
tasks = suite.tasks # Son 71 tasks en total

# Clasificación binaria
filtro = []

for task_id in tasks:
    task = openml.tasks.get_task(task_id)

    # Filtro de clasificación binaria 
    if len(task.class_labels) != 2: 
        continue

    dataset = task.get_dataset()
    q = dataset.qualities
    n, m = int(q['NumberOfInstances']), int(q['NumberOfFeatures'])

    # Filtro de tamaño:
    # 1. Filas
    if n < 1000 or n >= 50000:
        continue
    
    # 2. Columnas
    if m > 350:
        continue

    filtro.append(task.task_id)

filtered_tasks = [x for x in filtro if x != 359992]  # Nos quedamos con 20
print(len(filtered_tasks))

## Función objetivo de Optuna
def objective(trial, task, X, y):
    # HIPERPARÁMETROS
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': 25, 
        # Tasa de aprendizaje
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 
        # Estructura del árbol
        'num_leaves': trial.suggest_int('num_leaves', 8, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        # Control de sobreajuste 
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.5),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        # Regularización 
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    MAX_ITER = 1000
    EARLY_STOPPING = 50

    # VALIDACIÓN CRUZADA
    fold_curves_train = []
    fold_curves_val = []
    fold_best_scores = []
    fold_best_iters = []

    # Splits proporcionados por OpenML
    _, n_folds, _ = task.get_split_dimensions() # Todos tienen 10 pliegues

    for fold_idx in range(n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold_idx, repeat=0)
        X_train, X_val = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_val = y[train_indices], y[test_indices]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        evals_result = {}

        # Definimos el modelo
        model = lgb.train(
            param,
            train_data,
            num_boost_round = MAX_ITER,
            valid_sets = [train_data, val_data],
            valid_names = ['train', 'valid'],
            callbacks = [
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                lgb.record_evaluation(evals_result)
            ]
        )

        c_train = evals_result['train']['binary_logloss']
        c_val = evals_result['valid']['binary_logloss']
        best_it = model.best_iteration

        # Recortamos la curva hasta su mejor iteración
        fold_curves_train.append(c_train[:best_it])
        fold_curves_val.append(c_val[:best_it])
        fold_best_scores.append(model.best_score['valid']['binary_logloss'])
        fold_best_iters.append(best_it)

    # Guardamos los datos de cada pliegue individualmente
    trial.set_user_attr("fold_curves_train", fold_curves_train)
    trial.set_user_attr("fold_curves_val", fold_curves_val)
    trial.set_user_attr("fold_best_scores", fold_best_scores)
    trial.set_user_attr("fold_best_iters", fold_best_iters)
    
    # VALOR A OPTIMIZAR
    # Devolvemos la media de los mejores logloss (sobre validación)
    return np.mean(fold_best_scores)

## Generación del dataset
all_curves_data = []

for task_id in filtered_tasks:
    try:
        # Extraemos el dataset asociado y preprocesamos los datos
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=task.target_name)
        X_procesado = preprocesar_datos(X, null_threshold=0.75)
        y = LabelEncoder().fit_transform(y)
        X_shape = X_procesado.shape
        
        print(f"Procesando Task: {task_id}")

        # Estudio de Optuna (algoritmo TPE, exploración = 20 trials)
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                seed=42, 
                n_startup_trials=20
            )
        )

        # Ejecutar trials
        study.optimize(
            lambda trial: objective(trial, task, X_procesado, y),
            n_trials = 200 
        )

        # Resultados
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Extraemos los datos de los 10 folds
                folds_train = trial.user_attrs['fold_curves_train']
                folds_val = trial.user_attrs['fold_curves_val']
                folds_scores = trial.user_attrs['fold_best_scores']
                folds_iters = trial.user_attrs['fold_best_iters']

                # Iteramos sobre cada fold para generar una fila por pliegue
                for fold_idx in range(len(folds_train)):
                    res = {
                        'task_id': task_id,
                        'X_shape': X_shape,
                        'trial_id': trial.number,
                        'fold_id': fold_idx,
                        **trial.params,
                        'valid_logloss_fold': folds_scores[fold_idx],
                        'curves_length': folds_iters[fold_idx] 
                    }

                    c_train = folds_train[fold_idx]
                    c_val = folds_val[fold_idx]

                    for i in range(len(c_train)):
                        res[f"train_{i+1}"] = c_train[i]
                        res[f"val_{i+1}"] = c_val[i]

                    all_curves_data.append(res)

    except Exception as e:
        print(f"Error en el task {task_id}: {e}")
        continue

# Convertir a DataFrame y guardar
df_final = pd.DataFrame(all_curves_data)
df_final = df_final.round(5)
df_final.to_csv("dataset_curvas_pliegues_TPE_N200.csv", index=False)
print("Proceso finalizado. Archivo guardado.")

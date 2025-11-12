import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

# --- Importazione di tutti i modelli necessari ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Ignora i warning di Scikit-learn per maggiore pulizia
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def tune_model(model_name: str, X_train: pd.DataFrame, Y_train: pd.Series):
    """
    Esegue la GridSearchCV per un modello specifico e salva automaticamente
    scaler e modello nella cartella 'roba', dentro la directory del file.
    """

    print(f"\n--- Inizio Ottimizzazione per: {model_name} ---")

    # ============================================================
    # Percorso assoluto della cartella "roba"
    # ============================================================
    base_dir = os.path.dirname(__file__)                # percorso di questo file
    roba_dir = os.path.join(base_dir, "generated_models")           # crea path a 'roba'
    os.makedirs(roba_dir, exist_ok=True)                # crea la cartella se non esiste

    # ============================================================
    # Scaling Condizionale
    # ============================================================
    if model_name in ['logistic_regression', 'knn']:
        print("Scaling dati di addestramento...")
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)

        # Salvataggio scaler
        scaler_filename = os.path.join(roba_dir, f"{model_name}_scaler.pkl")
        print(f"Salvataggio scaler in: {scaler_filename}")
        with open(scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
    else:
        print("Nessuno scaling necessario per questo modello.")
        X_train_processed = X_train

    # ============================================================
    # Definizione Modelli e Griglie di Parametri (aggiornate)
    # ============================================================
    models_config = {
    # ------------------------------------------------------------
    # LOGISTIC REGRESSION
    # ------------------------------------------------------------
    'logistic_regression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    ),

    # ------------------------------------------------------------
    # K-NEAREST NEIGHBORS
    # ------------------------------------------------------------
    'knn': (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    ),

    # ------------------------------------------------------------
    # DECISION TREE
    # ------------------------------------------------------------
    'decision_tree': (
        DecisionTreeClassifier(random_state=42),
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2]
        }
    ),

    # ------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------
    'random_forest': (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'n_estimators': [100],           
            'max_depth': [10, 20],            
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1,3],
            'max_features': ['sqrt']
        }
),

    # ------------------------------------------------------------
    # ADABOOST
    # ------------------------------------------------------------
    'adaboost': (
        AdaBoostClassifier(random_state=42),
        {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'estimator': [
                DecisionTreeClassifier(max_depth=1),
                DecisionTreeClassifier(max_depth=2)
            ]
        }
    ),

    # ------------------------------------------------------------
    # XGBOOST
    # ------------------------------------------------------------
    'xgboost': (
        XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
        {  'n_estimators': [100, 250],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3]
        }
    )
}

    # ============================================================
    # Controllo modello
    # ============================================================
    if model_name not in models_config:
        print(f"Errore: Modello '{model_name}' non riconosciuto.")
        print(f"Modelli supportati: {list(models_config.keys())}")
        return None, None, None

    estimator, param_grid = models_config[model_name]

    # ============================================================
    # Cross-validation e Grid Search
    # ============================================================
    print(f"Esecuzione Grid Search (5-Fold CV) per {model_name}...")
    kf_grid = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        cv=kf_grid,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train_processed, Y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print("\n--- Risultati Ottimizzazione ---")
    print(f"Migliori Iperparametri trovati: {best_params}")
    print(f"Accuracy media di CV: {best_score:.4f}")

    # ============================================================
    # Salvataggio modello migliore
    # ============================================================
    model_filename = os.path.join(roba_dir, f"{model_name}_best_model.pkl")
    print(f"Salvataggio modello in: {model_filename}")
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Tutti i file salvati in: {roba_dir}")

    return best_model, best_params, best_score
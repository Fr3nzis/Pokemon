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
    Esegue la GridSearchCV per un modello specifico.

    Args:
        model_name (str): Chiave del modello da usare (es. 'logistic_regression', 'knn').
        X_train (pd.DataFrame): DataFrame delle feature di addestramento.
        Y_train (pd.Series): Serie dei target di addestramento.

    Returns:
        tuple: (best_estimator, best_params, best_score) o (None, None, None) in caso di errore.
    """
    
    print(f"\n--- Inizio Ottimizzazione per: {model_name} ---")

    # --- 1. Scaling Condizionale ---
    # Scala i dati solo per i modelli che ne beneficiano (LogReg, KNN)
    if model_name in ['logistic_regression', 'knn']:
        print("Scaling dati di addestramento...")
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        
        # Salviamo lo scaler per uso futuro
        scaler_filename = f"models/{model_name}_scaler.pkl"
        print(f"Salvataggio scaler in: {scaler_filename}")
        with open(scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
            
    else:
        print("Nessuno scaling necessario per questo modello.")
        X_train_processed = X_train
        

    # --- 2. Definizione Modelli e Griglie di Parametri ---
    # Nota: queste griglie sono esempi. Per una ricerca reale, potresti volerle più ampie.
    models_config = {
        'logistic_regression': (
            LogisticRegression(random_state=42, max_iter=1000),
            {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'] # 'liblinear' va bene per L1 e L2
            }
        ),
        'knn': (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'euclidean']
            }
        ),
        'decision_tree': (
            DecisionTreeClassifier(random_state=42),
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'random_forest': (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': [100, 200], # Tenuto basso per velocità
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        ),
        'adaboost': (
            AdaBoostClassifier(random_state=42),
            {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1]
            }
        ),
        'xgboost': (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
        )
    }

    # Controllo se il modello richiesto è nella nostra configurazione
    if model_name not in models_config:
        print(f"Errore: Modello '{model_name}' non riconosciuto.")
        print(f"Modelli supportati: {list(models_config.keys())}")
        return None, None, None

    # Estrae l'estimatore e la griglia giusti
    estimator, param_grid = models_config[model_name]

    # --- 3. Configurazione Cross-Validation ---
    kf_grid = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- 4. Creazione ed Esecuzione GridSearchCV ---
    print(f"Esecuzione Grid Search (5-Fold Cross-Validation) per {model_name}...")
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        cv=kf_grid,
        n_jobs=-1,  # Usa tutti i core disponibili
        refit=True    # Refitta il modello migliore sull'intero training set
    )
    
    # Addestramento
    grid_search.fit(X_train_processed, Y_train)

    # --- 5. Estrazione e Salvataggio Risultati ---
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print("\n--- Risultati Ottimizzazione ---")
    print(f"Migliori Iperparametri trovati: {best_params}")
    print(f"Accuracy Media di Cross-Validation: {best_score:.4f}")

    # Salvataggio del modello migliore
    model_filename = f"roba/{model_name}_best_model.pkl"
    print(f"Salvataggio modello in: {model_filename}")
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
        
    return best_model, best_params, best_score



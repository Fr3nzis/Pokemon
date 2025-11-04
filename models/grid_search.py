import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
import pickle # Libreria necessaria per salvare il modello

# --- 1. CARICAMENTO DATI ---
# Assumiamo che i percorsi ai file siano corretti e i dati già puliti
try:
    X_train = pd.read_pickle("data/X_train.pkl")
    Y_train = pd.read_pickle("data/Y_train.pkl")
except FileNotFoundError:
    print("Errore: Impossibile trovare i file. Assicurarsi che 'data/X_train.pkl' e 'data/Y_train.pkl' esistano.")
    exit()

# --- 2. SCALING DEI DATI ---
print("Scaling dati di addestramento...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 3. CONFIGURAZIONE HYPERPARAMETER TUNING ---

# a) Estimatore di base
logreg_base = LogisticRegression(random_state=42, max_iter=1000)

# b) Definizione della Griglia degli Iperparametri
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    # Fissiamo il solver compatibile con L1 e L2 per evitare errori
    'solver': ['liblinear']
}

# c) Configurazione della Cross-Validation
kf_grid = KFold(n_splits=5, shuffle=True, random_state=42)

# d) Creazione dell'oggetto GridSearchCV
print("Esecuzione Grid Search (5-Fold Cross-Validation)...")
grid_search = GridSearchCV(
    estimator=logreg_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=kf_grid,
    n_jobs=-1,
    refit=True
)

# --- 4. ESECUZIONE DELLA RICERCA ---
grid_search.fit(X_train_scaled, Y_train)

# --- 5. SALVATAGGIO DEI RISULTATI ---
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_


print("\n--- Risultati Ottimizzazione ---")
print(f"Migliori Iperparametri trovati: {best_params}")
print(f"Accuracy Media di Cross-Validation: {best_score:.4f}")

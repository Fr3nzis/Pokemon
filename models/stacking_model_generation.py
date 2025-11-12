import pickle
import os
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# --- PERCORSI ---
MODEL_DIR = "models/generated_models"
DATA_DIR = "data"

# --- FUNZIONE PICKLE ---
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --- CARICAMENTO DATI ---
X_train = pd.read_pickle(f"{DATA_DIR}/X_train.pkl")
X_test = pd.read_pickle(f"{DATA_DIR}/X_test.pkl")
Y_train = pd.read_pickle(f"{DATA_DIR}/Y_train.pkl")
test_df = pd.read_pickle(f"{DATA_DIR}/df_test.pkl")

# --- SCALER (USA QUELLO DEL MODELLO PRINCIPALE O PIÙ STABILE, ESEMPIO: LOGISTIC REGRESSION) ---
scaler_path = os.path.join(MODEL_DIR, "logistic_regression_scaler.pkl")
print(f"Caricamento scaler da: {scaler_path}")
scaler = load_pickle(scaler_path)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- CARICAMENTO MODELLI OTTIMIZZATI ---
logreg_best = load_pickle(os.path.join(MODEL_DIR, "logistic_regression_best_model.pkl"))
xgb_best = load_pickle(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"))
knn_best = load_pickle(os.path.join(MODEL_DIR, "knn_best_model.pkl"))


# --- STACKING ---
base_models = [
    ('logreg', logreg_best),
    ('xgb', xgb_best),
    ('knn', knn_best)
]

meta_model = LogisticRegression(random_state=123,max_iter=1000)

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False,
    n_jobs=-1
)

# --- GRID SEARCH SULLO STACKING ---
param_grid = {
    'final_estimator__C': [0.01, 0.1, 1, 10,100],
    'final_estimator__solver': ['lbfgs', 'liblinear'],
    'passthrough': [False, True]
}

print("\n--- Inizio Grid Search per Stacking ---")
grid = GridSearchCV(
    estimator=stack_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train_scaled, Y_train)

print("\n--- Risultati Grid Search ---")
print("Migliori parametri trovati:", grid.best_params_)
print(f"Accuracy media CV migliore: {grid.best_score_:.4f}")

# --- SALVATAGGIO MIGLIOR MODELLO ---
best_stack_model = grid.best_estimator_
with open(os.path.join(MODEL_DIR, "stacking_best_model.pkl"), "wb") as f:
    pickle.dump(best_stack_model, f)
print("Modello di stacking ottimizzato salvato.")

# --- VALIDAZIONE ESTERNA CON K-FOLD ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_stack_model, X_train_scaled, Y_train, cv=kf, scoring='accuracy', n_jobs=-1)

print(f"\nAccuratezze sui 5 fold: {cv_scores}")
print(f"Media accuracy (KFold): {cv_scores.mean():.4f}")
print(f"Deviazione standard: {cv_scores.std():.4f}")

# --- TRAIN FINALE ---
best_stack_model.fit(X_train_scaled, Y_train)

# --- PREDIZIONE TEST ---
y_pred_test = best_stack_model.predict(X_test_scaled)

submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": y_pred_test
})
submission.to_csv("fds-pokemon-battles-prediction-2025/submission_ST.csv", index=False)
print("\nFile 'submission_ST.csv' creato correttamente.")



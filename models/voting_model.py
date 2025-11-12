import pickle
import os
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
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

# --- SCALER (usa quello del modello più stabile, es. logistic regression) ---
scaler_path = os.path.join(MODEL_DIR, "logistic_regression_scaler.pkl")
print(f"Caricamento scaler da: {scaler_path}")
scaler = load_pickle(scaler_path)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- CARICAMENTO MODELLI OTTIMIZZATI ---
logreg_best = load_pickle(os.path.join(MODEL_DIR, "logistic_regression_best_model.pkl"))
xgb_best = load_pickle(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"))
knn_best = load_pickle(os.path.join(MODEL_DIR, "knn_best_model.pkl"))

# --- ENSEMBLE: VOTING CLASSIFIER ---
voting_clf = VotingClassifier(
    estimators=[
        ('logreg', logreg_best),
        ('xgb', xgb_best),
        ('knn', knn_best)
    ],
    voting='soft',   # media delle probabilità
    n_jobs=-1
)

# --- VALIDAZIONE K-FOLD ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_clf, X_train_scaled, Y_train, cv=kf, scoring='accuracy', n_jobs=-1)

print("\n--- Risultati Cross-Validation ---")
print(f"Accuratezze sui 5 fold: {cv_scores}")
print(f"Media accuracy: {cv_scores.mean():.4f}")
print(f"Deviazione standard: {cv_scores.std():.4f}")

# --- TRAIN SU TUTTO IL TRAIN SET ---
voting_clf.fit(X_train_scaled, Y_train)

# --- SALVATAGGIO MODELLO ---
with open(os.path.join(MODEL_DIR, "voting_best_model.pkl"), "wb") as f:
    pickle.dump(voting_clf, f)
print("\nModello di voting salvato correttamente.")

# --- PREDIZIONE SU TEST ---
y_pred_test = voting_clf.predict(X_test_scaled)

submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": y_pred_test
})
submission.to_csv("fds-pokemon-battles-prediction-2025/submission_VOT.csv", index=False)
print("File 'submission_VOT.csv' creato correttamente.")
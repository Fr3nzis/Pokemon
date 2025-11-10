from grid_func import tune_model
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === PARAMETRI ===
SELECTED_MODEL = 'logistic_regression'   # 👈 Cambia qui: 'random_forest', 'adaboost', ecc.
SUBMISSION_PATH = f"fds-pokemon-battles-prediction-2025/submission_{SELECTED_MODEL}.csv"

# === DATI ===
print("Caricamento dati...")
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
Y_train = pd.read_pickle("data/Y_train.pkl")
test_df = pd.read_pickle("data/df_test.pkl")

# === SCALING ===
# Se vuoi essere sicuro che il test venga scalato come il training
print("Scaling dati...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === TUNING DEL MODELLO ===
print(f"\nOttimizzazione e training modello selezionato: {SELECTED_MODEL.upper()}")
best_model, best_params, best_score = tune_model(SELECTED_MODEL, X_train_scaled, Y_train)

if best_model is None:
    raise ValueError("⚠  tune_model non ha restituito un modello valido!")

# === TRAINING FINALE ===
print(f"\nAddestramento finale del modello {SELECTED_MODEL} su tutto il training...")
best_model.fit(X_train_scaled, Y_train)

# === PREDIZIONE E CREAZIONE SUBMISSION ===
print("Creazione file di submission...")
test_predictions = best_model.predict(X_test_scaled)

submission_df = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": test_predictions
})

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\n✅ File di submission salvato con successo in: {SUBMISSION_PATH}")
print(f"Accuracy media in CV (dal tuning): {best_score:.4f}")
print(f"Migliori parametri trovati: {best_params}")
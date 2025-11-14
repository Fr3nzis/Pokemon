import os

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from models.stacking_functions import load_data, load_scaler, load_pickle

MODEL_DIR = "models/generated_models"
MODEL_FILENAME = "logistic_regression_best_model.pkl"
SUBMISSION_PATH = "fds-pokemon-battles-prediction-2025/submission_LOG.csv"


def main():
    X_train, X_test, Y_train, test_df = load_data()
    X_train_scaled, X_test_scaled = load_scaler(X_train, X_test)

    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    print(f"Caricamento modello logistico da: {model_path}")
    logistic_model = load_pickle(model_path)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        logistic_model,
        X_train_scaled,
        Y_train,
        cv=kf,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("\n--- Risultati Cross-Validation (Logistic Regression) ---")
    print(f"Accuratezze sui 5 fold: {cv_scores}")
    print(f"Media accuracy: {cv_scores.mean():.4f}")
    print(f"Deviazione standard: {cv_scores.std():.4f}")

    logistic_model.fit(X_train_scaled, Y_train)
    y_pred = logistic_model.predict(X_test_scaled)

    submission = pd.DataFrame({
        "battle_id": test_df["battle_id"],
        "player_won": y_pred,
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"File '{SUBMISSION_PATH}' creato correttamente.")


if __name__ == "__main__":
    main()

import os
import pickle

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, cross_val_score

MODEL_DIR = "models/generated_models"
DATA_DIR = "data"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    X_train = pd.read_pickle(f"{DATA_DIR}/X_train.pkl")
    X_test = pd.read_pickle(f"{DATA_DIR}/X_test.pkl")
    Y_train = pd.read_pickle(f"{DATA_DIR}/Y_train.pkl")
    test_df = pd.read_pickle(f"{DATA_DIR}/df_test.pkl")

    scaler_path = os.path.join(MODEL_DIR, "logistic_regression_scaler.pkl")
    print(f"Caricamento scaler da: {scaler_path}")
    scaler = load_pickle(scaler_path)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg_best = load_pickle(os.path.join(MODEL_DIR, "logistic_regression_best_model.pkl"))
    xgb_best = load_pickle(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"))
    adaboost_best = load_pickle(os.path.join(MODEL_DIR, "adaboost_best_model.pkl"))

    voting_clf = VotingClassifier(
        estimators=[
            ("logreg", logreg_best),
            ("xgb", xgb_best),
            ("adaboost", adaboost_best),
        ],
        voting="soft",
        n_jobs=-1,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(voting_clf, X_train_scaled, Y_train, cv=kf, scoring="accuracy", n_jobs=-1)

    print("\n--- Risultati Cross-Validation ---")
    print(f"Accuratezze sui 5 fold: {cv_scores}")
    print(f"Media accuracy: {cv_scores.mean():.4f}")
    print(f"Deviazione standard: {cv_scores.std():.4f}")

    voting_clf.fit(X_train_scaled, Y_train)

    with open(os.path.join(MODEL_DIR, "voting_best_model.pkl"), "wb") as f:
        pickle.dump(voting_clf, f)
    print("\nModello di voting salvato correttamente.")

    y_pred_test = voting_clf.predict(X_test_scaled)

    submission = pd.DataFrame({
        "battle_id": test_df["battle_id"],
        "player_won": y_pred_test,
    })
    submission.to_csv("fds-pokemon-battles-prediction-2025/submission_VOT.csv", index=False)
    print("File 'submission_VOT.csv' creato correttamente.")


if __name__ == "__main__":
    main()

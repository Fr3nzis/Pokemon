import pickle
import os
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
MODEL_DIR = "models/generated_models"
DATA_DIR = "data"

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_data():
    X_train = pd.read_pickle(f"{DATA_DIR}/X_train.pkl")
    X_test = pd.read_pickle(f"{DATA_DIR}/X_test.pkl")
    Y_train = pd.read_pickle(f"{DATA_DIR}/Y_train.pkl")
    test_df = pd.read_pickle(f"{DATA_DIR}/df_test.pkl")
    return X_train, X_test, Y_train, test_df

def load_scaler(X_train, X_test):
    scaler_path = os.path.join(MODEL_DIR, "logistic_regression_scaler.pkl")
    print(f"Caricamento scaler da: {scaler_path}")
    scaler = load_pickle(scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
'''
def load_base_models():
    logreg_best = load_pickle(os.path.join(MODEL_DIR, "logistic_regression_best_model.pkl"))
    xgb_best = load_pickle(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"))
    knn_best = load_pickle(os.path.join(MODEL_DIR, "knn_best_model.pkl"))
    return logreg_best, xgb_best, knn_best
    '''

def load_base_models():
    logreg_best = load_pickle(os.path.join(MODEL_DIR, "logistic_regression_best_model.pkl"))
    xgb_best = load_pickle(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"))
    return logreg_best, xgb_best
'''
def build_stacking_model(logreg_best, xgb_best, knn_best):
    base_models = [
        ('logreg', logreg_best),
        ('xgb', xgb_best),
        ('knn', knn_best)
    ]
    meta_model = LogisticRegression(random_state=123, max_iter=1000)
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        passthrough=False,
        n_jobs=-1
    )
    return stack_model
'''
def build_stacking_model(logreg_best, xgb_best):
    base_models = [
        ('logreg', logreg_best),
        ('xgb', xgb_best)
    ]
    meta_model = LogisticRegression(random_state=123, max_iter=1000)
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        passthrough=False,
        n_jobs=-1
    )
    return stack_model

def tune_stacking_model(stack_model, X_train, Y_train):
    param_grid = {
        'final_estimator__C': [0.01, 0.1, 1, 10, 100],
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
    grid.fit(X_train, Y_train)

    print("\n--- Risultati Grid Search ---")
    print("Migliori parametri trovati:", grid.best_params_)
    print(f"Accuracy media CV migliore: {grid.best_score_:.4f}")
    return grid.best_estimator_

def validate_model(model, X_train, Y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy', n_jobs=-1)
    print(f"\nAccuratezze sui 5 fold: {cv_scores}")
    print(f"Media accuracy (KFold): {cv_scores.mean():.4f}")
    print(f"Deviazione standard: {cv_scores.std():.4f}")

def train_and_predict(model, X_train, Y_train, X_test, test_df):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    submission = pd.DataFrame({
        "battle_id": test_df["battle_id"],
        "player_won": y_pred
    })
    submission.to_csv("fds-pokemon-battles-prediction-2025/submission_ST.csv", index=False)
    print("\nFile 'submission_ST.csv' creato correttamente.")

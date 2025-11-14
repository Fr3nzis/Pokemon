import pandas as pd

from models.tuning_function import tune_model

DATA_DIR = "data"


def main():
    X_train = pd.read_pickle(f"{DATA_DIR}/X_train.pkl")
    Y_train = pd.read_pickle(f"{DATA_DIR}/Y_train.pkl")

    for model_name in ["xgboost", "logistic_regression", "adaboost"]:
        tune_model(model_name, X_train, Y_train)


if __name__ == "__main__":
    main()


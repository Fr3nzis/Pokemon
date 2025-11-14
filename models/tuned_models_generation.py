from tuning_function import tune_model
import pandas as pd

#DATI:
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
Y_train = pd.read_pickle("data/Y_train.pkl")
test_df = pd.read_pickle('data/df_test.pkl')


for m in ['xgboost','logistic_regression','adaboost']:
    tune_model(m,X_train,Y_train)



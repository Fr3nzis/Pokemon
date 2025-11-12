from tuning_function import tune_model
from stacking_functions import load_base_models, load_scaler, build_stacking_model, validate_model, tune_stacking_model, train_and_predict
import pandas as pd
import os
import pickle 

#DATI:
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
Y_train = pd.read_pickle("data/Y_train.pkl")
test_df = pd.read_pickle('data/df_test.pkl')


for m in ['decision_tree', 'adaboost','xgboost','random_forest','knn', 'logistic_regression']:
    tune_model(m,X_train,Y_train)


MODEL_DIR = "models/generated_models"



X_train_scaled, X_test_scaled = load_scaler(X_train, X_test)

#Carica modelli base
logreg, xgb, knn = load_base_models()

#Costruisci stacking
stack_model = build_stacking_model(logreg, xgb, knn)

#(GridSearch)
best_model = tune_stacking_model(stack_model, X_train_scaled, Y_train)

#Salva modello migliore
with open(os.path.join(MODEL_DIR, "stacking_best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("Modello di stacking ottimizzato salvato.")

#Validazione
validate_model(best_model, X_train_scaled, Y_train)

#Training e predizione finale
train_and_predict(best_model, X_train_scaled, Y_train, X_test_scaled, test_df)



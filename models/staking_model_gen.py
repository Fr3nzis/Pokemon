import os
import pickle
from stacking_functions import (
    load_data,
    load_scaler,
    load_base_models,
    build_stacking_model,
    tune_stacking_model,
    validate_model,
    train_and_predict,
)

MODEL_DIR = "models/generated_models"


# 1. Carica dati
X_train, X_test, Y_train, test_df = load_data()

# 2. Applica scaler
X_train_scaled, X_test_scaled = load_scaler(X_train, X_test)

# 3. Carica modelli base
logreg, xgb, knn = load_base_models()

# 4. Costruisci stacking
stack_model = build_stacking_model(logreg, xgb, knn)

# 5. Ottimizza modello (GridSearch)
best_model = tune_stacking_model(stack_model, X_train_scaled, Y_train)

# 6. Salva modello migliore
with open(os.path.join(MODEL_DIR, "stacking_best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("Modello di stacking ottimizzato salvato.")

# 7. Validazione
validate_model(best_model, X_train_scaled, Y_train)

# 8. Training e predizione finale
train_and_predict(best_model, X_train_scaled, Y_train, X_test_scaled, test_df)


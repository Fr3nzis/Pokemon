from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd
from sklearn.metrics import accuracy_score

#DATI:
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
Y_train = pd.read_pickle("data/Y_train.pkl")
test_df = pd.read_pickle('data/df_test.pkl')



# --- 2. SCALING DEI DATI (RIMOSSO) ---
print("Scaling data... (Saltato, non necessario per Decision Tree)")
# X_train_scaled = scaler.fit_transform(X_train) # <- RIMOSSO
# X_test_scaled = scaler.transform(X_test) # <- RIMOSSO

# --- 3. Implementazione K-FOLD CROSS-VALIDATION (K=5) ---

# MODIFICATO: Inizializza il Decision Tree con Pruning (per evitare overfitting)
model = DecisionTreeClassifier(
    random_state=42, 
    max_depth=7,           # Limita la profondità 
    min_samples_leaf=20      # Richiede almeno 20 campioni per foglia
)

# Inizializza la K-Fold (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\nEsecuzione K-Fold Cross-Validation con K=5 sul modello Decision Tree...") # <- MODIFICATO

# Calcola il punteggio di Accuracy per ogni fold (usa X_train NON scalato)
cv_scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy', n_jobs=-1)
acc = cv_scores.mean()

# --- 4. Risultati ---
print("Risultati di Cross-Validation:")
print(f"Scores per ogni fold (K=5): {cv_scores}")
print("-----------------------------------------------------")
print(f"Accuracy Media (Voto Affidabile): {cv_scores.mean():.4f}")
print(f"Deviazione Standard (Varianza):    {cv_scores.std():.4f}")

# --- 5. ADDESTRAMENTO FINALE e PREDIZIONE ---
print("\nAddestramento finale del modello sul set di training completo...")
model.fit(X_train, Y_train) # Usa X_train NON scalato
train_accuracy = model.score(X_train, Y_train)

print("\n--- Risultati Totali dopo Addestramento Finale ---")
print(f"Accuracy di Training Totale (Ottimistica): {train_accuracy:.4f}")
print(f"Accuracy di Cross-Validation (Realistica): {acc:.4f}")

test_predictions = model.predict(X_test) # Usa X_test NON scalato

submission_df = pd.DataFrame({
    'battle_id': test_df['battle_id'],
    'player_won': test_predictions
})

# Save the DataFrame to a .csv file
submission_df.to_csv('submission.csv', index=False) 

print("\n'submission.csv' file created successfully!")
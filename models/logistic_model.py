from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

#DATI:
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
Y_train = pd.read_pickle("data/Y_train.pkl")
test_df = pd.read_pickle('data/df_test.pkl')

#SCALING DEI DATI
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementazione K-FOLD CROSS-VALIDATION (K=5)
model = LogisticRegression(random_state=42, max_iter=1000)

# Inizializza la K-Fold (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\nEsecuzione K-Fold Cross-Validation con K=5 sul modello Regressione Logistica...")

# Calcola il punteggio di Accuracy per ogni fold
cv_scores = cross_val_score(model, X_train_scaled, Y_train, cv=kf, scoring='accuracy', n_jobs=-1)
acc = cv_scores.mean()
#4 Risultati
''' 
print("Risultati di Cross-Validation:")
print(f"Scores per ogni fold (K=5): {cv_scores}")
print("-----------------------------------------------------")
print(f"Accuracy Media (Voto Affidabile): {cv_scores.mean():.4f}")
print(f"Deviazione Standard (Varianza):    {cv_scores.std():.4f}")
'''


#Addestramento modello sul set di training SCALATO COMPLETO (Per l'uso finale e per la metrica totale)
print("\nAddestramento finale del modello sul set di training completo e scalato...")
model.fit(X_train_scaled, Y_train)
train_accuracy = model.score(X_train_scaled, Y_train)

print("\n--- Risultati Totali dopo Addestramento Finale ---")
print(f"Accuracy di Training Totale (Ottimistica): {train_accuracy:.4f}")
print(f"Accuracy di Cross-Validation (Realistica): {acc:.4f}")

test_predictions = model.predict(X_test_scaled) 

submission_df = pd.DataFrame({
    'battle_id': test_df['battle_id'],
    'player_won': test_predictions
})

# Save the DataFrame to a .csv file
submission_df.to_csv('fds-pokemon-battles-prediction-2025/submission.csv', index=False)

print("\n'submission.csv' file created successfully!")
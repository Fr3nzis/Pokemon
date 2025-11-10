from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# --- DATI ---
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
Y_train = pd.read_pickle("data/Y_train.pkl")
test_df = pd.read_pickle("data/df_test.pkl")

# --- SCALING ---
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODELLI BASE ---
base_models = [
    ('logreg', LogisticRegression(max_iter=1000)),
    ('xgb', XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss'
    )),
    ('knn', KNeighborsClassifier(n_neighbors=9))
]

meta_model = LogisticRegression(max_iter=1000)
#meta_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
#meta_model = KNeighborsClassifier(n_neighbors=9)
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,              # CV interna per il meta-modello
    passthrough=False
)

# --- K-FOLD ESTERNO PER VALUTAZIONE ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(stack_model, X_train_scaled, Y_train, cv=kf, scoring='accuracy', n_jobs=-1)

print(f"Accuratezze sui 5 fold: {cv_scores}")
print(f"Media accuracy (KFold): {cv_scores.mean():.4f}")
print(f"Deviazione standard: {cv_scores.std():.4f}")

# --- ADDDESTRAMENTO FINALE SU TUTTO IL TRAIN ---
stack_model.fit(X_train_scaled, Y_train)

# --- PREDIZIONE SU TEST ---
y_pred_test = stack_model.predict(X_test_scaled)

submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": y_pred_test
})
submission.to_csv("fds-pokemon-battles-prediction-2025/submission_ST.csv", index=False)
print("File 'submission_ST.csv' creato correttamente.")
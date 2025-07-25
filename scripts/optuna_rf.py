import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# ===========================
# 1. Load dataset
# ===========================
df = pd.read_csv("telco_features_scaled.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ===========================
# 2. Train/Val/Test Split
# ===========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)
# ≈ 70% train / 15% val / 15% test

# ===========================
# 3. Apply SMOTE to training set
# ===========================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ===========================
# 4. Optuna Objective Function
# ===========================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train_sm, y_train_sm)

    y_val_pred = model.predict(X_val)
    score = f1_score(y_val, y_val_pred)
    return score

# ===========================
# 5. Run Optuna Study (100 trials)
# ===========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ===========================
# 6. Save Best Params and Model
# ===========================
print("Best F1 score:", study.best_value)
print("Best hyperparameters:", study.best_params)

best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train_sm, y_train_sm)

joblib.dump(best_model, "rf_best_model_full.joblib")

# Save best params to CSV
best_params_df = pd.DataFrame([study.best_params])
best_params_df.to_csv("best_rf_params_full.csv", index=False)

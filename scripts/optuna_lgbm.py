import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler

# ✅ Load 1% of the data for testing
df = pd.read_csv("telco_features_unscaled.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ✅ Split into train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ✅ Apply SMOTE to training set only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ✅ Objective function
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 512),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 100.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'n_estimators': 1000,
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    preds = model.predict(X_val)
    return f1_score(y_val, preds)

# ✅ Run Optuna study
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=300)

# ✅ Print and save best results
print("\n🎯 Best trial:")
print(f"  F1-score: {study.best_trial.value:.4f}")
print("  Best hyperparameters:")
for key, val in study.best_trial.params.items():
    print(f"    {key}: {val}")

# Save best hyperparameters to CSV
pd.DataFrame([study.best_trial.params]).to_csv("best_lgbm_params.csv", index=False)

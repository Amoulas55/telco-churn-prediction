import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler

# Load full dataset
df = pd.read_csv("telco_features_unscaled.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Apply SMOTE on training set only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
dval = xgb.DMatrix(X_val, label=y_val)

# Define Optuna objective function
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 100.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'tree_method': 'hist',
        'n_estimators': 300,
        'verbosity': 0,
        'seed': 42
    }

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=300,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )

    preds = model.predict(dval)
    pred_labels = (preds > 0.5).astype(int)
    return f1_score(y_val, pred_labels)

# Run the Optuna study
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150)

# Show best result
print("\n🎯 Best Trial:")
print(f"  F1-score: {study.best_trial.value:.4f}")
print("  Best hyperparameters:")
for key, val in study.best_trial.params.items():
    print(f"    {key}: {val}")

# Save best params to CSV
pd.DataFrame([study.best_trial.params]).to_csv("best_xgboost_params.csv", index=False)

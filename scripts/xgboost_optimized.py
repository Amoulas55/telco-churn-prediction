import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib
import json

# ===========================
# 1. Load data
# ===========================
df = pd.read_csv("telco_features_unscaled.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===========================
# 2. Train/val/test split
# ===========================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
# Train: 70%, Val: 15%, Test: 15%

# ===========================
# 3. SMOTE on training data
# ===========================
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# ===========================
# 4. Load best Optuna params
# ===========================
best_params = pd.read_csv("best_xgboost_params.csv").iloc[0].to_dict()

# Fix types
int_keys = ["max_depth", "min_child_weight"]
float_keys = ["learning_rate", "subsample", "colsample_bytree", "gamma", "lambda", "alpha"]
if "grow_policy" in best_params:
    best_params["grow_policy"] = str(best_params["grow_policy"])

for k in int_keys:
    best_params[k] = int(best_params[k])
for k in float_keys:
    best_params[k] = float(best_params[k])

# Required params for training
best_params["objective"] = "binary:logistic"
best_params["eval_metric"] = "logloss"
best_params["seed"] = 42

# ===========================
# 5. Convert to DMatrix
# ===========================
dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# ===========================
# 6. Train model
# ===========================
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=True
)

# ===========================
# 7. Predict and evaluate
# ===========================
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1 Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc:.4f}\n")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===========================
# 8. Save results for META
# ===========================
results_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_pred_proba
})
results_df.to_csv("xgboost_predictions_meta.csv", index=False)

with open("xgboost_metrics_meta.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }, f, indent=4)

joblib.dump(model, "xgboost_best_model_meta.joblib")

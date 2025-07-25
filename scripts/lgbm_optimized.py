import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import json

# === Load Data ===
df = pd.read_csv("telco_features_unscaled.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# === Split Data ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# === Balance Training Set ===
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# === Define Best LightGBM Model ===
model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    metric="binary_logloss",
    n_estimators=420,
    learning_rate=0.0173545,
    max_depth=22,
    num_leaves=59,
    min_child_samples=40,
    subsample=0.77059,
    colsample_bytree=0.72049,
    reg_alpha=0.34801,
    reg_lambda=0.52983,
    random_state=42
)

# === Train with Early Stopping ===
model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)

# === Predict and Evaluate ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save Predictions ===
results_df = pd.DataFrame({
    "index": X_test.index,
    "true_label": y_test.values,
    "pred_label": y_pred,
    "pred_proba": y_proba
})
results_df.to_csv("lgbm_test_results_meta.csv", index=False)
print("\n📁 Test results saved to 'lgbm_test_results_meta.csv'")

# === Save Metrics for Meta Comparison ===
metrics = {
    "accuracy": round(acc, 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4)
}
with open("lgbm_metrics_meta.json", "w") as f:
    json.dump(metrics, f)
print("📄 Metrics saved to 'lgbm_metrics_meta.json'")

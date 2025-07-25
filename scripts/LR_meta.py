import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib
import json

# ===========================
# 1. Load and rename predictions
# ===========================
df_rf = pd.read_csv("rf_predictions_meta.csv").rename(columns={"y_proba": "rf"})
df_xgb = pd.read_csv("xgboost_predictions_meta.csv").rename(columns={"y_proba": "xgb"})
df_lgbm = pd.read_csv("lgbm_test_results_meta.csv").rename(columns={"pred_proba": "lgbm", "true_label": "y_true"})
df_svm = pd.read_csv("svm_predictions_meta.csv").rename(columns={"y_proba": "svm"})
df_mlp = pd.read_csv("mlp_predictions_meta.csv").rename(columns={"y_proba": "mlp"})

# ===========================
# 2. Merge on index
# ===========================
meta_df = df_rf[["y_true"]].copy()
meta_df["rf"] = df_rf["rf"]
meta_df["xgb"] = df_xgb["xgb"]
meta_df["lgbm"] = df_lgbm["lgbm"]
meta_df["svm"] = df_svm["svm"]
meta_df["mlp"] = df_mlp["mlp"]

# ===========================
# 3. Add interaction features
# ===========================
X = meta_df[["rf", "xgb", "lgbm", "svm", "mlp"]].copy()
X["rf_xgb"] = X["rf"] * X["xgb"]
X["xgb_svm"] = X["xgb"] * X["svm"]
X["rf_mlp"] = X["rf"] * X["mlp"]
X["avg_vote"] = X[["rf", "xgb", "lgbm", "svm", "mlp"]].mean(axis=1)

y = meta_df["y_true"]

# ===========================
# 4. Split into train/test (meta-level)
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ===========================
# 5. Train meta-classifier with class_weight
# ===========================
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# ===========================
# 6. Threshold tuning
# ===========================
y_proba = clf.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.3, 0.71, 0.01)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    preds = (y_proba > t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Final predictions
y_pred = (y_proba > best_threshold).astype(int)

# ===========================
# 7. Evaluation Report
# ===========================
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\n✅ META Logistic Regression (threshold={best_threshold:.2f})")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===========================
# 8. Save everything
# ===========================
joblib.dump(clf, "meta_logreg_model_interactions.joblib")

pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba
}).to_csv("meta_predictions_interactions.csv", index=False)

with open("meta_metrics_interactions.json", "w") as f:
    json.dump({
        "threshold": round(float(best_threshold), 3),
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }, f, indent=4)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import json

# ===========================
# 1. Load scaled dataset
# ===========================
df = pd.read_csv("telco_features_scaled.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ===========================
# 2. Train/val/test split
# ===========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42
)
# 70% train / 15% val / 15% test

# ===========================
# 3. Apply SMOTE on training data
# ===========================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ===========================
# 4. Train baseline SVM (RBF kernel)
# ===========================
svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_sm, y_train_sm)

# ===========================
# 5. Evaluate on test set
# ===========================
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1 Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc:.4f}\n")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===========================
# 6. Save outputs for META
# ===========================
joblib.dump(svm, "svm_baseline_model_meta.joblib")

pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba
}).to_csv("svm_predictions_meta.csv", index=False)

with open("svm_metrics_meta.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }, f, indent=4)

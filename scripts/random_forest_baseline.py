import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib

# ===========================
# 1. Load the dataset from CSV
# ===========================
df = pd.read_csv("telco_features_scaled.csv")

# ===========================
# 2. Separate features and target
# ===========================
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ===========================
# 3. Use only 1% sample
# ===========================
_, X_sample, _, y_sample = train_test_split(
    X, y, test_size=0.01, stratify=y, random_state=42
)

# ===========================
# 4. Split into train/val/test
# ===========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_sample, y_sample, test_size=0.15, random_state=42, stratify=y_sample
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)
# ≈ 70% train / 15% val / 15% test

# ===========================
# 5. Apply SMOTE to training set only
# ===========================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ===========================
# 6. Train Random Forest (baseline)
# ===========================
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)

# ===========================
# 7. Evaluate on validation set
# ===========================
y_val_pred = rf.predict(X_val)
y_val_proba = rf.predict_proba(X_val)[:, 1]

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation F1 Score:", f1_score(y_val, y_val_pred))
print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_proba))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# ===========================
# 8. Save Model
# ===========================
joblib.dump(rf, "rf_baseline_model.joblib")

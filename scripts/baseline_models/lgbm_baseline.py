import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load imbalanced dataset
df = pd.read_csv("telco_features_unscaled.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into train, validation, test (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Apply SMOTE to training set only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Initialize LightGBM model
model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    metric="binary_logloss",
    n_estimators=1000,
    random_state=42
)

# Fit with early stopping using callbacks
model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

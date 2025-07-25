import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df[df["TotalCharges"].notna()].reset_index(drop=True)
df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
y = df["Churn"]
df.drop("Churn", axis=1, inplace=True)

binary_map = {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
df = df.replace(binary_map)
df = df.infer_objects(copy=False)

cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df_encoded, y)
importances = pd.Series(rf.feature_importances_, index=df_encoded.columns)
low_importance_features = importances[importances < 0.005].index.tolist()
df_selected = df_encoded.drop(columns=low_importance_features)

df_selected["Churn"] = y
df_selected.to_csv("telco_features_unscaled.csv", index=False)
print("✅ Saved unscaled dataset: telco_features_unscaled.csv")

scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
df_scaled = df_selected.copy()
df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
df_scaled.to_csv("telco_features_scaled.csv", index=False)
print("✅ Saved scaled dataset: telco_features_scaled.csv")

print(f"\n📊 Final feature count: {df_scaled.shape[1] - 1} features + 1 target (Churn)")

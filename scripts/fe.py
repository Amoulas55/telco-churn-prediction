import pandas as pd

# Step 1: Load data
df = pd.read_csv("telco.csv")

# Preview data types
print(df.dtypes)

# Fix TotalCharges: it's supposed to be numeric but is object
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check if there are any NaNs introduced
print("\nMissing values in TotalCharges:", df["TotalCharges"].isna().sum())

# Drop rows where TotalCharges couldn't be converted
df = df[df["TotalCharges"].notna()]

# Drop customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Reset index after row drops
df.reset_index(drop=True, inplace=True)

print("\n✅ Step 1 complete. Dataset shape:", df.shape)

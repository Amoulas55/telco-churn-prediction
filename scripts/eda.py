import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = "telco.csv"  # Change path if needed
df = pd.read_csv(file_path)

# Create output folder
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

### --- PRINT + SAVE BASIC INFO ---
print("\n📄 Dataset Shape:", df.shape)
print("\n📊 Missing Values:\n", df.isnull().sum())
print("\n🔍 Column Types:\n", df.dtypes)

with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
    f.write("Dataset Shape:\n")
    f.write(str(df.shape) + "\n\n")
    f.write("Column Info:\n")
    df.info(buf=f)
    f.write("\n\nMissing Values:\n")
    f.write(str(df.isnull().sum()) + "\n")

### --- STATISTICAL SUMMARY ---
summary = df.describe(include="all")
print("\n📈 Summary Stats:\n", summary)
summary.to_csv(os.path.join(output_dir, "statistical_summary.csv"))

### --- HISTOGRAMS for NUMERIC FEATURES (with error handling) ---
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()
    except Exception as e:
        print(f"❌ Skipping {col} due to: {e}")

### --- BAR PLOTS for CATEGORICAL FEATURES ---
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Frequency of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{col}_bar.png"))
    plt.close()

### --- CORRELATION HEATMAP + BOXPLOTS if 'Churn' exists ---
if "Churn" in df.columns:
    df["Churn_encoded"] = df["Churn"].map({"Yes": 1, "No": 0})
    print("\n✅ 'Churn' column encoded for correlation.")

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    print("\n📊 Correlation Matrix:\n", corr)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    # Boxplots
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Churn", y=col, data=df)
        plt.title(f"{col} vs Churn")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_vs_Churn_boxplot.png"))
        plt.close()

### --- DATA PREVIEW ---
df.head(50).to_csv(os.path.join(output_dir, "data_preview.csv"), index=False)
print("\n✅ First 50 rows saved to 'data_preview.csv'")
print(f"\n🎯 All EDA outputs saved to: {output_dir}")

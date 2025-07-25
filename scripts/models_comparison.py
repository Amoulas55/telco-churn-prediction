import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score

print("\n--- LOADING METRICS ---")

# File and label mapping
model_files = {
    "Random Forest": "rf_predictions_meta.csv",
    "XGBoost": "xgboost_predictions_meta.csv",
    "SVM": "svm_predictions_meta.csv",
    "MLP": "mlp_predictions_meta.csv",
    "LogReg Meta": "meta_predictions_interactions.csv"
}

results = []

for model_name, file_name in model_files.items():
    try:
        df = pd.read_csv(file_name)
        print(f"{model_name}: {list(df.columns)}")
        
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        y_proba = df["y_proba"]  # can be used for ROC-AUC etc. if needed

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)

        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "F1-score": f1,
            "Precision": prec
        })

    except FileNotFoundError:
        print(f"⚠️ Skipping {model_name}: file '{file_name}' not found")

# Create DataFrame
results_df = pd.DataFrame(results)
results_df.set_index("Model", inplace=True)

print("\n--- FINAL COMPARISON TABLE (Accuracy, F1-score, Precision) ---")
print(results_df.sort_values(by="F1-score", ascending=False).round(4))

# Optionally save to CSV
results_df.to_csv("model_comparison_acc_f1_precision.csv")

# Optionally plot
import matplotlib.pyplot as plt

results_df.sort_values(by="F1-score", ascending=False).plot(kind="bar", figsize=(10, 6))
plt.title("Model Comparison (Accuracy, F1, Precision)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison_acc_f1_precision.png")
plt.show()

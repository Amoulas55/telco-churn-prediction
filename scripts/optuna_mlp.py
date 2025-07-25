import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader
import optuna
import joblib
import json

# ===========================
# 1. Load dataset
# ===========================
df = pd.read_csv("telco_features_scaled.csv")
X = df.drop(columns=["Churn"]).values.astype(np.float32)
y = df["Churn"].values.astype(np.int64)

# Train/val/test split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42)

# SMOTE
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. Define MLP
# ===========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ===========================
# 3. Optuna objective
# ===========================
def objective(trial):
    hidden1 = trial.suggest_int("hidden1", 32, 128)
    hidden2 = trial.suggest_int("hidden2", 16, 64)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = 20

    model = MLP(X.shape[1], hidden1, hidden2, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_sm), torch.tensor(y_train_sm)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                            batch_size=batch_size)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    val_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            val_probs += model(xb).cpu().numpy().flatten().tolist()

    val_preds = (np.array(val_probs) > 0.5).astype(int)
    return f1_score(y_val, val_preds)

# ===========================
# 4. Run Optuna (150 trials)
# ===========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

# ===========================
# 5. Final training with best params
# ===========================
print("Best F1 Score:", study.best_value)
print("Best Params:", study.best_params)

params = study.best_params
model = MLP(X.shape[1], params["hidden1"], params["hidden2"], params["dropout"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
criterion = nn.BCELoss()
batch_size = params["batch_size"]
epochs = 20

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_sm), torch.tensor(y_train_sm)),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                         batch_size=batch_size)

for _ in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.float().to(device).view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

# ===========================
# 6. Evaluate on test set
# ===========================
model.eval()
test_probs, test_preds = [], []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        probs = model(xb).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        test_probs.extend(probs)
        test_preds.extend(preds)

y_pred = np.array(test_preds)
y_proba = np.array(test_probs)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1 Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc:.4f}")

# ===========================
# 7. Save everything for META
# ===========================
torch.save(model.state_dict(), "mlp_best_model_meta.pt")

pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_proba": y_proba
}).to_csv("mlp_predictions_meta.csv", index=False)

with open("mlp_metrics_meta.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }, f, indent=4)

pd.DataFrame([params]).to_csv("mlp_best_params.csv", index=False)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json

# ===========================
# 1. Load and split data
# ===========================
df = pd.read_csv("telco_features_scaled.csv")
X = df.drop(columns=["Churn"]).values.astype(np.float32)
y = df["Churn"].values.astype(np.int64)

# Split: 70% train / 15% val / 15% test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42)

# ===========================
# 2. Apply SMOTE on training
# ===========================
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ===========================
# 3. Device setup
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 4. Dataset and Dataloaders
# ===========================
def to_loader(X, y, batch_size=64, shuffle=False):
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train_sm, y_train_sm, shuffle=True)
val_loader = to_loader(X_val, y_val)
test_loader = to_loader(X_test, y_test)

# ===========================
# 5. Define MLP model
# ===========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===========================
# 6. Training loop
# ===========================
epochs = 20
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.float().to(device).view(-1, 1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

# ===========================
# 7. Evaluation on test set
# ===========================
model.eval()
all_preds, all_probs = [], []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        probs = model(xb).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_probs.extend(probs)

y_pred = np.array(all_preds)
y_proba = np.array(all_probs)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1 Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===========================
# 8. Save for META
# ===========================
torch.save(model.state_dict(), "mlp_baseline_model_meta.pt")

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

import os

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "scripts/baseline_models",
    "scripts/optimized_models",
    "scripts/meta_model",
    "scripts/eda",
    "scripts/preprocessing",
    "scripts/utils",
    "outputs/figures",
    "outputs/models",
    "outputs/results",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Folder structure created.")

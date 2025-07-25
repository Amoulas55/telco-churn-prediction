# Telco Churn Prediction

This project delivers a full machine learning pipeline to accurately predict customer churn for a telecommunications company. It combines strong business focus, rich exploratory analysis, and cutting-edge ML models—including model stacking and hyperparameter optimization using Optuna.

## 📊 Business Context

* **Objective:** Identify customers likely to churn and enable proactive retention strategies.
* **Dataset:** Public Telco Customer Churn dataset.
* **Tools & Libraries:** Python · pandas · scikit-learn · XGBoost · LightGBM · Optuna · PyTorch · Matplotlib · Seaborn

## 📁 Project Structure

```
├── scripts/              # Preprocessing, feature engineering, modeling, and tuning scripts
├── visualizations/       # EDA charts, ROC curves, model comparison plots
├── requirements.txt      # Python dependencies
├── README.md             # Project overview and instructions
```

## 🔁 End-to-End Workflow

### 1. Data Preprocessing

* Fix numeric column errors (e.g., TotalCharges)
* Handle missing values
* Encode categorical variables
* Standardize numerical features
* Feature selection using Random Forest importance

Scripts: `preprocessing.py`, `fe.py`

### 2. Exploratory Data Analysis (EDA)

* Distribution of churn vs. demographics
* Payment behavior insights
* Correlation heatmaps and boxplots

Script: `eda.py`

### 3. Model Training & Optimization

* **Models implemented:**

  * Logistic Regression (meta-model), Random Forest, SVM, XGBoost, LightGBM, Multi-layer Perceptron (MLP)
* Each model includes both a baseline and an Optuna-tuned version

Scripts:

* `random_forest_baseline.py`, `optuna_rf.py`, `random_forest_optimized.py`
* `svm_baseline.py`, `optuna_svm.py`
* `xgboost_baseline.py`, `optuna_xgboost.py`, `xgboost_optimized.py`
* `lgbm_baseline.py`, `optuna_lgbm.py`, `lgbm_optimized.py`
* `mlp_baseline.py`, `optuna_mlp.py`

### 4. Meta-Model Stacking

* Combines predictions from all base models
* Meta-model is a Logistic Regression trained on predicted probabilities and interaction features

Script: `LR_meta.py`

### 5. Model Evaluation & Comparison

* Metrics:

  * **Accuracy** · F1-score · ROC-AUC · Precision · Recall
* Visualization:

  * ROC curves, bar plots, confusion matrices

Scripts: `models_comparison.py`, saved plots in `visualizations/`

## 🌟 Key Highlights

* 🔄 **Stacking** meta-model improves predictive performance
* ⚙️ Full model optimization with **Optuna**
* 🧠 Feature interactions used in meta-model for added predictive power
* 🧱 Clean and modular structure ready for deployment or CI/CD extension

## 👤 Author

**Angelos Moulas**
Master’s in Data Science & Society · Tilburg University

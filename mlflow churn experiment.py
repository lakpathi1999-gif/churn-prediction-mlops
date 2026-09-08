"""
Customer Churn Prediction — MLflow Experiment Tracking
--------------------------------------------------------
Simulates a realistic telecom-style churn dataset and trains multiple
model variants, logging params/metrics/artifacts to MLflow for comparison.

This mirrors a real business ML workflow:
  - engineered, business-relevant features
  - multiple model configs compared side by side
  - confusion matrix + feature importance logged as artifacts
  - best model identifiable directly from the MLflow UI
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# -----------------------------------------------------------------
# 1. Generate a realistic synthetic churn dataset
# -----------------------------------------------------------------
np.random.seed(42)
n_customers = 2000

tenure_months = np.random.randint(1, 72, n_customers)
monthly_charges = np.round(np.random.uniform(20, 120, n_customers), 2)
total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.9, 1.0, n_customers), 2)
num_support_calls = np.random.poisson(1.5, n_customers)
contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], n_customers, p=[0.55, 0.25, 0.2])
payment_method = np.random.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n_customers)
has_tech_support = np.random.choice([0, 1], n_customers, p=[0.6, 0.4])

# Churn probability driven by realistic business logic
churn_score = (
    -0.04 * tenure_months
    + 0.02 * monthly_charges
    + 0.35 * num_support_calls
    - 0.8 * has_tech_support
    + np.where(contract_type == "Month-to-month", 1.2, 0)
    + np.where(contract_type == "One year", 0.3, 0)
    + np.random.normal(0, 1, n_customers)
)
churn_prob = 1 / (1 + np.exp(-churn_score))
churn = (churn_prob > np.random.uniform(0.4, 0.6, n_customers)).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "num_support_calls": num_support_calls,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "has_tech_support": has_tech_support,
    "churn": churn,
})

# One-hot encode categoricals for the model
df_encoded = pd.get_dummies(df, columns=["contract_type", "payment_method"], drop_first=True)

X = df_encoded.drop(columns=["churn"])
y = df_encoded["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dataset: {len(df)} customers, churn rate = {y.mean():.1%}")

# -----------------------------------------------------------------
# 2. Track multiple model configs with MLflow
# -----------------------------------------------------------------
mlflow.set_experiment("customer-churn-prediction")

hyperparam_sets = [
    {"n_estimators": 100, "max_depth": 4, "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"},
    {"n_estimators": 300, "max_depth": None, "class_weight": "balanced"},
]

for params in hyperparam_sets:
    run_name = f"rf_n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        # Log params + dataset info
        mlflow.log_params(params)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Metrics that actually matter for churn (imbalanced classes -> recall/AUC matter more than accuracy)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_metrics(metrics)

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, colorbar=False)
        plt.title(run_name)
        plt.tight_layout()
        fig_path = f"confusion_matrix_{run_name}.png"
        plt.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path)

        # Feature importance artifact
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        importances.head(8).plot(kind="barh", ax=ax2)
        ax2.invert_yaxis()
        plt.title(f"Top features — {run_name}")
        plt.tight_layout()
        fi_path = f"feature_importance_{run_name}.png"
        plt.savefig(fi_path)
        plt.close(fig2)
        mlflow.log_artifact(fi_path)

        # Log the model itself
        mlflow.sklearn.log_model(model, "model")

        print(f"{run_name} -> recall={metrics['recall']:.3f}, roc_auc={metrics['roc_auc']:.3f}")

print("\nDone. Run `mlflow ui` in this directory, then open http://localhost:5000")
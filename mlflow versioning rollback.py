"""
MLflow Model Registry — Versioning & Rollback
-------------------------------------------------
Simulates the real-world scenario: a new model version (v2) gets
registered and promoted, but turns out to perform worse than the
current Production version (v1). We detect this and roll back.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score
import numpy as np
import pandas as pd

client = MlflowClient()
MODEL_NAME = "churn-predictor"

# -----------------------------------------------------------------
# 0. Recreate the same dataset (in a real project this would be a
#    shared data-loading module, not copy-pasted — noting that for
#    when you clean this up later)
# -----------------------------------------------------------------
np.random.seed(42)
n_customers = 2000
tenure_months = np.random.randint(1, 1002, n_customers)
monthly_charges = np.round(np.random.uniform(20, 120, n_customers), 2)
num_support_calls = np.random.poisson(1.5, n_customers)
contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], n_customers, p=[0.55, 0.25, 0.2])
payment_method = np.random.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n_customers)
has_tech_support = np.random.choice([0, 1], n_customers, p=[0.6, 0.4])
churn_score = (
    -0.04 * tenure_months + 0.02 * monthly_charges + 0.35 * num_support_calls
    - 0.8 * has_tech_support
    + np.where(contract_type == "Month-to-month", 1.2, 0)
    + np.where(contract_type == "One year", 0.3, 0)
    + np.random.normal(0, 1, n_customers)
)
churn_prob = 1 / (1 + np.exp(-churn_score))
churn = (churn_prob > np.random.uniform(0.4, 0.6, n_customers)).astype(int)
df = pd.DataFrame({
    "tenure_months": tenure_months, "monthly_charges": monthly_charges,
    "num_support_calls": num_support_calls, "contract_type": contract_type,
    "payment_method": payment_method, "has_tech_support": has_tech_support, "churn": churn,
})
df_encoded = pd.get_dummies(df, columns=["contract_type", "payment_method"], drop_first=True)
X, y = df_encoded.drop(columns=["churn"]), df_encoded["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------------------------------------------
# 1. Train a deliberately WORSE model to simulate a bad v2
#    (shallow tree, no class balancing -> weaker recall on churners)
# -----------------------------------------------------------------
mlflow.set_experiment("customer-churn-prediction")

with mlflow.start_run(run_name="rf_v2_regression_candidate"):
    bad_params = {"n_estimators": 20, "max_depth": 2, "class_weight": None}
    mlflow.log_params(bad_params)

    model_v2 = RandomForestClassifier(random_state=42, **bad_params)
    model_v2.fit(X_train, y_train)

    preds = model_v2.predict(X_test)
    probs = model_v2.predict_proba(X_test)[:, 1]
    recall_v2 = recall_score(y_test, preds)
    auc_v2 = roc_auc_score(y_test, probs)
    mlflow.log_metric("recall", recall_v2)
    mlflow.log_metric("roc_auc", auc_v2)
    mlflow.sklearn.log_model(model_v2, "model")

    run_id_v2 = mlflow.active_run().info.run_id

print(f"v2 candidate -> recall={recall_v2:.3f}, roc_auc={auc_v2:.3f}")

# -----------------------------------------------------------------
# 2. Register v2 and promote it (as if someone did this without checking)
# -----------------------------------------------------------------
registered_v2 = mlflow.register_model(f"runs:/{run_id_v2}/model", MODEL_NAME)
client.transition_model_version_stage(
    name=MODEL_NAME, version=registered_v2.version,
    stage="Production", archive_existing_versions=True,
)
print(f"'{MODEL_NAME}' version {registered_v2.version} promoted to Production (this is the mistake)")

# -----------------------------------------------------------------
# 3. Detect the regression: compare current Production metrics
#    against the previous version
# -----------------------------------------------------------------
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
all_versions_sorted = sorted(all_versions, key=lambda v: int(v.version))

current_prod = next(v for v in all_versions_sorted if v.current_stage == "Production")
previous_version = next(v for v in all_versions_sorted if int(v.version) == int(current_prod.version) - 1)

prod_run = client.get_run(current_prod.run_id)
prev_run = client.get_run(previous_version.run_id)

prod_recall = prod_run.data.metrics.get("recall")
prev_recall = prev_run.data.metrics.get("recall")

print(f"\nCurrent Production (v{current_prod.version}) recall = {prod_recall:.3f}")
print(f"Previous version   (v{previous_version.version}) recall = {prev_recall:.3f}")

# -----------------------------------------------------------------
# 4. Roll back if the new version regressed
# -----------------------------------------------------------------
if prod_recall < prev_recall:
    print(f"\nRegression detected. Rolling back to v{previous_version.version}...")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=previous_version.version,
        stage="Production", archive_existing_versions=True,
    )
    print(f"v{previous_version.version} is Production again. v{current_prod.version} archived.")
else:
    print("\nNo regression detected — new version stays in Production.")
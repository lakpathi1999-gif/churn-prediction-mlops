"""
MLflow Model Registry — Promote the Best Run to Production
-------------------------------------------------------------
Finds the run with the highest recall in the churn experiment,
registers its model under a named model, and promotes it to
the "Production" stage.
"""

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "churn-predictor"

# -----------------------------------------------------------------
# 1. Find the best run automatically (highest recall)
# -----------------------------------------------------------------
experiment = client.get_experiment_by_name("customer-churn-prediction")

best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.recall DESC"],
    max_results=1,
)[0]

print(f"Best run: {best_run.info.run_name} (id={best_run.info.run_id})")
print(f"  recall  = {best_run.data.metrics['recall']:.3f}")
print(f"  roc_auc = {best_run.data.metrics['roc_auc']:.3f}")

# -----------------------------------------------------------------
# 2. Register that run's model under a named model
# -----------------------------------------------------------------
model_uri = f"runs:/{best_run.info.run_id}/model"
registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

print(f"\nRegistered as '{MODEL_NAME}' version {registered_model.version}")

# -----------------------------------------------------------------
# 3. Promote that version to "Production"
#    (Note: this stage-based API is being replaced by "aliases" in
#    newer MLflow versions, but stages are still widely used and
#    the clearest way to learn the concept first.)
# -----------------------------------------------------------------
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True,  # demotes any prior "Production" version to "Archived"
)

print(f"'{MODEL_NAME}' version {registered_model.version} is now in Production stage.")
print("\nLoad it anytime in another script with:")
print(f'  model = mlflow.sklearn.load_model("models:/{MODEL_NAME}/Production")')
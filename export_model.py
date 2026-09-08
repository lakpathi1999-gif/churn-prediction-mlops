"""
Export the current @production model from MLflow Registry to a plain
joblib file -- run this ONCE locally whenever you want to "freeze" a
new model version into the Docker image.
"""

import mlflow.sklearn
import joblib

model = mlflow.sklearn.load_model("models:/churn-predictor@production")
joblib.dump(model, "churn_model.pkl")

print("Saved current @production model to churn_model.pkl")
print(f"Model type: {type(model).__name__}")
print(f"Expects {len(model.feature_names_in_)} features")
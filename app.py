"""
Churn Prediction API
--------------------------------------------------------------------------------
Loads the model ONCE at startup (not per-request -- that would be slow),
then exposes a /predict endpoint that takes raw customer fields and
returns a churn prediction + probability. Root ("/") serves the
standalone prediction UI directly.

Run with:
    uvicorn app:app --reload
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_PATH = "churn_model.pkl"

# Docs/OpenAPI disabled so users only see the standalone prediction UI.
app = FastAPI(title="Churn Prediction API", version="1.0", docs_url=None, redoc_url=None, openapi_url=None)

# Mount the static UI assets, guarded so a missing `ui/` folder (e.g. a
# stripped-down test environment) never crashes the whole app on import.
if os.path.isdir("ui"):
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# -----------------------------------------------------------------
# Load the model once when the server starts -- NOT inside the
# /predict function. Loading a model is slow; do it once, reuse it
# for every request. This is a common beginner mistake worth avoiding.
#
# This model file is bundled directly into the Docker image (see
# Dockerfile) -- no MLflow server needs to be reachable at runtime.
# To update the live model, run export_model.py locally to refresh
# churn_model.pkl, then rebuild the image.
# -----------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Could not load model from '{MODEL_PATH}'. "
        f"Make sure you've run export_model.py first, and that "
        f"churn_model.pkl exists in this directory.\nOriginal error: {e}"
    )


# -----------------------------------------------------------------
# Request schema -- FastAPI + Pydantic validate incoming JSON
# automatically, and reject bad requests before they ever reach
# the model (wrong types, missing fields, out-of-range values).
# -----------------------------------------------------------------
class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=100, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_calls: int = Field(..., ge=0)
    contract_type: str = Field(..., description="Month-to-month | One year | Two year")
    payment_method: str = Field(..., description="Electronic check | Credit card | Bank transfer | Mailed check")
    has_tech_support: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months": 3,
                "monthly_charges": 95.5,
                "total_charges": 286.5,
                "num_support_calls": 4,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "has_tech_support": 0,
            }
        }


@app.get("/")
def root():
    # Serve the static UI index directly at the root path so visiting
    # http://host:8000/ shows the prediction UI (not FastAPI docs).
    return FileResponse("ui/index.html")


@app.get("/health")
def health():
    # FIX: this previously referenced MODEL_URI, a name that no longer
    # exists after switching from MLflow registry loading to a bundled
    # local file -- that leftover reference is what crashed this endpoint.
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(customer: CustomerInput):
    # Convert the incoming request into the same encoded shape the model was trained on.
    input_df = pd.DataFrame([customer.model_dump()])
    input_encoded = pd.get_dummies(input_df, columns=["contract_type", "payment_method"])

    # Align columns to exactly what the model expects. Any dummy column
    # that doesn't appear for THIS customer (e.g. they're not "Two year")
    # gets filled with 0 -- this is what makes single-row inference safe
    # even though one-hot encoding a single row can't produce every category.
    try:
        input_aligned = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model doesn't expose feature_names_in_ -- was it trained on a DataFrame with named columns?"
        )

    prediction = model.predict(input_aligned)[0]
    probability = model.predict_proba(input_aligned)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "will_likely_churn": bool(prediction == 1),
    }
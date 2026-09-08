"""
Automated tests for the Churn Prediction API.
Run locally with: pytest -v

These use FastAPI's TestClient, which calls the app directly in-process
(no need to have `uvicorn` running separately) -- but it DOES load the
real churn_model.pkl at import time, same as the real app would.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 95.5,
    "total_charges": 286.5,
    "num_support_calls": 4,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "has_tech_support": 0,
}


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint_returns_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_with_valid_input_returns_expected_fields():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert "will_likely_churn" in body
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_with_missing_field_returns_422():
    bad_input = VALID_CUSTOMER.copy()
    del bad_input["tenure_months"]  # required field missing

    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_predict_with_invalid_type_returns_422():
    bad_input = VALID_CUSTOMER.copy()
    bad_input["tenure_months"] = "not a number"

    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_with_out_of_range_value_returns_422():
    bad_input = VALID_CUSTOMER.copy()
    bad_input["has_tech_support"] = 5  # only 0 or 1 allowed

    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

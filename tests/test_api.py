"""Pruebas de integración de la API FastAPI (requieren el modelo entrenado --
ver `conftest.py`, que lo genera automáticamente si hace falta)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID_CUSTOMER = {
    "credit_score": 650,
    "geography": "North",
    "gender": "Female",
    "age": 40,
    "tenure_years": 5,
    "balance": 20000.0,
    "num_products": 2,
    "has_credit_card": True,
    "is_active_member": True,
    "estimated_salary": 60000.0,
    "num_complaints": 0,
    "satisfaction_score": 4,
    "monthly_fee_revenue": 60.0,
}

HIGH_RISK_CUSTOMER = {
    "credit_score": 500,
    "geography": "West",
    "gender": "Male",
    "age": 30,
    "tenure_years": 0,
    "balance": 0.0,
    "num_products": 1,
    "has_credit_card": False,
    "is_active_member": False,
    "estimated_salary": 25000.0,
    "num_complaints": 6,
    "satisfaction_score": 1,
    "monthly_fee_revenue": 20.0,
}


@pytest.fixture(scope="module")
def client():
    # Como context manager para que el lifespan (carga del modelo) se dispare.
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["optimal_threshold"] <= 1.0
    assert body["campaign_cost"] > 0
    assert 0.0 < body["retention_success_rate"] <= 1.0


def test_predict_returns_valid_probability(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["recommended_action"] in ("contact", "no_contact")
    assert body["expected_financial_value"] is None  # no se envió ltv


def test_predict_with_ltv_includes_expected_financial_value(client):
    payload = {**VALID_CUSTOMER, "ltv": 2000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["expected_financial_value"] is not None


def test_predict_high_risk_customer_is_flagged_for_contact(client):
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] > 0.5
    assert body["recommended_action"] == "contact"


def test_predict_batch_returns_one_prediction_per_customer(client):
    response = client.post("/predict/batch", json=[VALID_CUSTOMER, HIGH_RISK_CUSTOMER])
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2


def test_predict_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400


def test_predict_rejects_invalid_geography(client):
    invalid = {**VALID_CUSTOMER, "geography": "Atlantis"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_satisfaction_score(client):
    invalid = {**VALID_CUSTOMER, "satisfaction_score": 10}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

"""API de inferencia en tiempo real para el sistema de churn y retención financiera.

Sirve el modelo LightGBM entrenado en `src/models/train.py`, aplicando el mismo
umbral de decisión optimizado por retorno financiero (no 0.5) que se validó ahí.
Si el cliente incluye su LTV proyectado en la request, la respuesta agrega el valor
financiero esperado de la decisión — la API no solo dice "va a fugarse", dice
"conviene gastar en retenerlo".
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_metadata.json"

_state: dict = {"model": None, "metadata": {}}


def _load_artifacts() -> None:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo entrenado en {MODEL_PATH}. "
            "Corré `python -m src.models.train` antes de levantar la API."
        )
    _state["model"] = joblib.load(MODEL_PATH)
    _state["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Customer Churn & Financial Retention API",
    description=(
        "Predicción de fuga de clientes con recomendación de retención basada en "
        "retorno financiero esperado, no solo en probabilidad de churn."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    """Atributos de un cliente — mismas columnas que usó `src/models/train.py` para entrenar."""

    credit_score: int = Field(ge=300, le=850)
    geography: Literal["North", "South", "East", "West", "Central"]
    gender: Literal["Male", "Female"]
    age: int = Field(ge=18, le=100)
    tenure_years: int = Field(ge=0, le=50)
    balance: float = Field(ge=0)
    num_products: int = Field(ge=1, le=4)
    has_credit_card: bool
    is_active_member: bool
    estimated_salary: float = Field(ge=0)
    num_complaints: int = Field(ge=0, le=20)
    satisfaction_score: int = Field(ge=1, le=5)
    monthly_fee_revenue: float = Field(ge=0)
    ltv: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "LTV proyectado del cliente. Si se envía, la respuesta agrega el valor "
            "financiero esperado de la decisión de contactarlo o no."
        ),
    )


class ChurnPrediction(BaseModel):
    churn_probability: float
    optimal_threshold: float
    recommended_action: Literal["contact", "no_contact"]
    expected_financial_value: Optional[float] = None


class ModelInfo(BaseModel):
    feature_columns: list[str]
    categorical_features: list[str]
    optimal_threshold: float
    campaign_cost: float
    retention_success_rate: float
    roc_auc: float
    pr_auc: float
    test_financial_value: float
    mlflow_run_id: str


def _to_feature_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    metadata = _state["metadata"]
    rows = []
    for customer in customers:
        row = customer.model_dump(exclude={"ltv"})
        row["has_credit_card"] = int(row["has_credit_card"])
        row["is_active_member"] = int(row["is_active_member"])
        rows.append(row)

    df = pd.DataFrame(rows)
    # LightGBM exige que las categóricas lleguen como dtype "category" -- no hace
    # falta replicar los niveles exactos de entrenamiento (el booster ya guarda su
    # propio mapeo interno vía pandas_categorical), pero sin este cast falla.
    for col in metadata["categorical_features"]:
        df[col] = df[col].astype("category")
    return df[metadata["feature_columns"]]


def _score_batch(customers: list[CustomerFeatures]) -> list[ChurnPrediction]:
    model = _state["model"]
    metadata = _state["metadata"]
    threshold = metadata["optimal_threshold"]
    cost = metadata["campaign_cost"]
    success_rate = metadata["retention_success_rate"]

    features = _to_feature_frame(customers)
    probabilities = model.predict_proba(features)[:, 1]

    predictions = []
    for customer, proba in zip(customers, probabilities):
        proba = float(proba)
        recommended_action = "contact" if proba >= threshold else "no_contact"

        expected_value = None
        if customer.ltv is not None:
            ev_contact = proba * success_rate * customer.ltv - cost
            ev_no_contact = -proba * customer.ltv
            expected_value = round(ev_contact if recommended_action == "contact" else ev_no_contact, 2)

        predictions.append(ChurnPrediction(
            churn_probability=round(proba, 4),
            optimal_threshold=threshold,
            recommended_action=recommended_action,
            expected_financial_value=expected_value,
        ))
    return predictions


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if not _state["metadata"]:
        raise HTTPException(status_code=503, detail="Modelo no cargado todavía.")
    return ModelInfo(**_state["metadata"])


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures) -> ChurnPrediction:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado todavía.")
    return _score_batch([customer])[0]


@app.post("/predict/batch", response_model=list[ChurnPrediction])
def predict_batch(customers: list[CustomerFeatures]) -> list[ChurnPrediction]:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado todavía.")
    if not customers:
        raise HTTPException(status_code=400, detail="La lista de clientes no puede estar vacía.")
    return _score_batch(customers)

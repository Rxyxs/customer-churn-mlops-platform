"""UI interactiva: predicción individual (vía la API) + simulador de ROI de la
campaña de retención.

El simulador es el corazón de esta app: deja ajustar el costo de campaña y la tasa
de éxito de retención en vivo, y recalcula el umbral óptimo y el retorno financiero
resultante sobre la base de clientes — para que un analista de negocio pueda
explorar "¿y si la oferta cuesta menos?" sin tocar código.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.models.train import compute_financial_value, find_optimal_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customers.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_metadata.json"

# En Docker Compose (Fase 3) esto se sobreescribe con el nombre del servicio de la API.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn & Financial Retention", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def score_customers(_model, feature_columns: list[str], categorical_features: list[str]) -> np.ndarray:
    """Probabilidad de churn para toda la base — cacheada porque no cambia entre
    corridas del simulador (solo cambian costo/tasa, no el modelo ni los datos)."""
    df = load_customers().copy()
    for col in categorical_features:
        df[col] = df[col].astype("category")
    return _model.predict_proba(df[feature_columns])[:, 1]


def render_roi_simulator(model, metadata: dict) -> None:
    st.header("Simulador de ROI de la campaña de retención")
    st.caption(
        "Recalcula el umbral óptimo y el retorno financiero neto sobre toda la base "
        "de clientes, con el costo de campaña y la tasa de éxito que elijas — no son "
        "valores fijos del entrenamiento, son supuestos de negocio que podés explorar."
    )

    customers = load_customers()
    proba = score_customers(model, metadata["feature_columns"], metadata["categorical_features"])
    y_true = customers["churn"].to_numpy()
    ltv = customers["ltv"].to_numpy()

    col1, col2 = st.columns(2)
    with col1:
        campaign_cost = st.slider(
            "Costo de la oferta de retención por cliente ($)",
            min_value=10.0, max_value=500.0, value=float(metadata["campaign_cost"]), step=5.0,
        )
    with col2:
        success_rate = st.slider(
            "Probabilidad de retener a un cliente con la oferta",
            min_value=0.05, max_value=0.80, value=float(metadata["retention_success_rate"]), step=0.01,
        )

    best_threshold, best_value, thresholds, values = find_optimal_threshold(
        y_true, proba, ltv, thresholds=np.round(np.arange(0.005, 1.0, 0.005), 3)
    )

    everyone_value = compute_financial_value(y_true, np.ones_like(y_true), ltv, campaign_cost, success_rate)
    nobody_value = compute_financial_value(y_true, np.zeros_like(y_true), ltv, campaign_cost, success_rate)
    n_contacted = int((proba >= best_threshold).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Umbral óptimo", f"{best_threshold:.3f}")
    m2.metric("Valor neto (óptimo)", f"${best_value:,.0f}")
    m3.metric("Valor neto (contactar a todos)", f"${everyone_value:,.0f}")
    m4.metric("Clientes a contactar", f"{n_contacted:,} / {len(customers):,}")

    st.subheader("Retorno financiero vs. umbral de decisión")
    curve_df = pd.DataFrame({"umbral": thresholds, "valor_neto": values}).set_index("umbral")
    st.line_chart(curve_df)

    st.subheader("Comparación de estrategias")
    comparison_df = pd.DataFrame({
        "Estrategia": ["No contactar a nadie", "Contactar a todos", "Umbral óptimo (modelo)"],
        "Valor neto ($)": [nobody_value, everyone_value, best_value],
    }).set_index("Estrategia")
    st.bar_chart(comparison_df)


def render_individual_prediction() -> None:
    st.header("Predicción individual")
    st.caption(f"Llama a la API en `{API_URL}` — necesita estar corriendo (`uvicorn src.api.main:app`).")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input("Credit score", 300, 850, 650)
            geography = st.selectbox("Geografía", ["North", "South", "East", "West", "Central"])
            gender = st.selectbox("Género", ["Male", "Female"])
            age = st.number_input("Edad", 18, 100, 40)
        with c2:
            tenure_years = st.number_input("Antigüedad (años)", 0, 50, 5)
            balance = st.number_input("Balance ($)", 0.0, 500_000.0, 20_000.0, step=1000.0)
            num_products = st.selectbox("Número de productos", [1, 2, 3, 4])
            estimated_salary = st.number_input("Salario estimado ($)", 0.0, 500_000.0, 60_000.0, step=1000.0)
        with c3:
            has_credit_card = st.checkbox("Tiene tarjeta de crédito", value=True)
            is_active_member = st.checkbox("Miembro activo", value=True)
            num_complaints = st.number_input("Quejas registradas", 0, 20, 0)
            satisfaction_score = st.slider("Satisfacción (1-5)", 1, 5, 3)

        monthly_fee_revenue = st.number_input("Ingreso mensual del cliente ($)", 0.0, 1000.0, 60.0, step=5.0)
        ltv = st.number_input("LTV proyectado ($) — opcional, deja en 0 para omitir", 0.0, 50_000.0, 0.0, step=100.0)

        submitted = st.form_submit_button("Predecir")

    if not submitted:
        return

    payload = {
        "credit_score": credit_score, "geography": geography, "gender": gender, "age": age,
        "tenure_years": tenure_years, "balance": balance, "num_products": num_products,
        "has_credit_card": has_credit_card, "is_active_member": is_active_member,
        "estimated_salary": estimated_salary, "num_complaints": num_complaints,
        "satisfaction_score": satisfaction_score, "monthly_fee_revenue": monthly_fee_revenue,
    }
    if ltv > 0:
        payload["ltv"] = ltv

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"No se pudo contactar a la API en {API_URL}: {exc}")
        return

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Probabilidad de churn", f"{result['churn_probability']:.1%}")
    action_label = "📞 Contactar" if result["recommended_action"] == "contact" else "✅ No contactar"
    r2.metric("Recomendación", action_label)
    if result["expected_financial_value"] is not None:
        r3.metric("Valor financiero esperado", f"${result['expected_financial_value']:,.2f}")
    else:
        r3.metric("Valor financiero esperado", "— (sin LTV)")


def main() -> None:
    st.title("💳 Customer Churn & Financial Retention System")
    st.caption(
        "Sistema de predicción de fuga de clientes orientado a retorno financiero, "
        "no solo a métricas de clasificación."
    )

    model = load_model()
    metadata = load_metadata()

    tab_roi, tab_predict = st.tabs(["📊 Simulador de ROI", "🔍 Predicción individual"])
    with tab_roi:
        render_roi_simulator(model, metadata)
    with tab_predict:
        render_individual_prediction()


if __name__ == "__main__":
    main()

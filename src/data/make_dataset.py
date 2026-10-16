"""Dataset de clientes bancarios para el sistema de churn.

Fuente principal (real): "Bank Customer Churn" de Kaggle
(`radheshyamkollipara/bank-customer-churn`, archivo `Customer-Churn-Records.csv`):
10.000 clientes reales de un banco europeo (Francia, Alemania, España) con su
atributo de fuga real (`Exited`). `build_customers()` lo descarga con la API de
Kaggle si no está en `data/raw/`, y lo lleva al esquema del proyecto.

Lo que el dataset real NO trae, y por eso se calcula con supuestos explícitos:
- `monthly_fee_revenue`: ingreso mensual del cliente, estimado como
  15 + 8 x productos + 0,0008 x balance (USD). Es un supuesto de negocio, no un dato.
- `ltv`: ingreso mensual x meses de vida útil futura esperados según actividad.
Ninguna de las dos entra como feature del modelo; solo alimentan el valor financiero.

`num_complaints` viene de la columna real `Complain`, que es una filtración casi
perfecta del target (2.034 de los 2.038 clientes que se fueron tienen queja, contra
10 de 7.962 que se quedaron): la queja se registra junto con la salida. Se conserva
en el dataset pero `src/models/train.py` la excluye de las features.

`generate_customers()` es el generador sintético original, con el mismo esquema que
los datos reales; se usa como respaldo cuando no hay credenciales de Kaggle (CI).

Supuesto de negocio para el LTV: un cliente activo tiene por delante más meses de
vida útil esperados que uno inactivo (más "pegado" al banco). El LTV proyectado es
el ingreso mensual actual del cliente multiplicado por esos meses futuros
esperados — lo que hay en juego si se fuga, no lo ya facturado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "customers.csv"

N_CUSTOMERS = 10_000
RANDOM_SEED = 42

GEOGRAPHIES = ["France", "Germany", "Spain"]
CARD_TYPES = ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]
KAGGLE_DATASET = "radheshyamkollipara/bank-customer-churn"
REAL_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
GENDERS = ["Male", "Female"]

# Meses de vida útil FUTURA esperados, según el nivel de actividad del cliente
# (independiente de cuánta antigüedad ya acumuló).
EXPECTED_FUTURE_MONTHS = {"active": 48, "inactive": 18}
MIN_REMAINING_MONTHS = 12  # piso conservador: siempre se asume al menos 1 año más


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_customers(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Genera el DataFrame sintético completo: atributos, churn (con un modelo de
    riesgo logístico) y LTV proyectado."""
    rng = np.random.default_rng(seed)

    credit_score = rng.normal(650, 100, n_customers).clip(300, 850).round().astype(int)
    geography = rng.choice(GEOGRAPHIES, size=n_customers)
    gender = rng.choice(GENDERS, size=n_customers)
    age = rng.normal(42, 13, n_customers).clip(18, 85).round().astype(int)
    tenure_years = rng.integers(0, 16, size=n_customers)
    num_products = rng.choice([1, 2, 3, 4], size=n_customers, p=[0.45, 0.40, 0.10, 0.05])
    has_credit_card = rng.choice([0, 1], size=n_customers, p=[0.30, 0.70])
    is_active_member = rng.choice([0, 1], size=n_customers, p=[0.45, 0.55])
    estimated_salary = rng.uniform(20_000, 200_000, n_customers).round(2)

    # Balance con inflación de ceros: muchos clientes de un solo producto no
    # mantienen saldo relevante en el banco.
    has_balance = rng.choice([0, 1], size=n_customers, p=[0.6, 0.4])
    balance = np.where(
        num_products == 1,
        has_balance * rng.uniform(1_000, 80_000, n_customers),
        rng.uniform(1_000, 150_000, n_customers),
    ).round(2)

    num_complaints = rng.poisson(0.6, n_customers) + np.where(
        is_active_member == 0, rng.poisson(0.4, n_customers), 0
    )
    num_complaints = num_complaints.clip(0, 8)

    satisfaction_score = rng.integers(1, 6, size=n_customers)
    # Muchas quejas empujan la satisfacción reportada hacia abajo (no son variables independientes).
    satisfaction_score = np.clip(satisfaction_score - (num_complaints > 2).astype(int), 1, 5)

    monthly_fee_revenue = (
        15 + 8 * num_products + 0.0008 * balance + rng.normal(0, 5, n_customers)
    ).clip(5, None).round(2)

    # --- Modelo de riesgo de fuga (logit) ---
    # Menos antigüedad, menor satisfacción, más quejas, inactividad y mono-producto
    # empujan el riesgo hacia arriba; intercepto calibrado empíricamente (con esta
    # semilla) para una tasa de churn ~20%, similar a datasets de churn bancario
    # reales de referencia.
    z = (
        1.8
        - 0.25 * tenure_years
        - 0.75 * satisfaction_score
        + 0.60 * num_complaints
        - 0.90 * is_active_member
        + 0.45 * (num_products == 1).astype(int)
        + 0.01 * (age - 42)
        + rng.normal(0, 0.05, n_customers)
    )
    churn_prob = _sigmoid(z)
    churn = rng.binomial(1, churn_prob)

    # LTV = ingreso mensual actual x meses de vida útil FUTURA esperados, estos
    # últimos según el segmento de actividad (no restando la antigüedad ya
    # transcurrida: cuánto lleva un cliente no predice cuánto le queda, y restar
    # tenure invertiría el incentivo -- los clientes más nuevos, que además son los
    # de mayor riesgo de fuga, terminarían pareciendo los de mayor valor futuro).
    activity_key = np.where(is_active_member == 1, "active", "inactive")
    remaining_months = pd.Series(activity_key).map(EXPECTED_FUTURE_MONTHS).to_numpy()
    remaining_months = np.maximum(remaining_months, MIN_REMAINING_MONTHS)
    ltv = (monthly_fee_revenue * remaining_months).round(2)

    return pd.DataFrame({
        "customer_id": [f"C-{i:06d}" for i in range(n_customers)],
        "credit_score": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure_years": tenure_years,
        "balance": balance,
        "num_products": num_products,
        "has_credit_card": has_credit_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
        "num_complaints": num_complaints,
        "satisfaction_score": satisfaction_score,
        "card_type": rng.choice(CARD_TYPES, size=n_customers),
        "points_earned": rng.integers(119, 1001, size=n_customers),
        "monthly_fee_revenue": monthly_fee_revenue,
        "ltv": ltv,
        "churn": churn,
    })


def monthly_fee_revenue_assumption(num_products: pd.Series, balance: pd.Series) -> pd.Series:
    """Supuesto de ingreso mensual (USD): el dataset real no trae ingresos por cliente."""
    return (15 + 8 * num_products + 0.0008 * balance).clip(lower=5).round(2)


def ltv_assumption(monthly_fee_revenue: pd.Series, is_active_member: pd.Series) -> pd.Series:
    remaining = np.where(is_active_member == 1, EXPECTED_FUTURE_MONTHS["active"], EXPECTED_FUTURE_MONTHS["inactive"])
    return (monthly_fee_revenue * np.maximum(remaining, MIN_REMAINING_MONTHS)).round(2)


def download_real(path: Path = REAL_RAW_PATH) -> Path:
    """Descarga el CSV real con la API de Kaggle (requiere ~/.kaggle/kaggle.json)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    path.parent.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_DATASET, path=str(path.parent), unzip=True)
    return path


def real_to_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Lleva el CSV real de Kaggle al esquema del proyecto."""
    df = pd.DataFrame({
        "customer_id": "C-" + raw["CustomerId"].astype(str),
        "credit_score": raw["CreditScore"],
        "geography": raw["Geography"],
        "gender": raw["Gender"],
        "age": raw["Age"],
        "tenure_years": raw["Tenure"],
        "balance": raw["Balance"],
        "num_products": raw["NumOfProducts"],
        "has_credit_card": raw["HasCrCard"],
        "is_active_member": raw["IsActiveMember"],
        "estimated_salary": raw["EstimatedSalary"],
        "num_complaints": raw["Complain"],
        "satisfaction_score": raw["Satisfaction Score"],
        "card_type": raw["Card Type"],
        "points_earned": raw["Point Earned"],
    })
    df["monthly_fee_revenue"] = monthly_fee_revenue_assumption(df["num_products"], df["balance"])
    df["ltv"] = ltv_assumption(df["monthly_fee_revenue"], df["is_active_member"])
    df["churn"] = raw["Exited"].astype(int)
    return df


def build_customers(allow_synthetic: bool = True) -> tuple[pd.DataFrame, str]:
    """Devuelve (dataset, fuente). Usa el CSV real; si no está, intenta bajarlo de
    Kaggle; si no hay credenciales y `allow_synthetic`, cae al generador sintético."""
    if not REAL_RAW_PATH.exists():
        try:
            download_real()
        except Exception as exc:  # sin credenciales o sin red
            if not allow_synthetic:
                raise
            print(f"Aviso: no se pudo descargar el dataset real ({exc}); se usa el generador sintético.")
            return generate_customers(), "synthetic"
    return real_to_schema(pd.read_csv(REAL_RAW_PATH)), "real"


if __name__ == "__main__":
    df, source = build_customers()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fuente: {source} | clientes: {len(df):,} | tasa de churn: {df['churn'].mean():.2%}")
    print(f"Guardado en {OUTPUT_PATH}")

"""Fixtures compartidas de la suite de integración.

Garantiza que exista un dataset y un modelo entrenado antes de correr cualquier
test -- los genera si hace falta, para que `pytest tests/` funcione en un clone
limpio o en CI sin pasos manuales previos.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customers.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model.joblib"


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model() -> None:
    if not RAW_DATA_PATH.exists():
        from src.data.make_dataset import OUTPUT_PATH, generate_customers

        df = generate_customers()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

    if not MODEL_PATH.exists():
        from src.models.train import run_training

        run_training()

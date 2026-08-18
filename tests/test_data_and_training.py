"""Pruebas de integración: generación sintética de datos y lógica de negocio del
entrenamiento (optimización de umbral por retorno financiero)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.make_dataset import generate_customers
from src.models.train import compute_financial_value, find_optimal_threshold, load_data, split_data


def test_generate_customers_schema_and_ranges():
    df = generate_customers(n_customers=500, seed=1)
    assert len(df) == 500
    assert set(df["churn"].unique()) <= {0, 1}
    assert (df["ltv"] > 0).all()
    assert df["credit_score"].between(300, 850).all()
    assert df["satisfaction_score"].between(1, 5).all()


def test_generate_customers_churn_rate_is_realistic():
    df = generate_customers(n_customers=5000, seed=1)
    churn_rate = df["churn"].mean()
    assert 0.10 < churn_rate < 0.30


def test_compute_financial_value_perfect_classifier():
    y_true = np.array([1, 1, 0, 0])
    y_pred = y_true.copy()
    ltv = np.array([1000.0, 2000.0, 500.0, 800.0])

    value = compute_financial_value(y_true, y_pred, ltv, campaign_cost=100, success_rate=0.3)
    # TP: (0.3*1000-100) + (0.3*2000-100) = 200 + 500 = 700; sin FP ni FN.
    assert value == pytest.approx(700.0)


def test_compute_financial_value_missed_churner_costs_full_ltv():
    y_true = np.array([1])
    y_pred = np.array([0])  # falso negativo
    ltv = np.array([1500.0])

    value = compute_financial_value(y_true, y_pred, ltv, campaign_cost=100, success_rate=0.3)
    assert value == pytest.approx(-1500.0)


def test_compute_financial_value_false_positive_only_costs_campaign():
    y_true = np.array([0])
    y_pred = np.array([1])  # falso positivo
    ltv = np.array([9999.0])  # no debería importar: no era churner

    value = compute_financial_value(y_true, y_pred, ltv, campaign_cost=100, success_rate=0.3)
    assert value == pytest.approx(-100.0)


def test_find_optimal_threshold_beats_a_naive_fixed_threshold():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.2, 1000)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0.1, 0.15, 1000), 0, 1)
    ltv = rng.uniform(500, 3000, 1000)

    best_threshold, best_value, thresholds, values = find_optimal_threshold(y_true, y_proba, ltv)
    naive_value = compute_financial_value(y_true, (y_proba >= 0.5).astype(int), ltv)

    assert 0.0 < best_threshold < 1.0
    assert best_value >= naive_value


def test_load_data_and_split_data_are_stratified_and_disjoint():
    df = load_data()
    train_df, val_df, test_df, feature_columns = split_data(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)

    overall_churn_rate = df["churn"].mean()
    for split in (train_df, val_df, test_df):
        assert abs(split["churn"].mean() - overall_churn_rate) < 0.05

    assert "ltv" not in feature_columns
    assert "churn" not in feature_columns
    assert "customer_id" not in feature_columns

"""Tests unitarios para el módulo de comparación de modelos (src/models/compare_models.py).

No reentrena las 3 familias completas en cada test (sería lento e innecesario);
en su lugar valida las piezas nuevas de forma aislada: la Focal Loss, el forward
pass del MLP para cada activación, el preprocesamiento de features y, en un test
de integración liviano, un ciclo completo de `run_comparison()` con pocas épocas
sobre una muestra pequeña del dataset sintético ya generado por los fixtures de
`conftest.py`.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import torch

from src.models import compare_models as cm
from src.models.train import DATA_PATH, TARGET_COLUMN, load_data, split_data


def test_focal_loss_penalizes_confident_wrong_predictions_more():
    criterion = cm.FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.tensor([1.0])

    confident_wrong = torch.tensor([-5.0])  # predicted prob ~0.007, target=1
    confident_right = torch.tensor([5.0])  # predicted prob ~0.993, target=1

    loss_wrong = criterion(confident_wrong, targets)
    loss_right = criterion(confident_right, targets)

    assert loss_wrong.item() > loss_right.item()
    assert loss_right.item() >= 0


def test_focal_loss_is_finite_and_nonnegative():
    criterion = cm.FocalLoss()
    logits = torch.randn(64)
    targets = torch.randint(0, 2, (64,)).float()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_mlp_forward_shape_for_every_activation():
    x = torch.randn(8, 10)
    for activation in cm.ACTIVATIONS:
        model = cm.ChurnMLP(n_features=10, activation=activation)
        out = model(x)
        assert out.shape == (8,)


def test_swish_matches_x_times_sigmoid():
    x = torch.tensor([-2.0, 0.0, 1.0, 3.0])
    expected = x * torch.sigmoid(x)
    assert torch.allclose(cm.swish(x), expected)


def test_preprocess_features_produces_consistent_shapes():
    df = load_data(DATA_PATH)
    train_df, val_df, test_df, feature_columns = split_data(df)

    X_train, X_val, X_test, columns = cm.preprocess_features(train_df, val_df, test_df, feature_columns)

    assert X_train.shape[0] == len(train_df)
    assert X_val.shape[0] == len(val_df)
    assert X_test.shape[0] == len(test_df)
    # Misma cantidad de columnas tras el one-hot alineado en los 3 splits.
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == len(columns)
    # Estandarizado: media ~0 en train.
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)


def test_binary_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = cm._binary_metrics(y_true, y_proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_run_comparison_end_to_end_smoke(tmp_path, monkeypatch):
    """Ciclo completo con pocas épocas y una muestra chica: valida que el
    pipeline entrena las 3 familias, genera las 4 figuras y persiste en DuckDB
    sin usar los artefactos reales del repo (todo redirigido a tmp_path)."""
    df = load_data(DATA_PATH)
    small_df = df.sample(n=600, random_state=0).reset_index(drop=True)
    small_csv = tmp_path / "small_customers.csv"
    small_df.to_csv(small_csv, index=False)

    figures_dir = tmp_path / "figures"
    duckdb_path = tmp_path / "comparison.duckdb"
    monkeypatch.setattr(cm, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(cm, "DUCKDB_PATH", duckdb_path)

    original_train_mlp = cm.train_mlp

    def fast_train_mlp(*args, **kwargs):
        kwargs["epochs"] = 3
        return original_train_mlp(*args, **kwargs)

    monkeypatch.setattr(cm, "train_mlp", fast_train_mlp)

    summary = cm.run_comparison(data_path=small_csv)

    assert set(summary["comparison"].keys()) == {"logistic_regression", "xgboost", "pytorch_mlp"}
    for metrics in summary["comparison"].values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    assert summary["best_mlp_activation"] in cm.ACTIVATIONS

    for filename in [
        "roc_pr_comparison.png",
        "confusion_matrices.png",
        "mlp_activation_loss_curves.png",
        "mlp_activation_comparison.png",
    ]:
        assert (figures_dir / filename).exists()

    assert duckdb_path.exists()
    con = duckdb.connect(str(duckdb_path))
    n_models = con.execute("SELECT COUNT(*) FROM model_comparison_metrics").fetchone()[0]
    n_activations = con.execute("SELECT COUNT(*) FROM mlp_activation_comparison").fetchone()[0]
    con.close()
    assert n_models == 3
    assert n_activations == len(cm.ACTIVATIONS)

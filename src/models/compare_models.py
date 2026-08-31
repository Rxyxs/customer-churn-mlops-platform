"""Comparación complementaria de enfoques de modelado para el churn de clientes.

`train.py` (el pipeline "productivo" de esta plataforma) ya resuelve la pregunta de
negocio central -- LightGBM + umbral óptimo por retorno financiero -- y ese script
no se toca aquí. Este módulo responde una pregunta distinta y complementaria, típica
de un MLOps platform maduro: *antes* de fijar LightGBM como el modelo servido en
producción, ¿qué tan bien se compara contra un baseline interpretable y contra deep
learning? Es la evidencia de model selection que normalmente vive en un registry o
en un experiment tracker junto al modelo ganador.

Tres enfoques, mismo split 50/30/20 y mismas features que `train.py`:

1. **Regresión logística** (`class_weight="balanced"`): baseline interpretable,
   coeficientes auditables, referencia mínima de calidad.
2. **XGBoost** (`scale_pos_weight`): segundo ensamble de árboles, gradient boosting
   con una librería distinta a la servida en producción (LightGBM), para verificar
   que la elección de librería no es la que explica el desempeño.
3. **PyTorch MLP con Focal Loss** (para el desbalance de churn, ~20% positivos) y
   comparación de activaciones ReLU vs. GELU vs. Swish en la misma arquitectura --
   la pregunta no es "¿qué red es mejor?" sino "¿la elección de no-linealidad mueve
   la aguja en un dataset tabular pequeño?" (spoiler habitual: poco, pero se mide en
   vez de asumirse).

Salidas:
- `reports/figures/*.png`: curvas ROC/PR comparadas, matrices de confusión, curvas
  de loss por época y comparación de activaciones.
- `reports/model_comparison.duckdb`: métricas de las 3 familias + las 3 activaciones
  del MLP, para consulta SQL -- complementa (no reemplaza) el registry de MLflow que
  ya usa `train.py`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.make_dataset import OUTPUT_PATH as RAW_DATA_PATH, generate_customers
from src.models.train import DATA_PATH, ID_COLUMN, LTV_COLUMN, TARGET_COLUMN, load_data, split_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DUCKDB_PATH = PROJECT_ROOT / "reports" / "model_comparison.duckdb"

RANDOM_STATE = 42
ACTIVATIONS = ["relu", "gelu", "swish"]

# Paleta consistente con train.py (azul / naranja) más un tercer tono para el MLP.
COLORS = {"logistic_regression": "#2a78d6", "xgboost": "#eb6834", "pytorch_mlp": "#3aa15c"}
ACTIVATION_COLORS = {"relu": "#2a78d6", "gelu": "#eb6834", "swish": "#3aa15c"}


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return _Swish()
    raise ValueError(f"Activación desconocida: {name}")


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) para el desbalance de clases del churn (~20%
    positivos). Downweighting de ejemplos fáciles (ya bien clasificados) vía
    `(1 - p_t) ** gamma`, para que el gradiente se concentre en los casos difíciles
    -- churners de bajo score y no-churners atípicos -- en vez de diluirse en el
    volumen de negativos fáciles que domina el batch."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        return (alpha_t * focal_term * bce).mean()


class ChurnMLP(nn.Module):
    """MLP simple de 2 capas ocultas; la activación es el único eje que varía entre
    corridas para aislar su efecto."""

    def __init__(self, n_features: int, activation: str = "relu", hidden: tuple[int, int] = (64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            _activation_layer(activation),
            nn.Linear(hidden[0], hidden[1]),
            _activation_layer(activation),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def preprocess_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """One-hot para categóricas + estandarización de numéricas, ajustado solo en
    train -- LR y el MLP necesitan features numéricas densas (a diferencia de
    LightGBM en `train.py`, que maneja categóricas nativamente)."""
    train_enc = pd.get_dummies(train_df[feature_columns], columns=["geography", "gender"])
    val_enc = pd.get_dummies(val_df[feature_columns], columns=["geography", "gender"])
    test_enc = pd.get_dummies(test_df[feature_columns], columns=["geography", "gender"])

    val_enc = val_enc.reindex(columns=train_enc.columns, fill_value=0)
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_enc.to_numpy(dtype=float))
    X_val = scaler.transform(val_enc.to_numpy(dtype=float))
    X_test = scaler.transform(test_enc.to_numpy(dtype=float))
    return X_train, X_val, X_test, list(train_enc.columns)


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=float(n_neg / max(n_pos, 1)),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    activation: str,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = RANDOM_STATE,
) -> tuple[ChurnMLP, dict]:
    torch.manual_seed(seed)
    model = ChurnMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n = X_train_t.shape[0]
    train_losses, val_losses = [], []
    generator = torch.Generator().manual_seed(seed)

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, history


@torch.no_grad()
def mlp_predict_proba(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def _binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def plot_roc_pr_comparison(results: dict[str, dict], y_test: np.ndarray, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, res in results.items():
        color = COLORS.get(name, "#555555")
        disp = RocCurveDisplay.from_predictions(y_test, res["test_proba"], name=name, ax=axes[0])
        disp.line_.set_color(color)
        disp_pr = PrecisionRecallDisplay.from_predictions(y_test, res["test_proba"], name=name, ax=axes[1])
        disp_pr.line_.set_color(color)
    axes[0].set_title("ROC — comparación de modelos (test)")
    axes[1].set_title("Precision-Recall — comparación de modelos (test)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(results: dict[str, dict], y_test: np.ndarray, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = (res["test_proba"] >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1], ["No churn", "Churn"])
        ax.set_yticks([0, 1], ["No churn", "Churn"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_mlp_activation_curves(histories: dict[str, dict], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for activation, history in histories.items():
        color = ACTIVATION_COLORS.get(activation, "#555555")
        ax.plot(history["val_loss"], label=f"{activation} (val)", color=color, linewidth=2)
        ax.plot(history["train_loss"], label=f"{activation} (train)", color=color, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Época")
    ax.set_ylabel("Focal Loss")
    ax.set_title("MLP: curvas de loss por activación (ReLU vs. GELU vs. Swish)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_activation_metric_bars(activation_metrics: dict[str, dict], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    activations = list(activation_metrics.keys())
    metrics = ["roc_auc", "pr_auc", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, activation in enumerate(activations):
        values = [activation_metrics[activation][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=activation, color=ACTIVATION_COLORS.get(activation))
    ax.set_xticks(x + width, metrics)
    ax.set_ylim(0, 1)
    ax.set_title("MLP: métricas de test por activación")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def persist_to_duckdb(
    comparison_rows: list[dict], activation_rows: list[dict], db_path: Path | None = None
) -> None:
    """Persiste métricas comparativas en DuckDB local. Complementa el Model
    Registry de MLflow que usa `train.py` para el modelo servido en producción --
    esta base es para model selection / auditoría, consultable con SQL plano."""
    if db_path is None:
        db_path = DUCKDB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS model_comparison_metrics (
            run_timestamp TIMESTAMP,
            model_name VARCHAR,
            roc_auc DOUBLE,
            pr_auc DOUBLE,
            f1 DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mlp_activation_comparison (
            run_timestamp TIMESTAMP,
            activation VARCHAR,
            roc_auc DOUBLE,
            pr_auc DOUBLE,
            f1 DOUBLE,
            final_train_loss DOUBLE,
            final_val_loss DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO model_comparison_metrics VALUES (?, ?, ?, ?, ?)",
        [(row["run_timestamp"], row["model_name"], row["roc_auc"], row["pr_auc"], row["f1"]) for row in comparison_rows],
    )
    con.executemany(
        "INSERT INTO mlp_activation_comparison VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["run_timestamp"],
                row["activation"],
                row["roc_auc"],
                row["pr_auc"],
                row["f1"],
                row["final_train_loss"],
                row["final_val_loss"],
            )
            for row in activation_rows
        ],
    )
    con.close()


def run_comparison(data_path: Path = DATA_PATH) -> dict:
    if not data_path.exists():
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_customers().to_csv(RAW_DATA_PATH, index=False)

    df = load_data(data_path)
    train_df, val_df, test_df, feature_columns = split_data(df)

    X_train, X_val, X_test, _ = preprocess_features(train_df, val_df, test_df, feature_columns)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_val = val_df[TARGET_COLUMN].to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    results: dict[str, dict] = {}

    lr_model = train_logistic_regression(X_train, y_train)
    results["logistic_regression"] = {"test_proba": lr_model.predict_proba(X_test)[:, 1]}

    xgb_model = train_xgboost(X_train, y_train)
    results["xgboost"] = {"test_proba": xgb_model.predict_proba(X_test)[:, 1]}

    # --- Comparación de activaciones del MLP ---
    histories = {}
    activation_metrics = {}
    activation_rows = []
    run_timestamp = pd.Timestamp.utcnow().tz_localize(None)
    best_activation, best_pr_auc, best_model = None, -1.0, None
    for activation in ACTIVATIONS:
        model, history = train_mlp(X_train, y_train, X_val, y_val, activation=activation)
        val_proba = mlp_predict_proba(model, X_val)
        metrics = _binary_metrics(y_val, val_proba)
        histories[activation] = history
        activation_metrics[activation] = metrics
        activation_rows.append({
            "run_timestamp": run_timestamp,
            "activation": activation,
            **metrics,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        })
        if metrics["pr_auc"] > best_pr_auc:
            best_activation, best_pr_auc, best_model = activation, metrics["pr_auc"], model

    results["pytorch_mlp"] = {"test_proba": mlp_predict_proba(best_model, X_test), "best_activation": best_activation}

    # --- Métricas de test para las 3 familias ---
    comparison_rows = []
    for name, res in results.items():
        metrics = _binary_metrics(y_test, res["test_proba"])
        res["metrics"] = metrics
        comparison_rows.append({"run_timestamp": run_timestamp, "model_name": name, **metrics})

    # --- Gráficos ---
    plot_roc_pr_comparison(results, y_test, FIGURES_DIR / "roc_pr_comparison.png")
    plot_confusion_matrices(results, y_test, FIGURES_DIR / "confusion_matrices.png")
    plot_mlp_activation_curves(histories, FIGURES_DIR / "mlp_activation_loss_curves.png")
    plot_activation_metric_bars(activation_metrics, FIGURES_DIR / "mlp_activation_comparison.png")

    persist_to_duckdb(comparison_rows, activation_rows, DUCKDB_PATH)

    summary = {
        "comparison": {name: res["metrics"] for name, res in results.items()},
        "best_mlp_activation": best_activation,
        "activation_metrics": activation_metrics,
    }
    (FIGURES_DIR.parent / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Comparación de modelos (test) ===")
    for name, metrics in summary["comparison"].items():
        print(f"{name:>20}: ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}")
    print(f"\nMejor activación del MLP (por PR-AUC en validación): {best_activation}")
    print(f"Figuras guardadas en {FIGURES_DIR}")
    print(f"Métricas persistidas en {DUCKDB_PATH}")

    return summary


if __name__ == "__main__":
    start = time.time()
    run_comparison()
    print(f"\nTiempo total: {time.time() - start:.1f}s")

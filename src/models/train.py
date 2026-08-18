"""Entrenamiento del modelo de churn: LightGBM + tracking en MLflow + optimización
del umbral de decisión por retorno financiero.

La pregunta que responde este script no es solo "¿qué tan bien clasifica el
modelo?" sino "¿en qué punto de corte maximizamos el valor ($) neto de la campaña
de retención?" — ese es el entregable de negocio, no el AUC. El AUC/PR-AUC se
reportan igual porque siguen siendo la señal de si el modelo aprendió algo, pero
la decisión operativa (a quién llamar) la fija el umbral financiero, no 0.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customers.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model.joblib"
METADATA_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_metadata.json"
THRESHOLD_CURVE_PATH = PROJECT_ROOT / "data" / "processed" / "figures" / "threshold_vs_value.png"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"

TARGET_COLUMN = "churn"
LTV_COLUMN = "ltv"
ID_COLUMN = "customer_id"
CATEGORICAL_FEATURES = ["geography", "gender"]

# --- Supuestos de negocio para la optimización financiera del umbral ---
# Costo de una oferta de retención dirigida a un cliente (descuento de comisiones,
# bono, mejora de tasa).
CAMPAIGN_COST = 100.0
# Probabilidad de que la oferta efectivamente retenga a un cliente que iba a fugarse.
RETENTION_SUCCESS_RATE = 0.30
# Con estos supuestos, ofrecer la campaña a TODOS los clientes ya es rentable en
# agregado (hay suficiente LTV en riesgo) -- pero de forma ineficiente, porque
# desperdicia el costo de campaña en clientes que nunca se iban a fugar. El valor del
# modelo no es "convertir una pérdida en ganancia": es concentrar el gasto donde el
# retorno es mayor. El umbral óptimo por retorno financiero captura casi toda la
# recall relevante contactando bastante menos gente que "a todos", por eso su valor
# neto termina siendo sustancialmente mayor que el de la campaña indiscriminada.

MLFLOW_EXPERIMENT_NAME = "customer_churn_retention"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset y tipa las categóricas para que LightGBM las maneje nativamente."""
    df = pd.read_csv(path)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def split_data(df: pd.DataFrame, random_state: int = 42):
    """Split 50/30/20 estratificado por churn.

    train: ajusta el modelo. val: elige el umbral óptimo de retorno financiero — se
    le da un 30% (no el 20% más habitual) porque el LTV tiene cola pesada, y con
    pocas filas unos pocos clientes de alto valor pueden mover el total en decenas
    de miles de dólares; un val más grande estabiliza esa selección. test: holdout
    final, nunca visto durante el entrenamiento ni la optimización del umbral — así
    el número que reportamos al final no está inflado por haber "espiado" los mismos
    datos dos veces.
    """
    feature_columns = [c for c in df.columns if c not in (ID_COLUMN, TARGET_COLUMN, LTV_COLUMN)]

    train_df, temp_df = train_test_split(
        df, test_size=0.5, stratify=df[TARGET_COLUMN], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.4, stratify=temp_df[TARGET_COLUMN], random_state=random_state
    )
    return train_df, val_df, test_df, feature_columns


def train_model(train_df: pd.DataFrame, feature_columns: list[str], random_state: int = 42) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(
        train_df[feature_columns],
        train_df[TARGET_COLUMN],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model


def compute_financial_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ltv: np.ndarray,
    campaign_cost: float = CAMPAIGN_COST,
    success_rate: float = RETENTION_SUCCESS_RATE,
) -> float:
    """Valor financiero neto de aplicar la campaña de retención según `y_pred`.

    - Verdadero positivo (se ataca a un churner real): se espera recuperar
      `success_rate * ltv` de valor; el costo de campaña siempre se paga.
    - Falso positivo (se ataca a alguien que no iba a fugarse): solo costo, sin
      beneficio (no había nada que retener).
    - Falso negativo (churner real no detectado): se pierde su LTV completo — el
      costo de no actuar.
    - Verdadero negativo: sin costo ni beneficio (correctamente no se interviene).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ltv = np.asarray(ltv, dtype=float)

    true_positive = (y_pred == 1) & (y_true == 1)
    false_positive = (y_pred == 1) & (y_true == 0)
    false_negative = (y_pred == 0) & (y_true == 1)

    value = 0.0
    value += (success_rate * ltv[true_positive] - campaign_cost).sum()
    value += (-campaign_cost * np.ones(int(false_positive.sum()))).sum()
    value += (-ltv[false_negative]).sum()
    return float(value)


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, ltv: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Grid search sobre umbrales: el óptimo es el que maximiza el valor financiero
    neto en el set de validación, no el que maximiza F1 o accuracy."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)

    values = np.array([
        compute_financial_value(y_true, (y_proba >= t).astype(int), ltv) for t in thresholds
    ])
    best_idx = int(values.argmax())
    return float(thresholds[best_idx]), float(values[best_idx]), thresholds, values


def plot_threshold_curve(
    thresholds: np.ndarray, values: np.ndarray, best_threshold: float, output_path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, values, color="#2a78d6", linewidth=2)
    ax.axvline(best_threshold, color="#eb6834", linestyle="--", label=f"Óptimo = {best_threshold:.2f}")
    ax.axhline(0, color="#898781", linewidth=1)
    ax.set_xlabel("Umbral de decisión (probabilidad de churn)")
    ax.set_ylabel("Valor financiero neto de la campaña ($)")
    ax.set_title("Retorno financiero vs. umbral de decisión (set de validación)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "report": classification_report(y_true, y_pred, target_names=["No churn", "Churn"], digits=4),
    }


def run_training(data_path: Path = DATA_PATH) -> dict:
    df = load_data(data_path)
    train_df, val_df, test_df, feature_columns = split_data(df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        model = train_model(train_df, feature_columns)

        val_proba = model.predict_proba(val_df[feature_columns])[:, 1]
        best_threshold, best_value, thresholds, values = find_optimal_threshold(
            val_df[TARGET_COLUMN].to_numpy(), val_proba, val_df[LTV_COLUMN].to_numpy()
        )

        test_proba = model.predict_proba(test_df[feature_columns])[:, 1]
        test_metrics = evaluate_at_threshold(test_df[TARGET_COLUMN].to_numpy(), test_proba, best_threshold)
        test_pred = (test_proba >= best_threshold).astype(int)
        test_financial_value = compute_financial_value(
            test_df[TARGET_COLUMN].to_numpy(), test_pred, test_df[LTV_COLUMN].to_numpy()
        )

        mlflow.log_params({
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "num_leaves": model.num_leaves,
            "campaign_cost": CAMPAIGN_COST,
            "retention_success_rate": RETENTION_SUCCESS_RATE,
        })

        # Métricas técnicas: ¿el modelo separa señal de ruido?
        mlflow.log_metrics({
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        })

        # Métricas de negocio: ¿cuánto dinero mueve esta decisión?
        mlflow.log_metrics({
            "optimal_threshold": best_threshold,
            "validation_financial_value": best_value,
            "test_financial_value": test_financial_value,
            "test_financial_value_per_customer": test_financial_value / len(test_df),
        })

        curve_path = plot_threshold_curve(thresholds, values, best_threshold, THRESHOLD_CURVE_PATH)
        mlflow.log_artifact(str(curve_path))

        try:
            mlflow.lightgbm.log_model(model, name="model", registered_model_name="churn_lightgbm")
        except mlflow.exceptions.MlflowException as exc:
            print(f"Aviso: no se pudo registrar el modelo en el Model Registry ({exc}); se loguea sin registrar.")
            mlflow.lightgbm.log_model(model, name="model")

        print(test_metrics["report"])
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"Umbral óptimo (retorno financiero): {best_threshold:.2f}")
        print(f"Valor financiero neto en test: ${test_financial_value:,.2f} ({len(test_df):,} clientes)")
        print(f"MLflow run: {run.info.run_id}")

        # Artefactos locales: permiten servir el modelo (Fase 2, API) sin depender
        # de que el tracking server de MLflow esté disponible en runtime.
        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)

        metadata = {
            "feature_columns": feature_columns,
            "categorical_features": CATEGORICAL_FEATURES,
            "optimal_threshold": best_threshold,
            "campaign_cost": CAMPAIGN_COST,
            "retention_success_rate": RETENTION_SUCCESS_RATE,
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "test_financial_value": test_financial_value,
            "mlflow_run_id": run.info.run_id,
        }
        METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nModelo guardado en {MODEL_OUTPUT_PATH}")
        print(f"Metadata guardada en {METADATA_OUTPUT_PATH}")

        return metadata


if __name__ == "__main__":
    run_training()

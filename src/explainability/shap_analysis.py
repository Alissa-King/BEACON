"""
BEACON XAI layer: SHAP-based explainability for the XGBoost model.

Produces:
  - Global feature importance (SHAP summary)
  - Per-organization SHAP values for BEAM trigger logic
  - Saved plots to reports/figures/
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.bdi import BDI_FEATURE_COLUMNS

FIGURES_DIR = Path("reports/figures")
MODEL_PATH = Path("models/xgboost.pkl")

FEATURE_LABELS = {
    "months_cash_on_hand": "Months of Cash on Hand",
    "current_ratio": "Current Ratio",
    "unrestricted_net_assets_ratio": "Unrestricted Net Assets Ratio",
    "operating_margin": "Operating Margin",
    "consecutive_deficits": "Consecutive Deficits",
    "gov_grant_concentration": "Gov. Grant Concentration",
    "revenue_hhi": "Revenue HHI (Concentration)",
    "debt_to_equity": "Debt-to-Equity Ratio",
}


def load_model_and_explainer(X_train: pd.DataFrame):
    pipeline = joblib.load(MODEL_PATH)
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X_train)
    explainer = shap.TreeExplainer(clf)
    return explainer, scaler, clf


def compute_shap_values(explainer, scaler, X: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return explainer.shap_values(X_scaled)


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, save: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_labeled = X.rename(columns=FEATURE_LABELS)
    shap_vals = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        X_labeled,
        plot_type="dot",
        max_display=8,
        show=False,
        color_bar=True,
    )
    plt.title("SHAP Summary Plot – BEACON Financial Distress Drivers", fontsize=13, pad=12)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP summary to {path}")
    plt.close()


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, save: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap_vals = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": BDI_FEATURE_COLUMNS, "mean_shap": mean_abs})
        .replace({"feature": FEATURE_LABELS})
        .sort_values("mean_shap", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(importance_df)))
    ax.barh(importance_df["feature"], importance_df["mean_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Global Feature Importance – XGBoost SHAP Values", fontsize=13)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "shap_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP importance to {path}")
    plt.close()


def get_org_shap_drivers(
    shap_values: np.ndarray, idx: int, top_n: int = 3
) -> list[dict]:
    """Return top N SHAP drivers for a single organization row."""
    shap_vals = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]
    row = shap_vals[idx]
    sorted_idx = np.argsort(np.abs(row))[::-1][:top_n]
    drivers = []
    for i in sorted_idx:
        feature = BDI_FEATURE_COLUMNS[i]
        drivers.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "shap_value": round(float(row[i]), 4),
            "direction": "increases" if row[i] > 0 else "decreases",
        })
    return drivers


def run_shap_analysis(df: pd.DataFrame) -> tuple[np.ndarray, shap.TreeExplainer]:
    X = df[BDI_FEATURE_COLUMNS]
    explainer, scaler, clf = load_model_and_explainer(X)
    shap_values = compute_shap_values(explainer, scaler, X)
    plot_shap_summary(shap_values, X)
    plot_shap_bar(shap_values, X)
    return shap_values, explainer

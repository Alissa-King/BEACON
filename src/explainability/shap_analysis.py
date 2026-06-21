"""
BEACON XAI layer: SHAP-based explainability for the XGBoost model.

SHAP values are used for post-hoc interpretation of model outputs and
feature contribution analysis at the organization level. They are
associative explanations of model behavior, not causal explanations of
organizational failure.

Two interpretive outputs:
  1. Global feature importance (mean |SHAP|) — identifies which financial
     characteristics most strongly predict distress across the sample
  2. Domain-level SHAP aggregation — groups features into the four BEACON
     explanatory domains to produce governance-ready risk attribution
     (this is the "explanatory decomposition layer," not the BDI formula)
  3. Per-organization drivers — top-N features explaining a single org's
     BDI score, fed into the BEAM action matrix
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.bdi import BDI_FEATURE_COLUMNS, SHAP_DOMAIN_MAP

FIGURES_DIR = Path("reports/figures")
MODEL_PATH = Path("models/random_forest.pkl")

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


def load_model_and_explainer(X_ref: pd.DataFrame):
    """Load fitted Random Forest pipeline and build a TreeExplainer."""
    pipeline = joblib.load(MODEL_PATH)
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    explainer = shap.TreeExplainer(clf)
    return explainer, scaler, clf


def compute_shap_values(explainer, scaler, X: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler.transform(X)
    sv = explainer.shap_values(X_scaled)
    # XGBoost binary returns 2D array; multi-output returns 3D — normalise
    return sv if sv.ndim == 2 else sv[:, :, 1]


def compute_shap_domain_contributions(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate SHAP values into the four BEACON explanatory domains.

    Returns a DataFrame with one row per observation and one column per domain,
    containing the mean signed SHAP contribution for that domain.
    A positive domain contribution indicates the domain is increasing the
    predicted distress probability for that organization.
    """
    domain_shap = {}
    feat_idx = {f: i for i, f in enumerate(BDI_FEATURE_COLUMNS)}
    for domain, features in SHAP_DOMAIN_MAP.items():
        idxs = [feat_idx[f] for f in features if f in feat_idx]
        if idxs:
            domain_shap[domain] = shap_values[:, idxs].sum(axis=1)
    return pd.DataFrame(domain_shap, index=X.index)


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_labeled = X.rename(columns=FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_labeled,
        plot_type="dot", max_display=8, show=False, color_bar=True,
    )
    plt.title(
        "SHAP Summary: Feature Contributions to P(Distress)\n"
        "(associative predictors — not causal drivers)",
        fontsize=12, pad=10,
    )
    plt.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary to {path}")


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": BDI_FEATURE_COLUMNS, "mean_shap": mean_abs})
        .replace({"feature": FEATURE_LABELS})
        .sort_values("mean_shap", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(importance_df)))
    ax.barh(importance_df["feature"], importance_df["mean_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP Value| — contribution to P(Distress)", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP) — Random Forest", fontsize=12)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP importance to {path}")


def plot_shap_domain_contributions(domain_df: pd.DataFrame) -> None:
    """Bar chart of mean signed SHAP contribution per BEACON domain."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    means = domain_df.mean()
    colors = ["#c0392b" if v > 0 else "#2e8b57" for v in means]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(means.index, means.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean SHAP contribution to P(Distress)", fontsize=11)
    ax.set_title(
        "SHAP Domain-Level Explanatory Decomposition\n"
        "(red = increases distress risk on average; green = protective)",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIGURES_DIR / "shap_domain_contributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved domain contributions to {path}")


def get_org_shap_drivers(
    shap_values: np.ndarray, idx: int, top_n: int = 3
) -> list[dict]:
    """Top-N SHAP drivers for a single organization (by absolute value)."""
    row = shap_values[idx]
    sorted_idx = np.argsort(np.abs(row))[::-1][:top_n]
    return [
        {
            "feature": BDI_FEATURE_COLUMNS[i],
            "label": FEATURE_LABELS.get(BDI_FEATURE_COLUMNS[i], BDI_FEATURE_COLUMNS[i]),
            "shap_value": round(float(row[i]), 4),
            "direction": "increases" if row[i] > 0 else "decreases",
        }
        for i in sorted_idx
    ]


def run_shap_analysis(
    df: pd.DataFrame, shap_sample_n: int = 5000
) -> tuple[np.ndarray, object]:
    X = df[BDI_FEATURE_COLUMNS]
    explainer, scaler, clf = load_model_and_explainer(X)

    # Subsample for global plots — SHAP on 300K+ rows takes hours
    if len(X) > shap_sample_n:
        X_sample = X.sample(n=shap_sample_n, random_state=42)
        print(f"  SHAP: subsampling {shap_sample_n:,} of {len(X):,} rows for global plots")
    else:
        X_sample = X

    shap_values = compute_shap_values(explainer, scaler, X_sample)
    domain_df = compute_shap_domain_contributions(shap_values, X_sample)
    plot_shap_summary(shap_values, X_sample)
    plot_shap_bar(shap_values, X_sample)
    plot_shap_domain_contributions(domain_df)

    # Return full-dataset SHAP values for per-org BEAM reports
    shap_values_full = compute_shap_values(explainer, scaler, X)
    return shap_values_full, explainer

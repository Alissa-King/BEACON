"""
BEACON result visualizations.
Includes calibration reliability diagram (Brier-score companion plot)
added per committee recommendation: a good classifier that produces
miscalibrated probabilities is not suitable as a risk-scoring tool.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split

from src.features.bdi import BDI_FEATURE_COLUMNS
from src.models.train import TEST_YEARS, CALIBRATION_YEARS

FIGURES_DIR = Path("reports/figures")
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (uncalibrated)",
}
COLORS = {
    "logistic_regression": "#e07b54",
    "random_forest": "#5b9bd5",
    "xgboost": "#2e8b57",
}


def _test_split(df: pd.DataFrame):
    mask = df["fiscal_year"].isin(TEST_YEARS)
    X = df.loc[mask, BDI_FEATURE_COLUMNS]
    y = df.loc[mask, "financial_distress"].astype(int)
    return X, y


def plot_roc_curves(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = _test_split(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance (AUC = 0.50)")

    for key, label in MODEL_NAMES.items():
        pipeline = joblib.load(Path("models") / f"{key}.pkl")
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=COLORS[key], label=f"{label} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC Curves — BEACON Predictive Models\n(test set: FY2022–2023)", fontsize=12
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {path}")


def plot_precision_recall(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = _test_split(df)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (prevalence = {baseline:.2f})")

    for key, label in MODEL_NAMES.items():
        pipeline = joblib.load(Path("models") / f"{key}.pkl")
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, color=COLORS[key], label=f"{label} (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curves — BEACON Predictive Models\n(test set: FY2022–2023)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "precision_recall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved precision-recall curves to {path}")


def plot_calibration_curve(df: pd.DataFrame) -> None:
    """
    Reliability diagram comparing raw Random Forest probabilities to isotonic-
    calibrated probabilities against observed distress rates.

    A well-calibrated model's curve tracks the diagonal; systematic
    deviations indicate over/under-confidence in probability estimates.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cal_mask = df["fiscal_year"].isin(CALIBRATION_YEARS)
    X_cal = df.loc[cal_mask, BDI_FEATURE_COLUMNS]
    y_cal = df.loc[cal_mask, "financial_distress"].astype(int)

    pipeline = joblib.load(Path("models/random_forest.pkl"))
    calibrator = joblib.load(Path("models/random_forest_calibrator.pkl"))

    raw_probs = pipeline.predict_proba(X_cal)[:, 1]
    cal_probs = calibrator.transform(raw_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for probs, label, color in [
        (raw_probs, "Random Forest (uncalibrated)", "#e07b54"),
        (cal_probs, "Random Forest + Isotonic Regression", "#2e8b57"),
    ]:
        fraction_pos, mean_pred = calibration_curve(y_cal, probs, n_bins=10)
        ax.plot(mean_pred, fraction_pos, "s-", lw=2, color=color, label=label)

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Distress Rate", fontsize=11)
    ax.set_title(
        "Reliability Diagram (Calibration Curve)\n"
        "Random Forest before and after isotonic calibration",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    path = FIGURES_DIR / "calibration_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration curve to {path}")


def plot_bdi_distribution(df: pd.DataFrame) -> None:
    if "bdi_score" not in df.columns:
        print("bdi_score not found; skipping BDI distribution plot.")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 100, 41)
    ax.hist(df["bdi_score"], bins=bins, edgecolor="white", linewidth=0.4, color="#5b9bd5")

    for threshold, color in [(40, "#2e8b57"), (60, "#f0c040"), (80, "#c0392b")]:
        ax.axvline(threshold, color=color, lw=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("BDI Score  (0 = Low Risk → 100 = Severe Risk)", fontsize=11)
    ax.set_ylabel("Number of Organization-Year Observations", fontsize=11)
    ax.set_title(
        "Distribution of BEACON Distress Index (BDI) Scores\n"
        "BDI = 100 × calibrated P(distress)",
        fontsize=12,
    )
    patches = [
        mpatches.Patch(color="#2e8b57", label="Low Risk (0–39)"),
        mpatches.Patch(color="#f0c040", label="Moderate Risk (40–59)"),
        mpatches.Patch(color="#e07b54", label="Elevated Risk (60–79)"),
        mpatches.Patch(color="#c0392b", label="Severe Risk (80–100)"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="upper right")
    plt.tight_layout()
    path = FIGURES_DIR / "bdi_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved BDI distribution to {path}")


def plot_bdi_distress_rates(df: pd.DataFrame) -> None:
    if "bdi_category" not in df.columns:
        print("bdi_category not found; skipping distress rate plot.")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    order = ["Low Risk", "Moderate Risk", "Elevated Risk", "Severe Risk"]
    colors = ["#2e8b57", "#f0c040", "#e07b54", "#c0392b"]
    rates = (
        df.groupby("bdi_category", observed=True)["financial_distress"]
        .mean()
        .reindex(order)
        * 100
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(order, rates, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Observed Distress Rate (%)", fontsize=11)
    ax.set_xlabel("BDI Risk Category", fontsize=11)
    ax.set_title("Predictive Validity: Observed Distress Rate by BDI Category", fontsize=12)
    ax.set_ylim(0, rates.max() * 1.25)
    plt.tight_layout()
    path = FIGURES_DIR / "bdi_distress_rates.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved BDI distress rates to {path}")

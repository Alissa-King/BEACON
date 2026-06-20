"""
BEACON Distress Index (BDI) — defense-ready definition.

The BDI is a calibrated transformation of the Random Forest model's predicted
probability of financial distress:

    BDI = 100 × P(Distress | X)_calibrated

where P(Distress) is produced by an isotonic-regression calibrator fitted
on a held-out calibration set (fiscal years 2020-2021). This ensures
probabilities reflect observed distress frequencies rather than raw
classifier scores.

The four BEACON domains (Financial Capacity, Financial Sustainability,
Resource Dependence, Organizational Risk) are NOT inputs to the BDI
formula. They are an explanatory decomposition layer applied post-hoc
via SHAP value aggregation to make risk drivers interpretable for
governance audiences.

Feature list and domain assignments are defined in variable_dictionary.py
and imported here — that module is the single source of truth for
all variable metadata used in BEACON.

BDI interpretation scale:
  80–100  Severe Risk   — high likelihood of distress event within 24 months
  60–79   Elevated Risk — monitor and initiate contingency planning
  40–59   Moderate Risk — early-warning; review primary risk drivers
  0–39    Low Risk      — stable; maintain current trajectory

Limitation: BDI is validated through predictive performance against
observed distress outcomes (calibration, AUC-ROC). It is an operational
risk score, not a latent financial construct.
"""

import numpy as np
import pandas as pd

from src.features.variable_dictionary import PREDICTOR_NAMES, DOMAIN_MAP, FEATURE_LABELS


# Feature columns fed into the ML models — order matters for SHAP alignment
BDI_FEATURE_COLUMNS: list[str] = PREDICTOR_NAMES

# Domain groupings for SHAP explanatory decomposition (not BDI formula inputs)
SHAP_DOMAIN_MAP: dict[str, list[str]] = DOMAIN_MAP

# Risk category thresholds — HIGH BDI = HIGH DISTRESS RISK
_BINS = [-np.inf, 40, 60, 80, np.inf]
_LABELS = ["Low Risk", "Moderate Risk", "Elevated Risk", "Severe Risk"]

# Colour aliases used in BEAM trigger logic (high risk = Red)
CATEGORY_TO_COLOUR = {
    "Low Risk": "Green",
    "Moderate Risk": "Yellow",
    "Elevated Risk": "Orange",
    "Severe Risk": "Red",
}


def compute_bdi(df: pd.DataFrame, calibrated_probs: np.ndarray) -> pd.DataFrame:
    """
    Attach BDI scores and risk categories to the dataframe.

    Parameters
    ----------
    df : cleaned panel dataframe containing BDI_FEATURE_COLUMNS
    calibrated_probs : 1-D array of calibrated P(distress) in [0, 1],
                       one value per row of df, produced by the isotonic
                       regression calibrator

    Returns
    -------
    df with new columns:
        bdi_score      float  0–100
        bdi_category   str    Severe / Elevated / Moderate / Low Risk
        bdi_colour     str    Red / Orange / Yellow / Green
    """
    if len(calibrated_probs) != len(df):
        raise ValueError(
            f"calibrated_probs length ({len(calibrated_probs)}) "
            f"must match dataframe rows ({len(df)})"
        )

    df = df.copy()
    df["bdi_score"] = np.clip(calibrated_probs * 100, 0, 100)
    df["bdi_category"] = pd.cut(
        df["bdi_score"], bins=_BINS, labels=_LABELS
    ).astype(str)
    df["bdi_colour"] = df["bdi_category"].map(CATEGORY_TO_COLOUR)
    return df

"""
Probability calibration for the BEACON XGBoost model.

Uses isotonic regression (Platt scaling alternative) to ensure predicted
distress probabilities reflect observed distress frequencies. An
uncalibrated classifier may rank organizations correctly (high AUC) but
produce unreliable probability estimates — problematic for a risk-scoring
system used in governance decisions.

Calibration is fitted on a held-out calibration set (fiscal years
2020-2021) that is neither part of training nor the final test set.

Metrics added:
  Brier Score   — mean squared error of probability estimates; lower = better
  Reliability   — measured via calibration curve (see visualize.py)

See: Guo et al. (2017), "On Calibration of Modern Neural Networks";
     Platt (1999), "Probabilistic Outputs for Support Vector Machines"
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def fit_calibrator(raw_probs: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator on calibration-set probabilities.

    Parameters
    ----------
    raw_probs : uncalibrated P(distress) from the base model [0, 1]
    y_true    : observed binary distress labels

    Returns
    -------
    Fitted IsotonicRegression object
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_probs, y_true)
    return cal


def calibrate(calibrator: IsotonicRegression, raw_probs: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to new probability estimates."""
    return calibrator.transform(raw_probs)


def brier_score(y_true: np.ndarray, calibrated_probs: np.ndarray) -> float:
    """
    Compute Brier score (lower = better probabilistic calibration).
    Perfect calibration = 0.0; random = 0.25 for balanced binary outcome.
    """
    return round(float(brier_score_loss(y_true, calibrated_probs)), 4)

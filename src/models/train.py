"""
BEACON predictive model training — temporal validation design.

Validation strategy (Section 3.3):
  TRAIN       fiscal years 2013–2019  (earlier years only)
  CALIBRATION fiscal years 2020–2021  (held out for isotonic calibration)
  TEST        fiscal years 2022–2023  (final holdout; never seen during training)

Preprocessing protocol (no leakage):
  CleaningPipeline is fit on TRAIN years only, then applied to CAL and TEST.
  Winsorization quantiles and KNN imputer statistics are frozen to the
  training distribution before any calibration or test data is touched.

Within-training CV uses custom fiscal-year expanding folds, NOT TimeSeriesSplit
on row indices.  The training dataframe is sorted by (ein, fiscal_year), so
row-index splits would not respect global chronological order across organizations.

  Fold 1: train 2013–2014, validate 2015
  Fold 2: train 2013–2015, validate 2016
  Fold 3: train 2013–2016, validate 2017
  Fold 4: train 2013–2017, validate 2018
  Fold 5: train 2013–2018, validate 2019

Preprocessing is re-fit within each fold on that fold's training years only.

Three models evaluated (Section 3.6):
  Logistic Regression   baseline / interpretable reference
  Random Forest         primary model (prespecified; calibrated; SHAP-explained)
  XGBoost               secondary nonlinear benchmark

Random Forest was prespecified as the primary BDI model before holdout
evaluation, selected for interpretability, calibration stability, and
compatibility with SHAP TreeExplainer.  XGBoost achieves marginally higher
holdout AUC (0.734 vs. 0.731) but was not prespecified as primary and is
reported as a robustness benchmark.

Metrics (Section 3.7):
  AUC-ROC            primary discrimination metric
  Precision-Recall   important under class imbalance
  F1 Score           harmonic mean of precision and recall
  Brier Score        calibration quality (added per committee recommendation)
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.bdi import BDI_FEATURE_COLUMNS
from src.ingestion.cleaning_pipeline import CleaningPipeline
from src.models.calibration import brier_score, calibrate, fit_calibrator

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
RANDOM_STATE = 42
MIN_TRAIN_YEARS = 2  # minimum training years before first CV validation fold

TRAIN_YEARS = list(range(2013, 2020))   # 2013–2019
CALIBRATION_YEARS = [2020, 2021]
TEST_YEARS = [2022, 2023]


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Split panel by fiscal year to preserve temporal ordering.
    Returns (df_train, df_cal, df_test) — full rows, preprocessing not yet applied.
    """
    train_mask = df["fiscal_year"].isin(TRAIN_YEARS)
    cal_mask   = df["fiscal_year"].isin(CALIBRATION_YEARS)
    test_mask  = df["fiscal_year"].isin(TEST_YEARS)

    df_train = df[train_mask].copy()
    df_cal   = df[cal_mask].copy()
    df_test  = df[test_mask].copy()

    print(
        f"  Train  ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)}): "
        f"{len(df_train):,} obs, distress rate {df_train['financial_distress'].mean():.1%}\n"
        f"  Cal    ({min(CALIBRATION_YEARS)}–{max(CALIBRATION_YEARS)}): "
        f"{len(df_cal):,} obs, distress rate {df_cal['financial_distress'].mean():.1%}\n"
        f"  Test   ({min(TEST_YEARS)}–{max(TEST_YEARS)}): "
        f"{len(df_test):,} obs, distress rate {df_test['financial_distress'].mean():.1%}"
    )
    return df_train, df_cal, df_test


# ── Model definitions ─────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                C=0.5,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ]),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    calibrated_prob: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":       round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
    }
    if calibrated_prob is not None:
        metrics["brier_score"]     = brier_score(y_true, calibrated_prob)
        metrics["brier_score_raw"] = brier_score(y_true, y_prob)
    return metrics


# ── True temporal CV (fiscal-year expanding folds) ────────────────────────────

def fiscal_year_cv_folds(
    df_train: pd.DataFrame,
    min_train_years: int = MIN_TRAIN_YEARS,
) -> list[tuple[np.ndarray, np.ndarray, list[int], int]]:
    """
    Build expanding-window CV fold index tuples from fiscal year membership.

    Each fold uses all years up to (but not including) the validation year as
    training, and exactly one fiscal year as validation.  The dataframe index
    is used so iloc lookups remain correct even after reset_index.

    Returns list of (train_idx, val_idx, train_years, val_year).
    """
    years = sorted(df_train["fiscal_year"].unique())
    folds = []
    for i in range(min_train_years, len(years)):
        val_year   = years[i]
        train_yrs  = years[:i]
        train_mask = df_train["fiscal_year"].isin(train_yrs).values
        val_mask   = (df_train["fiscal_year"] == val_year).values
        train_idx  = np.where(train_mask)[0]
        val_idx    = np.where(val_mask)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx, train_yrs, val_year))
    return folds


def time_series_cv(pipeline, df_train_raw: pd.DataFrame) -> dict:
    """
    True expanding-window temporal CV by fiscal year.

    Preprocessing (CleaningPipeline) is re-fit within each fold on that
    fold's training years only, then applied to the validation year.
    This prevents winsorization/imputation statistics from the validation
    year from leaking into fold-level training estimates.
    """
    folds = fiscal_year_cv_folds(df_train_raw)
    if not folds:
        return {}

    fold_metrics = []
    for train_idx, val_idx, train_yrs, val_year in folds:
        df_fold_tr  = df_train_raw.iloc[train_idx].copy()
        df_fold_val = df_train_raw.iloc[val_idx].copy()

        # Fit preprocessing on this fold's training years only
        fold_cleaner = CleaningPipeline()
        df_fold_tr_clean  = fold_cleaner.fit_transform(df_fold_tr)
        df_fold_val_clean = fold_cleaner.transform(df_fold_val)

        X_tr  = df_fold_tr_clean[BDI_FEATURE_COLUMNS]
        y_tr  = df_fold_tr_clean["financial_distress"].astype(int)
        X_val = df_fold_val_clean[BDI_FEATURE_COLUMNS]
        y_val = df_fold_val_clean["financial_distress"].astype(int)

        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            continue

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        fold_metrics.append(evaluate(y_val.values, y_pred, y_prob))

    if not fold_metrics:
        return {}

    df_cv = pd.DataFrame(fold_metrics)
    return {
        col: {"mean": round(df_cv[col].mean(), 4), "std": round(df_cv[col].std(), 4)}
        for col in df_cv.columns
    }


# ── Robustness checks ─────────────────────────────────────────────────────────

def robustness_checks(
    pipeline,
    df_test: pd.DataFrame,
    y_test: pd.Series,
    df_cal: pd.DataFrame | None = None,
    y_cal: pd.Series | None = None,
) -> dict:
    """
    Section 3.11.3: subgroup and recession-period sensitivity.

    NTEE subgroup analysis runs on the test set (FY2022–2023).
    Recession-period sensitivity (COVID-19 FY2020–2021) runs on the
    calibration set, which covers those years — not the test set.
    """
    results = {}

    # NTEE subgroup analysis on test holdout
    for code in ["L", "P"]:
        mask = df_test["ntee_code"] == code
        if mask.sum() < 10:
            continue
        X_sub = df_test.loc[mask, BDI_FEATURE_COLUMNS]
        y_sub = y_test.loc[mask]
        if y_sub.nunique() < 2:
            continue
        y_prob = pipeline.predict_proba(X_sub)[:, 1]
        y_pred = pipeline.predict(X_sub)
        results[f"ntee_{code}"] = evaluate(y_sub.values, y_pred, y_prob)

    # Recession-period sensitivity on calibration set (FY2020–2021)
    if df_cal is not None and y_cal is not None and len(df_cal) > 10:
        if y_cal.nunique() >= 2:
            X_rec  = df_cal[BDI_FEATURE_COLUMNS]
            y_prob = pipeline.predict_proba(X_rec)[:, 1]
            y_pred = pipeline.predict(X_rec)
            results["recession_period_2020_21"] = evaluate(y_cal.values, y_pred, y_prob)

    return results


# ── Main training function ────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, CleaningPipeline]:
    """
    Train and evaluate all models with strict temporal preprocessing protocol.

    Parameters
    ----------
    df : pd.DataFrame
        Panel after apply_exclusion_criteria() and align_fiscal_years() but
        BEFORE winsorization or imputation.  Temporal split and preprocessing
        are handled here to prevent leakage.

    Returns
    -------
    results : dict
        Nested metrics dict (CV, holdout, robustness) per model.
    cleaning : CleaningPipeline
        Fitted preprocessing object (also saved to models/cleaning_pipeline.pkl).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Temporal split on raw (unprocessed) data ───────────────────────────
    print("Temporal split:")
    df_train_raw, df_cal_raw, df_test_raw = temporal_split(df)

    # ── 2. Fit preprocessing on training years only ───────────────────────────
    print("\nFitting CleaningPipeline on training years (FY2013–2019) only…")
    cleaning = CleaningPipeline()
    df_train = cleaning.fit_transform(df_train_raw)
    df_cal   = cleaning.transform(df_cal_raw)
    df_test  = cleaning.transform(df_test_raw)
    joblib.dump(cleaning, MODEL_DIR / "cleaning_pipeline.pkl")
    print("  Preprocessing fitted and saved to models/cleaning_pipeline.pkl")

    X_train = df_train[BDI_FEATURE_COLUMNS]
    y_train = df_train["financial_distress"].astype(int)
    X_cal   = df_cal[BDI_FEATURE_COLUMNS]
    y_cal   = df_cal["financial_distress"].astype(int)
    X_test  = df_test[BDI_FEATURE_COLUMNS]
    y_test  = df_test["financial_distress"].astype(int)

    pipelines = build_pipelines()
    results   = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}…")

        # True temporal CV within training window (fiscal-year expanding folds)
        cv_scores = time_series_cv(pipeline, df_train_raw)

        # Final fit on full training window (using training-preprocessed data)
        pipeline.fit(X_train, y_train)

        # Calibrate Random Forest only (prespecified primary model)
        calibrator = None
        if name == "random_forest":
            raw_cal_probs = pipeline.predict_proba(X_cal)[:, 1]
            calibrator = fit_calibrator(raw_cal_probs, y_cal.values)
            joblib.dump(calibrator, MODEL_DIR / "random_forest_calibrator.pkl")
            print("  Calibrated with isotonic regression on FY2020–2021")

        # Evaluate on time-based holdout test set
        y_prob_raw = pipeline.predict_proba(X_test)[:, 1]
        y_pred     = pipeline.predict(X_test)
        cal_probs  = calibrate(calibrator, y_prob_raw) if calibrator else None
        holdout    = evaluate(y_test.values, y_pred, y_prob_raw, cal_probs)

        # Robustness: NTEE subgroups on test set; recession period on calibration set
        robustness = robustness_checks(pipeline, df_test, y_test, df_cal, y_cal)

        results[name] = {
            "cv_temporal":  cv_scores,
            "holdout_test": holdout,
            "robustness":   robustness,
        }

        print(
            f"  Holdout → Acc: {holdout['accuracy']:.3f} | "
            f"Prec: {holdout['precision']:.3f} | "
            f"Rec: {holdout['recall']:.3f} | "
            f"F1: {holdout['f1']:.3f} | "
            f"AUC: {holdout['auc_roc']:.3f} | "
            f"AvgP: {holdout['avg_precision']:.3f}"
            + (f" | Brier: {holdout['brier_score']:.4f}" if calibrator else "")
        )

        joblib.dump(pipeline, MODEL_DIR / f"{name}.pkl")

    with open(MODEL_DIR / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation report saved to models/evaluation_report.json")

    return results, cleaning


if __name__ == "__main__":
    from src.ingestion.synthetic_data import generate_synthetic_990
    from src.ingestion.cleaning_pipeline import apply_exclusion_criteria, align_fiscal_years

    df_raw      = generate_synthetic_990(n_orgs=2000)
    df_prepared = align_fiscal_years(apply_exclusion_criteria(df_raw))
    results, _  = train_and_evaluate(df_prepared)

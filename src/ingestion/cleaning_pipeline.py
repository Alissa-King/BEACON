"""
BEACON data cleaning pipeline (Appendix C).

Handles:
  - Exclusion of EZ/N filers outside NTEE L/P
  - Outlier winsorization at 1st/99th percentiles
  - KNN imputation for missing continuous variables
  - Longitudinal alignment

IMPORTANT — fit preprocessing on training data only.
Use CleaningPipeline.fit() on the training temporal split, then .transform()
on each subsequent split to prevent calibration/test distributions from
contaminating training-time statistics (winsorization quantiles, imputer neighbors).
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


CONTINUOUS_FEATURES = [
    "months_cash_on_hand",
    "current_ratio",
    "unrestricted_net_assets_ratio",
    "operating_margin",
    "gov_grant_concentration",
    "revenue_hhi",
    "debt_to_equity",
    "consecutive_deficits",
    "program_expense_ratio",
    "total_revenue_log",
]

WINSORIZE_BOUNDS = (0.01, 0.99)


class CleaningPipeline:
    """
    Stateful preprocessing object.  Fits winsorization quantiles and KNN
    imputer on the training split only, then applies the frozen transforms
    to calibration and test splits.

    Usage:
        cleaner = CleaningPipeline()
        df_train_clean = cleaner.fit_transform(df_train_raw)
        df_cal_clean   = cleaner.transform(df_cal_raw)
        df_test_clean  = cleaner.transform(df_test_raw)
        joblib.dump(cleaner, "models/cleaning_pipeline.pkl")

    The saved object must accompany the trained models so that new data
    receives identical winsorization bounds and imputer statistics.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.winsorize_bounds_: dict[str, tuple[float, float]] = {}
        self.imputer_: KNNImputer | None = None
        self.imputer_cols_: list[str] = []
        self._is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "CleaningPipeline":
        """Compute winsorization quantiles and fit KNN imputer on training data only."""
        for col in CONTINUOUS_FEATURES:
            if col in df_train.columns:
                self.winsorize_bounds_[col] = (
                    float(df_train[col].quantile(WINSORIZE_BOUNDS[0])),
                    float(df_train[col].quantile(WINSORIZE_BOUNDS[1])),
                )
        self.imputer_cols_ = [c for c in CONTINUOUS_FEATURES if c in df_train.columns]
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer_.fit(df_train[self.imputer_cols_])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply training-fitted winsorization bounds and imputer to any split."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() on training data before transform().")
        df = df.copy()
        for col, (lo, hi) in self.winsorize_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        # Use the same column list the imputer was fitted on
        cols = [c for c in self.imputer_cols_ if c in df.columns]
        if df[cols].isnull().any().any():
            imputed = pd.DataFrame(
                self.imputer_.transform(df[cols]),
                index=df.index,
                columns=cols,
            )
            df.update(imputed)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def apply_exclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to NTEE L/P only (metadata filter — no feature statistics computed)."""
    return df[df["ntee_code"].isin(["L", "P"])].copy()


def align_fiscal_years(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure fiscal_year is integer and records are sorted by (ein, fiscal_year)."""
    df = df.copy()
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    return df.sort_values(["ein", "fiscal_year"]).reset_index(drop=True)


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper that applies all cleaning steps to a single dataframe.

    Suitable for dashboard auto-training on synthetic data and standalone scoring.
    For model training workflows, use CleaningPipeline.fit() on the training
    temporal split only, then .transform() each split separately — this function
    fits preprocessing statistics on the entire dataframe it receives.
    """
    df = apply_exclusion_criteria(df)
    df = align_fiscal_years(df)
    cleaner = CleaningPipeline()
    return cleaner.fit_transform(df)

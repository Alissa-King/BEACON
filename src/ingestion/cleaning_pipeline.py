"""
BEACON data cleaning pipeline (Appendix C).

Handles:
  - Exclusion of EZ/N filers
  - Outlier winsorization at 1st/99th percentiles
  - KNN imputation for missing continuous variables
  - Longitudinal alignment
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


CONTINUOUS_FEATURES = [
    "months_cash_on_hand",
    "current_ratio",
    "unrestricted_net_assets_ratio",
    "operating_margin",
    "gov_grant_concentration",
    "revenue_hhi",
    "debt_to_equity",
    "consecutive_deficits",
]

WINSORIZE_BOUNDS = (0.01, 0.99)


def winsorize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            lo = df[col].quantile(WINSORIZE_BOUNDS[0])
            hi = df[col].quantile(WINSORIZE_BOUNDS[1])
            df[col] = df[col].clip(lo, hi)
    return df


def impute_missing(df: pd.DataFrame, columns: list[str], n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    subset = df[columns].copy()
    if subset.isnull().any().any():
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[columns] = imputer.fit_transform(subset)
    return df


def apply_exclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records outside NTEE L/P (already filtered in synthetic, enforced here for real data)."""
    return df[df["ntee_code"].isin(["L", "P"])].copy()


def align_fiscal_years(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure fiscal_year is integer and records are sorted."""
    df = df.copy()
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    return df.sort_values(["ein", "fiscal_year"]).reset_index(drop=True)


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_exclusion_criteria(df)
    df = align_fiscal_years(df)
    df = winsorize(df, CONTINUOUS_FEATURES)
    df = impute_missing(df, CONTINUOUS_FEATURES)
    return df

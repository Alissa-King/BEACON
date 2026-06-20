"""
BEACON Distress Index (BDI) calculation (Appendix A).

BDI = sum(W_i * D_i)  where D_i is a standardized 0-100 domain score.

Domains and weights:
  Financial Capacity      (W=0.30): months_cash_on_hand, current_ratio,
                                    unrestricted_net_assets_ratio
  Financial Sustainability(W=0.30): operating_margin, consecutive_deficits
  Resource Dependence     (W=0.20): gov_grant_concentration, revenue_hhi
  Organizational Risk     (W=0.20): debt_to_equity
"""

import numpy as np
import pandas as pd


DOMAIN_WEIGHTS = {
    "financial_capacity": 0.30,
    "financial_sustainability": 0.30,
    "resource_dependence": 0.20,
    "organizational_risk": 0.20,
}

# Variables where HIGHER value = BETTER resilience (score goes up)
POSITIVE_INDICATORS = {
    "months_cash_on_hand",
    "current_ratio",
    "unrestricted_net_assets_ratio",
    "operating_margin",
}

# Variables where HIGHER value = WORSE resilience (score is inverted)
NEGATIVE_INDICATORS = {
    "consecutive_deficits",
    "gov_grant_concentration",
    "revenue_hhi",
    "debt_to_equity",
}

# Clip bounds (5th/95th percentiles per dissertation)
PERCENTILE_CLIP = (5, 95)


def _minmax_scale(series: pd.Series, clip_pct: tuple = PERCENTILE_CLIP) -> pd.Series:
    """Scale series to 0-100 after clipping at given percentiles."""
    lo = np.percentile(series.dropna(), clip_pct[0])
    hi = np.percentile(series.dropna(), clip_pct[1])
    clipped = series.clip(lo, hi)
    if hi == lo:
        return pd.Series(50.0, index=series.index)
    return (clipped - lo) / (hi - lo) * 100


def _domain_score(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    """Average standardized score across domain variables, inverting negative indicators."""
    scores = []
    for var in variables:
        if var not in df.columns:
            continue
        scaled = _minmax_scale(df[var])
        if var in NEGATIVE_INDICATORS:
            scaled = 100 - scaled
        scores.append(scaled)
    if not scores:
        return pd.Series(50.0, index=df.index)
    return pd.concat(scores, axis=1).mean(axis=1)


def compute_bdi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-domain scores and the final BDI score to the dataframe.
    Returns the input dataframe with new columns added.
    """
    df = df.copy()

    df["d_financial_capacity"] = _domain_score(
        df, ["months_cash_on_hand", "current_ratio", "unrestricted_net_assets_ratio"]
    )
    df["d_financial_sustainability"] = _domain_score(
        df, ["operating_margin", "consecutive_deficits"]
    )
    df["d_resource_dependence"] = _domain_score(
        df, ["gov_grant_concentration", "revenue_hhi"]
    )
    df["d_organizational_risk"] = _domain_score(
        df, ["debt_to_equity"]
    )

    df["bdi_score"] = (
        DOMAIN_WEIGHTS["financial_capacity"] * df["d_financial_capacity"]
        + DOMAIN_WEIGHTS["financial_sustainability"] * df["d_financial_sustainability"]
        + DOMAIN_WEIGHTS["resource_dependence"] * df["d_resource_dependence"]
        + DOMAIN_WEIGHTS["organizational_risk"] * df["d_organizational_risk"]
    )

    df["bdi_category"] = pd.cut(
        df["bdi_score"],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["Red", "Orange", "Yellow", "Green"],
    )

    return df


BDI_FEATURE_COLUMNS = [
    "months_cash_on_hand",
    "current_ratio",
    "unrestricted_net_assets_ratio",
    "operating_margin",
    "consecutive_deficits",
    "gov_grant_concentration",
    "revenue_hhi",
    "debt_to_equity",
]

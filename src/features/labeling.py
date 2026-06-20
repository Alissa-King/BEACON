"""
Forward-looking financial distress label computation (Options 1 and 2).

CRITICAL DESIGN NOTE — preventing self-prediction:
  All features are backward-looking (measured at or before year T).
  All labels are forward-looking (based on what happens at T+1 and T+2).
  This means consecutive_deficits at T counts past deficits ending at T,
  while financial_distress at T asks whether distress BEGINS in T+1 or T+2.
  There is no overlap between the feature and the label.

Option 1 — Sustained Operating Deficit (primary label):
  distress_label1[T] = 1 if operating_margin < 0 in BOTH T+1 AND T+2
  Precedent: Greenlee & Trussel (2000). Computable from total revenue and
  total expenses — fields reliably returned by ProPublica and NCCS.
  Limitation: intentional deficits (e.g., spending down a restricted grant)
  may produce false positives. Label captures financial stress, not failure.

Option 2 — Net Asset Insolvency (robustness label):
  distress_label2[T] = 1 if unrestricted_net_assets < 0 in T+1 OR T+2
  Closer to genuine solvency risk. Rarer → heavier class imbalance.
  Requires unrestricted net asset field (available in NCCS Core, partial
  in ProPublica). Report alongside Option 1 as a robustness check.

Rows without two observable future years are dropped (right-censoring).
The caller is responsible for re-fitting the cleaning/imputation pipeline
after label computation, since the labeled dataset may differ from the
pre-label dataset.
"""

import numpy as np
import pandas as pd


PREDICTION_HORIZON = 2  # years ahead


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking Option 1 and Option 2 distress labels.

    Requires columns: ein, fiscal_year, operating_margin,
                      unrestricted_net_assets (for Option 2; optional).

    Returns dataframe with new columns:
        financial_distress     — Option 1 label (primary)
        financial_distress_2   — Option 2 label (robustness); NaN if
                                 unrestricted_net_assets unavailable
        label_year_t1          — fiscal_year + 1 operating_margin (audit trail)
        label_year_t2          — fiscal_year + 2 operating_margin (audit trail)

    Rows where T+1 or T+2 is not observable are dropped.
    """
    df = df.copy().sort_values(["ein", "fiscal_year"]).reset_index(drop=True)

    # Build a lookup: (ein, year) → operating_margin and unrestricted_net_assets
    margin_lookup = df.set_index(["ein", "fiscal_year"])["operating_margin"].to_dict()
    has_una = "unrestricted_net_assets" in df.columns
    if has_una:
        una_lookup = df.set_index(["ein", "fiscal_year"])["unrestricted_net_assets"].to_dict()

    labels1, labels2, m_t1, m_t2 = [], [], [], []

    for _, row in df.iterrows():
        ein, yr = row["ein"], int(row["fiscal_year"])
        margin1 = margin_lookup.get((ein, yr + 1), np.nan)
        margin2 = margin_lookup.get((ein, yr + 2), np.nan)

        # Drop row if either future year is unobservable
        if np.isnan(margin1) or np.isnan(margin2):
            labels1.append(np.nan)
            labels2.append(np.nan)
            m_t1.append(np.nan)
            m_t2.append(np.nan)
            continue

        # Option 1: both T+1 and T+2 have negative operating margin
        labels1.append(int(margin1 < 0 and margin2 < 0))
        m_t1.append(margin1)
        m_t2.append(margin2)

        # Option 2: unrestricted net assets go negative in T+1 or T+2
        if has_una:
            una1 = una_lookup.get((ein, yr + 1), np.nan)
            una2 = una_lookup.get((ein, yr + 2), np.nan)
            if np.isnan(una1) and np.isnan(una2):
                labels2.append(np.nan)
            else:
                neg1 = (not np.isnan(una1)) and (una1 < 0)
                neg2 = (not np.isnan(una2)) and (una2 < 0)
                labels2.append(int(neg1 or neg2))
        else:
            labels2.append(np.nan)

    df["financial_distress"] = labels1
    df["financial_distress_2"] = labels2
    df["label_year_t1"] = m_t1
    df["label_year_t2"] = m_t2

    # Drop rows with no label (last PREDICTION_HORIZON years per org)
    before = len(df)
    df = df.dropna(subset=["financial_distress"]).copy()
    df["financial_distress"] = df["financial_distress"].astype(int)
    after = len(df)

    print(
        f"  Labeling: {before - after:,} rows dropped (right-censored last "
        f"{PREDICTION_HORIZON} years per org)\n"
        f"  Labeled rows: {after:,} | "
        f"Option 1 distress rate: {df['financial_distress'].mean():.1%}"
    )
    if has_una and df["financial_distress_2"].notna().any():
        rate2 = df["financial_distress_2"].mean()
        print(f"  Option 2 distress rate: {rate2:.1%}")

    return df

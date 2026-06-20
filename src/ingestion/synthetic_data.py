"""
Synthetic IRS Form 990 data generator for BEACON model development.

Generates realistic nonprofit financial data mirroring NTEE Categories L and P,
based on the variable definitions in Appendix B. Use this until real NCCS/IRS data
is obtained.

Self-prediction fix: all raw financials are generated first (all years), then
forward-looking distress labels are applied in a second pass via compute_labels().
The consecutive_deficits feature at year T reflects deficits through year T;
the financial_distress label at year T is determined by what happens at T+1 and T+2.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.features.labeling import compute_labels


NTEE_CATEGORIES = ["L", "P"]
STATES = [
    "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
FISCAL_YEARS = list(range(2013, 2026))  # extends to 2025 so 2022-2023 can be labeled


def _generate_org_base(n_orgs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    orgs = pd.DataFrame({
        "ein": [f"{rng.integers(10_000_000, 99_999_999)}" for _ in range(n_orgs)],
        "org_name": [f"NP_Org_{i:05d}" for i in range(n_orgs)],
        "ntee_code": rng.choice(NTEE_CATEGORIES, n_orgs),
        "state": rng.choice(STATES, n_orgs),
        # Underlying true financial health: 0=healthy, 1=distressed
        "true_risk_profile": rng.uniform(0, 1, n_orgs),
    })
    return orgs


def generate_synthetic_990(n_orgs: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a longitudinal panel of synthetic Form 990 financial variables.

    Returns one row per (org, fiscal_year) with all BEACON feature columns
    and a binary 'financial_distress' label.

    Labels are forward-looking: financial_distress at year T is 1 if operating_margin
    is negative in BOTH T+1 and T+2 (Option 1 / sustained deficit). Rows for the
    last two years of each org are dropped (right-censored — no observable future).
    """
    rng = np.random.default_rng(seed)
    orgs = _generate_org_base(n_orgs, seed)

    records = []
    for _, org in orgs.iterrows():
        risk = org["true_risk_profile"]

        # Persistent org-level characteristics
        base_revenue = rng.lognormal(mean=13 + (1 - risk) * 2, sigma=1.2)
        base_gov_dep = np.clip(rng.beta(2 + 4 * risk, 2), 0.05, 0.99)

        deficit_streak = 0

        for year in FISCAL_YEARS:
            # Revenue with year-over-year shock
            shock = rng.normal(0, 0.08 + 0.12 * risk)
            total_revenue = base_revenue * (1 + shock)

            expense_ratio = rng.normal(0.96 + 0.08 * risk, 0.05)
            total_expenses = total_revenue * expense_ratio

            operating_margin = (total_revenue - total_expenses) / total_revenue
            if operating_margin < 0:
                deficit_streak += 1
            else:
                deficit_streak = 0

            # Cash: riskier orgs have less cash relative to expenses
            months_cash = np.clip(
                rng.lognormal(mean=1.2 - 1.5 * risk, sigma=0.6), 0.0, 30.0
            )

            # Balance sheet items
            total_liabilities = total_expenses * rng.uniform(0.1 + 0.3 * risk, 0.5 + 0.6 * risk)
            net_assets_hi = max(0.10, 0.8 * (1 - risk))
            total_net_assets = np.maximum(total_revenue * rng.uniform(0.05, net_assets_hi), 1)
            # High-risk orgs can have negative unrestricted net assets (accumulated deficits
            # exceed the unrestricted pool; reflected in post-ASU 2016-14 990 reporting)
            if risk > 0.65:
                una_frac = rng.uniform(-0.25, 0.65)
            else:
                una_frac = rng.uniform(0.30, 0.90)
            unrestricted_net_assets = total_net_assets * una_frac
            current_assets = total_expenses / 12 * (months_cash + rng.uniform(0, 2))
            current_liabilities = current_assets / np.clip(rng.lognormal(0.5, 0.4), 0.3, 10)
            current_ratio = current_assets / np.maximum(current_liabilities, 1)

            # Revenue concentration (HHI)
            gov_dep = np.clip(base_gov_dep + rng.normal(0, 0.05), 0.01, 0.99)
            n_streams = max(2, int(rng.normal(5, 2)))
            weights = rng.dirichlet(np.ones(n_streams) * (1 - risk) * 3)
            # Ensure gov grants take their share
            weights[0] = gov_dep
            tail_sum = weights[1:].sum()
            if tail_sum > 0:
                weights[1:] = (1 - gov_dep) * weights[1:] / tail_sum
            hhi = float(np.sum(weights ** 2))

            debt_to_equity = total_liabilities / np.maximum(total_net_assets, 1)
            unrestricted_net_assets_ratio = unrestricted_net_assets / np.maximum(
                total_net_assets, 1
            )

            # Program expense ratio: healthier orgs spend more on programs vs. admin
            # Low-risk orgs: ~75-90% program; high-risk orgs: ~50-75% (admin bloat or cuts)
            program_expense_ratio = np.clip(
                rng.normal(0.82 - 0.15 * risk, 0.08), 0.30, 0.98
            )

            # Organizational scale: log total revenue (larger orgs more resilient)
            total_revenue_log = float(np.log(np.maximum(total_revenue, 1)))

            records.append({
                "ein": org["ein"],
                "org_name": org["org_name"],
                "ntee_code": org["ntee_code"],
                "state": org["state"],
                "fiscal_year": year,
                # Form 990 derived variables (Appendix B / variable_dictionary.py)
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "months_cash_on_hand": months_cash,
                "current_ratio": current_ratio,
                "unrestricted_net_assets": unrestricted_net_assets,
                "unrestricted_net_assets_ratio": unrestricted_net_assets_ratio,
                "total_net_assets": total_net_assets,
                "total_liabilities": total_liabilities,
                "operating_margin": operating_margin,
                "consecutive_deficits": deficit_streak,  # backward-looking: deficits through T
                "gov_grant_concentration": gov_dep,
                "revenue_hhi": hhi,
                "debt_to_equity": debt_to_equity,
                "program_expense_ratio": program_expense_ratio,
                "total_revenue_log": total_revenue_log,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["ein", "fiscal_year"]).reset_index(drop=True)

    # Apply forward-looking labels: financial_distress at T based on T+1 and T+2.
    # This is the only place labels are assigned — no label information exists
    # in the feature generation loop above.
    df = compute_labels(df)

    return df


if __name__ == "__main__":
    out = Path("data/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_990(n_orgs=2000)
    df.to_csv(out / "synthetic_990_panel.csv", index=False)
    print(f"Generated {len(df):,} records for {df['ein'].nunique():,} organizations.")
    print(f"Distress rate: {df['financial_distress'].mean():.1%}")

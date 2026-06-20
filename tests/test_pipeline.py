"""
Unit and integration tests for the BEACON pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.synthetic_data import generate_synthetic_990
from src.ingestion.cleaning_pipeline import run_cleaning_pipeline
from src.features.bdi import compute_bdi, BDI_FEATURE_COLUMNS
from src.beam.action_matrix import get_beam_actions, format_beam_report


@pytest.fixture(scope="module")
def sample_df():
    df_raw = generate_synthetic_990(n_orgs=100, seed=0)
    df_clean = run_cleaning_pipeline(df_raw)
    return compute_bdi(df_clean)


class TestSyntheticData:
    def test_row_count(self):
        df = generate_synthetic_990(n_orgs=50, seed=1)
        assert len(df) == 50 * 11  # 11 fiscal years

    def test_required_columns(self):
        df = generate_synthetic_990(n_orgs=10, seed=2)
        for col in BDI_FEATURE_COLUMNS + ["financial_distress", "ein", "ntee_code"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_ntee_categories(self):
        df = generate_synthetic_990(n_orgs=100, seed=3)
        assert set(df["ntee_code"].unique()).issubset({"L", "P"})

    def test_distress_rate_reasonable(self):
        df = generate_synthetic_990(n_orgs=500, seed=4)
        rate = df["financial_distress"].mean()
        assert 0.05 < rate < 0.65, f"Distress rate out of expected range: {rate:.2f}"


class TestCleaningPipeline:
    def test_no_nulls_in_features(self):
        df_raw = generate_synthetic_990(n_orgs=50, seed=5)
        df_clean = run_cleaning_pipeline(df_raw)
        assert df_clean[BDI_FEATURE_COLUMNS].isnull().sum().sum() == 0

    def test_winsorization_bounds(self):
        df_raw = generate_synthetic_990(n_orgs=100, seed=6)
        df_clean = run_cleaning_pipeline(df_raw)
        for col in ["months_cash_on_hand", "current_ratio", "debt_to_equity"]:
            original = df_raw[col]
            cleaned = df_clean[col]
            assert cleaned.max() <= original.quantile(0.995)


class TestBDI:
    def test_bdi_range(self, sample_df):
        assert sample_df["bdi_score"].between(0, 100).all(), \
            "BDI scores outside 0-100 range"

    def test_bdi_categories_present(self, sample_df):
        cats = set(sample_df["bdi_category"].astype(str))
        assert cats.issubset({"Green", "Yellow", "Orange", "Red"})

    def test_domain_scores_present(self, sample_df):
        for col in ["d_financial_capacity", "d_financial_sustainability",
                    "d_resource_dependence", "d_organizational_risk"]:
            assert col in sample_df.columns

    def test_weights_sum_to_one(self):
        from src.features.bdi import DOMAIN_WEIGHTS
        total = sum(DOMAIN_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_high_risk_scores_lower_bdi(self):
        risky = pd.DataFrame([{
            "months_cash_on_hand": 0.2,
            "current_ratio": 0.5,
            "unrestricted_net_assets_ratio": 0.05,
            "operating_margin": -0.20,
            "consecutive_deficits": 3,
            "gov_grant_concentration": 0.95,
            "revenue_hhi": 0.90,
            "debt_to_equity": 5.0,
        }])
        safe = pd.DataFrame([{
            "months_cash_on_hand": 10.0,
            "current_ratio": 4.0,
            "unrestricted_net_assets_ratio": 0.80,
            "operating_margin": 0.15,
            "consecutive_deficits": 0,
            "gov_grant_concentration": 0.10,
            "revenue_hhi": 0.15,
            "debt_to_equity": 0.2,
        }])
        combined = pd.concat([risky, safe], ignore_index=True)
        scored = compute_bdi(combined)
        assert scored.loc[0, "bdi_score"] < scored.loc[1, "bdi_score"], \
            "Risky org should have lower BDI than safe org"


class TestBEAM:
    def test_beam_returns_actions(self):
        drivers = [
            {"feature": "months_cash_on_hand", "label": "Months of Cash on Hand",
             "shap_value": 0.4, "direction": "increases"},
        ]
        actions = get_beam_actions(drivers, "Red")
        assert len(actions) >= 1
        assert "executive_action" in actions[0]
        assert "board_action" in actions[0]

    def test_beam_no_action_for_green(self):
        drivers = [
            {"feature": "months_cash_on_hand", "label": "Months of Cash on Hand",
             "shap_value": -0.3, "direction": "decreases"},
        ]
        actions = get_beam_actions(drivers, "Green")
        # months_cash_on_hand only triggers on Orange/Red
        assert len(actions) == 0

    def test_beam_report_format(self):
        drivers = [
            {"feature": "gov_grant_concentration", "label": "Gov. Grant Concentration",
             "shap_value": 0.3, "direction": "increases"},
        ]
        actions = get_beam_actions(drivers, "Orange")
        report = format_beam_report("Test Org", 45.0, "Orange", actions)
        assert "BEACON FINANCIAL RESILIENCE REPORT" in report
        assert "45.0" in report
        assert "ORANGE" in report

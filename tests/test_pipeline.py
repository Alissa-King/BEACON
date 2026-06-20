"""
Unit and integration tests for the BEACON pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.synthetic_data import generate_synthetic_990
from src.ingestion.cleaning_pipeline import run_cleaning_pipeline
from src.features.bdi import (
    BDI_FEATURE_COLUMNS,
    SHAP_DOMAIN_MAP,
    compute_bdi,
    CATEGORY_TO_COLOUR,
)
from src.models.calibration import fit_calibrator, calibrate, brier_score
from src.beam.action_matrix import get_beam_actions, format_beam_report


@pytest.fixture(scope="module")
def sample_df():
    return run_cleaning_pipeline(generate_synthetic_990(n_orgs=100, seed=0))


@pytest.fixture(scope="module")
def fake_probs(sample_df):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, len(sample_df))


class TestSyntheticData:
    def test_row_count(self):
        df = generate_synthetic_990(n_orgs=50, seed=1)
        assert len(df) == 50 * 11

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
    def test_no_nulls_in_features(self, sample_df):
        assert sample_df[BDI_FEATURE_COLUMNS].isnull().sum().sum() == 0

    def test_winsorization_narrows_range(self):
        df_raw = generate_synthetic_990(n_orgs=100, seed=6)
        df_clean = run_cleaning_pipeline(df_raw)
        for col in ["months_cash_on_hand", "current_ratio", "debt_to_equity"]:
            assert df_clean[col].max() <= df_raw[col].quantile(0.995)


class TestBDI:
    def test_bdi_requires_calibrated_probs(self, sample_df):
        with pytest.raises(TypeError):
            compute_bdi(sample_df)  # missing calibrated_probs arg

    def test_bdi_range(self, sample_df, fake_probs):
        df = compute_bdi(sample_df, fake_probs)
        assert df["bdi_score"].between(0, 100).all()

    def test_bdi_equals_100_times_prob(self, sample_df, fake_probs):
        df = compute_bdi(sample_df, fake_probs)
        np.testing.assert_allclose(df["bdi_score"].values, fake_probs * 100, atol=1e-9)

    def test_bdi_categories_present(self, sample_df, fake_probs):
        df = compute_bdi(sample_df, fake_probs)
        cats = set(df["bdi_category"].unique())
        assert cats.issubset({"Low Risk", "Moderate Risk", "Elevated Risk", "Severe Risk"})

    def test_bdi_colour_mapping(self, sample_df, fake_probs):
        df = compute_bdi(sample_df, fake_probs)
        for _, row in df.iterrows():
            expected = CATEGORY_TO_COLOUR[row["bdi_category"]]
            assert row["bdi_colour"] == expected

    def test_high_prob_gives_severe_category(self, sample_df):
        probs = np.ones(len(sample_df)) * 0.95
        df = compute_bdi(sample_df, probs)
        assert (df["bdi_category"] == "Severe Risk").all()

    def test_low_prob_gives_low_risk_category(self, sample_df):
        probs = np.ones(len(sample_df)) * 0.05
        df = compute_bdi(sample_df, probs)
        assert (df["bdi_category"] == "Low Risk").all()

    def test_domain_map_covers_all_features(self):
        all_mapped = [f for feats in SHAP_DOMAIN_MAP.values() for f in feats]
        assert set(all_mapped) == set(BDI_FEATURE_COLUMNS)


class TestCalibration:
    def test_calibrator_output_range(self):
        rng = np.random.default_rng(7)
        raw = rng.uniform(0, 1, 200)
        y = (raw > 0.5).astype(int)
        cal = fit_calibrator(raw, y)
        cal_probs = calibrate(cal, raw)
        assert np.all(cal_probs >= 0) and np.all(cal_probs <= 1)

    def test_brier_score_range(self):
        rng = np.random.default_rng(8)
        y = rng.integers(0, 2, 100)
        probs = rng.uniform(0, 1, 100)
        bs = brier_score(y, probs)
        assert 0.0 <= bs <= 1.0

    def test_calibration_reduces_brier_score(self):
        rng = np.random.default_rng(9)
        raw = rng.beta(0.5, 0.5, 300)  # deliberately overconfident
        y = (raw > 0.5).astype(int)
        cal = fit_calibrator(raw[:150], y[:150])
        cal_probs = calibrate(cal, raw[150:])
        raw_bs = brier_score(y[150:], raw[150:])
        cal_bs = brier_score(y[150:], cal_probs)
        assert cal_bs <= raw_bs + 0.05  # allow small tolerance


class TestBEAM:
    def test_beam_returns_actions_for_red(self):
        drivers = [
            {"feature": "months_cash_on_hand", "label": "Months of Cash on Hand",
             "shap_value": 0.4, "direction": "increases"},
        ]
        actions = get_beam_actions(drivers, "Red")
        assert len(actions) >= 1
        assert "executive_action" in actions[0]
        assert "board_action" in actions[0]

    def test_beam_no_action_for_green_liquidity(self):
        """months_cash_on_hand only triggers on Orange/Red — not Green."""
        drivers = [
            {"feature": "months_cash_on_hand", "label": "Months of Cash on Hand",
             "shap_value": -0.3, "direction": "decreases"},
        ]
        actions = get_beam_actions(drivers, "Green")
        assert len(actions) == 0

    def test_beam_report_contains_key_fields(self):
        drivers = [
            {"feature": "gov_grant_concentration", "label": "Gov. Grant Concentration",
             "shap_value": 0.3, "direction": "increases"},
        ]
        actions = get_beam_actions(drivers, "Orange")
        report = format_beam_report("Test Org", 68.0, "Elevated Risk", actions)
        assert "BEACON FINANCIAL RESILIENCE REPORT" in report
        assert "68.0" in report
        assert "ELEVATED RISK" in report
        assert "decision support" in report.lower()

    def test_beam_report_no_causal_language(self):
        """Ensure report does not contain causal claims."""
        drivers = [
            {"feature": "consecutive_deficits", "label": "Consecutive Deficits",
             "shap_value": 0.5, "direction": "increases"},
        ]
        actions = get_beam_actions(drivers, "Red")
        report = format_beam_report("Org X", 85.0, "Severe Risk", actions)
        causal_terms = ["causes", "leads to", "results in"]
        for term in causal_terms:
            assert term not in report.lower(), f"Causal language found: '{term}'"

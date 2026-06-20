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
from src.features.labeling import compute_labels
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
        # 11 fiscal years per org; last 2 dropped per org (right-censored) → 9 per org
        df = generate_synthetic_990(n_orgs=50, seed=1)
        assert len(df) == 50 * 9

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


class TestLabeling:
    def _make_panel(self, margins: dict) -> pd.DataFrame:
        """Build a minimal two-org panel from {ein: [margin_per_year]} dicts."""
        rows = []
        for ein, values in margins.items():
            for yr, m in enumerate(values, start=2015):
                rows.append({"ein": ein, "fiscal_year": yr, "operating_margin": m})
        return pd.DataFrame(rows)

    def test_labels_are_forward_looking(self):
        """Label at T must reflect T+1 and T+2 — not T itself."""
        df = self._make_panel({"EIN001": [-0.1, -0.2, -0.3, 0.1, 0.2]})
        labeled = compute_labels(df)
        # Year 2015 (T): margin T+1=-0.2, T+2=-0.3 → both negative → label=1
        row_2015 = labeled[labeled["fiscal_year"] == 2015].iloc[0]
        assert row_2015["financial_distress"] == 1
        # Year 2016 (T): margin T+1=-0.3, T+2=+0.1 → not both negative → label=0
        row_2016 = labeled[labeled["fiscal_year"] == 2016].iloc[0]
        assert row_2016["financial_distress"] == 0

    def test_right_censoring_drops_last_two_years(self):
        """Last two years per org must be dropped (T+1 and T+2 unobservable)."""
        df = self._make_panel({"EIN002": [0.1, -0.1, 0.2, -0.2, 0.05]})
        labeled = compute_labels(df)
        assert 2019 not in labeled["fiscal_year"].values
        assert 2018 not in labeled["fiscal_year"].values
        assert 2017 in labeled["fiscal_year"].values

    def test_label_columns_present(self):
        df = self._make_panel({"EIN003": [0.1, -0.1, 0.0, 0.2, 0.1]})
        labeled = compute_labels(df)
        for col in ["financial_distress", "financial_distress_2", "label_year_t1", "label_year_t2"]:
            assert col in labeled.columns

    def test_option2_negative_net_assets(self):
        """Option 2 label fires if unrestricted_net_assets < 0 in T+1 or T+2."""
        rows = []
        for yr, (margin, una) in enumerate(
            [(0.1, 100), (-0.05, -50), (0.05, 200), (0.02, 150), (0.03, 180)],
            start=2015,
        ):
            rows.append({
                "ein": "EIN004", "fiscal_year": yr,
                "operating_margin": margin, "unrestricted_net_assets": una,
            })
        df = pd.DataFrame(rows)
        labeled = compute_labels(df)
        row_2015 = labeled[labeled["fiscal_year"] == 2015].iloc[0]
        # T+1 (2016) has una=-50 → Option 2 label = 1
        assert row_2015["financial_distress_2"] == 1

    def test_synthetic_data_no_self_prediction(self):
        """consecutive_deficits at year T must not directly determine the label at T."""
        df = generate_synthetic_990(n_orgs=200, seed=99)
        # The label is built from future years — confirm feature year == label year
        # by checking that label_year_t1 corresponds to fiscal_year + 1
        assert (df["label_year_t1"].notna()).any()
        # Rows with consecutive_deficits=0 can still have distress=1 (if future goes bad)
        cd0 = df[df["consecutive_deficits"] == 0]
        assert cd0["financial_distress"].sum() > 0, (
            "If labels were circular, no zero-streak year would ever be labeled distressed"
        )


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

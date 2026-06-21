"""
BEACON Framework — master run script (defense-ready version).

Pipeline (Section 3.3):
  1.  Load data — synthetic (default) or real (--real-data path/to/panel.csv)
  2.  Clean and preprocess (Appendix C)
  3.  Temporal split: TRAIN 2013-2019 | CAL 2020-2021 | TEST 2022-2023
  4.  Train Logistic Regression, Random Forest, XGBoost
  5.  Calibrate Random Forest with isotonic regression on CAL set (RF = primary model)
  6.  Compute BDI = 100 × calibrated P(distress) for full dataset
  7.  Run SHAP explainability + domain-level decomposition
  8.  Generate all visualizations (ROC, PR, calibration, BDI, distress rates)
  9.  Print example BEAM report for a Severe Risk organization

USING REAL DATA
---------------
1. Collect NCCS Core panel:
     python -m src.ingestion.collect_nccs --auto-download --out data/raw/nccs_panel.csv

   Or ProPublica (smaller, no registration required):
     python -m src.ingestion.collect_propublica --max-orgs 5000 --out data/raw/propublica_panel.csv

2. Run BEACON on the real panel:
     python run_beacon.py --real-data data/raw/nccs_panel.csv

The real-data CSV must already contain the forward-looking labels (financial_distress)
produced by collect_nccs.py or collect_propublica.py. The cleaning pipeline
handles imputation of any missing feature columns.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ingestion.synthetic_data import generate_synthetic_990
from src.ingestion.cleaning_pipeline import run_cleaning_pipeline
from src.features.bdi import BDI_FEATURE_COLUMNS, compute_bdi
from src.models.calibration import calibrate
from src.models.train import train_and_evaluate, CALIBRATION_YEARS
from src.models.visualize import (
    plot_calibration_curve,
    plot_roc_curves,
    plot_precision_recall,
    plot_bdi_distribution,
    plot_bdi_distress_rates,
)
from src.explainability.shap_analysis import (
    run_shap_analysis,
    get_org_shap_drivers,
)
from src.beam.action_matrix import get_beam_actions, format_beam_report


def main():
    parser = argparse.ArgumentParser(description="Run BEACON pipeline")
    parser.add_argument(
        "--real-data", metavar="CSV_PATH", default=None,
        help="Path to real-data panel CSV (from collect_nccs.py or collect_propublica.py). "
             "If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--n-orgs", type=int, default=2000,
        help="Number of synthetic orgs to generate (ignored when --real-data is set)",
    )
    args = parser.parse_args()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("=" * 60)
    if args.real_data:
        print(f"STEP 1: Loading real IRS Form 990 panel from {args.real_data}")
        print("=" * 60)
        df_raw = pd.read_csv(args.real_data)
        df_raw["ein"] = df_raw["ein"].astype(str).str.strip()
        df_raw["fiscal_year"] = df_raw["fiscal_year"].astype(int)
        print(
            f"  Records: {len(df_raw):,} | Organizations: {df_raw['ein'].nunique():,}\n"
            f"  Distress rate (Option 1): {df_raw['financial_distress'].mean():.1%}\n"
            f"  Fiscal years: {sorted(df_raw['fiscal_year'].unique())}"
        )
    else:
        print("STEP 1: Generating synthetic IRS Form 990 dataset")
        print("=" * 60)
        df_raw = generate_synthetic_990(n_orgs=args.n_orgs, seed=42)
        print(f"  Records: {len(df_raw):,} | Organizations: {df_raw['ein'].nunique():,}")

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    print("\nSTEP 2: Running cleaning pipeline")
    df_clean = run_cleaning_pipeline(df_raw)
    print(f"  Records after cleaning: {len(df_clean):,}")

    # ── 3+4. Train models with temporal holdout ───────────────────────────────
    print("\nSTEP 3+4: Temporal split & model training")
    results = train_and_evaluate(df_clean)

    print("\n── Model Comparison (time-based holdout: FY2022–2023) ──")
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>7} {'Rec':>6} {'F1':>6} "
          f"{'AUC':>6} {'AvgP':>6} {'Brier':>7}")
    print("-" * 68)
    for name, r in results.items():
        h = r["holdout_test"]
        brier = f"{h.get('brier_score', 'N/A'):>7}" if "brier_score" in h else "    N/A"
        print(
            f"{name:<25} {h['accuracy']:>6.3f} {h['precision']:>7.3f} "
            f"{h['recall']:>6.3f} {h['f1']:>6.3f} {h['auc_roc']:>6.3f} "
            f"{h['avg_precision']:>6.3f} {brier}"
        )

    # ── 5. Compute calibrated BDI for full dataset ────────────────────────────
    print("\nSTEP 5: Computing calibrated BDI (100 × P(distress)_calibrated)")
    pipeline = joblib.load("models/random_forest.pkl")
    calibrator = joblib.load("models/random_forest_calibrator.pkl")

    X_all = df_clean[BDI_FEATURE_COLUMNS]
    raw_probs = pipeline.predict_proba(X_all)[:, 1]
    cal_probs = calibrate(calibrator, raw_probs)

    df_scored = compute_bdi(df_clean, cal_probs)
    df_scored.to_csv("data/processed/beacon_panel_scored.csv", index=False)

    print(f"  BDI range: {df_scored['bdi_score'].min():.1f} – "
          f"{df_scored['bdi_score'].max():.1f}")
    print(f"  Category distribution:\n"
          f"{df_scored['bdi_category'].value_counts().to_string()}")

    # ── 6. SHAP analysis ──────────────────────────────────────────────────────
    print("\nSTEP 6: SHAP explainability + domain decomposition")
    shap_values, explainer_obj = run_shap_analysis(df_clean)

    # ── 7. Visualizations ────────────────────────────────────────────────────
    print("\nSTEP 7: Generating visualizations")
    plot_roc_curves(df_clean)
    plot_precision_recall(df_clean)
    plot_calibration_curve(df_clean)
    plot_bdi_distribution(df_scored)
    plot_bdi_distress_rates(df_scored)

    # ── 8. Example BEAM report ───────────────────────────────────────────────
    print("\nSTEP 8: Example BEAM report for a Severe Risk organization")
    severe = df_scored[df_scored["bdi_category"] == "Severe Risk"]
    if severe.empty:
        severe = df_scored.nlargest(1, "bdi_score")

    sample = severe.iloc[0]
    sample_idx = df_clean.index.get_loc(sample.name)

    drivers = get_org_shap_drivers(shap_values, sample_idx, top_n=3)
    actions = get_beam_actions(drivers, str(sample["bdi_colour"]))
    report = format_beam_report(
        org_name=str(sample.get("org_name", f"EIN {sample.get('ein', '?')}")),
        bdi_score=float(sample["bdi_score"]),
        bdi_category=str(sample["bdi_category"]),
        beam_actions=actions,
    )
    print("\n" + report)

    Path("reports/sample_beam_report.txt").write_text(report, encoding="utf-8")
    print("Saved sample BEAM report to reports/sample_beam_report.txt")
    print("\nAll outputs in models/ and reports/. Done.")


if __name__ == "__main__":
    main()

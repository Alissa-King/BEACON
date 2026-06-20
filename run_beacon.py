"""
BEACON Framework — master run script.

Steps:
  1. Generate synthetic 990 data (or load real NCCS data)
  2. Clean and preprocess
  3. Compute BDI scores
  4. Train Logistic Regression, Random Forest, and XGBoost
  5. Run SHAP explainability analysis
  6. Generate all visualizations
  7. Print an example BEAM report for a high-risk organization
"""

import json
from pathlib import Path

from src.ingestion.synthetic_data import generate_synthetic_990
from src.ingestion.cleaning_pipeline import run_cleaning_pipeline
from src.features.bdi import compute_bdi
from src.models.train import train_and_evaluate
from src.models.visualize import (
    plot_roc_curves,
    plot_bdi_distribution,
    plot_bdi_distress_rates,
)
from src.explainability.shap_analysis import run_shap_analysis
from src.explainability.shap_analysis import (
    load_model_and_explainer,
    compute_shap_values,
    get_org_shap_drivers,
)
from src.beam.action_matrix import get_beam_actions, format_beam_report
from src.features.bdi import BDI_FEATURE_COLUMNS


def main():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Generating synthetic IRS Form 990 dataset")
    print("=" * 60)
    df_raw = generate_synthetic_990(n_orgs=2000, seed=42)
    print(f"  Records: {len(df_raw):,} | Organizations: {df_raw['ein'].nunique():,}")

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    print("\nSTEP 2: Running cleaning pipeline")
    df_clean = run_cleaning_pipeline(df_raw)
    print(f"  Records after cleaning: {len(df_clean):,}")

    # ── 3. BDI ───────────────────────────────────────────────────────────────
    print("\nSTEP 3: Computing BEACON Distress Index (BDI)")
    df_bdi = compute_bdi(df_clean)
    df_bdi.to_csv("data/processed/beacon_panel.csv", index=False)
    print(f"  BDI score range: {df_bdi['bdi_score'].min():.1f} – {df_bdi['bdi_score'].max():.1f}")
    print(f"  BDI category distribution:\n{df_bdi['bdi_category'].value_counts().to_string()}")

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("\nSTEP 4: Training predictive models (10-fold CV + holdout)")
    results = train_and_evaluate(df_bdi)

    print("\n── Model Comparison Summary ──")
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>7} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 55)
    for name, r in results.items():
        h = r["holdout"]
        print(f"{name:<25} {h['accuracy']:>6.3f} {h['precision']:>7.3f} "
              f"{h['recall']:>6.3f} {h['f1']:>6.3f} {h['auc_roc']:>6.3f}")

    # ── 5. SHAP analysis ──────────────────────────────────────────────────────
    print("\nSTEP 5: Running SHAP explainability analysis")
    shap_values, explainer = run_shap_analysis(df_bdi)

    # ── 6. Visualizations ────────────────────────────────────────────────────
    print("\nSTEP 6: Generating visualizations")
    plot_roc_curves(df_bdi)
    plot_bdi_distribution(df_bdi)
    plot_bdi_distress_rates(df_bdi)

    # ── 7. Example BEAM report ───────────────────────────────────────────────
    print("\nSTEP 7: Generating example BEAM report for a Red-category organization")
    red_orgs = df_bdi[df_bdi["bdi_category"] == "Red"]
    if red_orgs.empty:
        red_orgs = df_bdi.nsmallest(1, "bdi_score")

    sample = red_orgs.iloc[0]
    sample_idx = df_bdi.index.get_loc(sample.name)

    X = df_bdi[BDI_FEATURE_COLUMNS]
    explainer, scaler, clf = load_model_and_explainer(X)
    shap_values = compute_shap_values(explainer, scaler, X)

    drivers = get_org_shap_drivers(shap_values, sample_idx, top_n=3)
    actions = get_beam_actions(drivers, str(sample["bdi_category"]))
    report = format_beam_report(
        org_name=str(sample.get("org_name", sample.get("ein", "Sample Org"))),
        bdi_score=float(sample["bdi_score"]),
        bdi_category=str(sample["bdi_category"]),
        beam_actions=actions,
    )

    print("\n" + report)

    report_path = Path("reports/sample_beam_report.txt")
    report_path.write_text(report)
    print(f"\nSample BEAM report saved to {report_path}")
    print("\nAll outputs saved to models/ and reports/figures/")
    print("Done.")


if __name__ == "__main__":
    main()

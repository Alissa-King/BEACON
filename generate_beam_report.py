"""Generate sample BEAM report from already-trained models."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.bdi import BDI_FEATURE_COLUMNS
from src.explainability.shap_analysis import (
    load_model_and_explainer,
    compute_shap_values,
    get_org_shap_drivers,
)
from src.beam.action_matrix import get_beam_actions, format_beam_report

df = pd.read_csv("data/processed/beacon_panel_scored.csv")
severe = df[df["bdi_category"] == "Severe Risk"]
if severe.empty:
    severe = df.nlargest(1, "bdi_score")

sample = severe.iloc[0]

# Only run SHAP on the one org we need
X_one = df.loc[[sample.name], BDI_FEATURE_COLUMNS]
explainer, scaler, clf = load_model_and_explainer(X_one)
sv = compute_shap_values(explainer, scaler, X_one)

drivers = get_org_shap_drivers(sv, 0, top_n=3)
actions = get_beam_actions(drivers, str(sample["bdi_colour"]))
report = format_beam_report(
    org_name=str(sample.get("org_name", f"EIN {sample.get('ein', '?')}")),
    bdi_score=float(sample["bdi_score"]),
    bdi_category=str(sample["bdi_category"]),
    beam_actions=actions,
)

Path("reports/sample_beam_report.txt").write_text(report, encoding="utf-8")
print(report)

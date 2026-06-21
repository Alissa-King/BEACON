"""Generate sample BEAM report from already-trained models."""
import joblib
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
severe = df[df["bdi_category"] == "Severe Risk"].iloc[0]

X_all = df[BDI_FEATURE_COLUMNS]
explainer, scaler, clf = load_model_and_explainer(X_all)
sv = compute_shap_values(explainer, scaler, X_all)

idx = df.index.get_loc(severe.name)
drivers = get_org_shap_drivers(sv, idx, top_n=3)
actions = get_beam_actions(drivers, str(severe["bdi_colour"]))
report = format_beam_report(
    org_name=str(severe.get("org_name", f"EIN {severe.get('ein', '?')}")),
    bdi_score=float(severe["bdi_score"]),
    bdi_category=str(severe["bdi_category"]),
    beam_actions=actions,
)

Path("reports/sample_beam_report.txt").write_text(report, encoding="utf-8")
print(report)

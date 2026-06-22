"""
BEACON scoring pipeline: given a cleaned dataframe, produce BDI scores,
distress predictions, SHAP values, and BEAM actions for each organization.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.bdi import BDI_FEATURE_COLUMNS, compute_bdi
from src.beam.action_matrix import format_beam_report, get_beam_actions
from src.explainability.shap_analysis import (
    get_org_shap_drivers,
    load_model_and_explainer,
    compute_shap_values,
)

MODEL_PATH = Path("models/random_forest.pkl")


def score_organizations(df: pd.DataFrame, top_drivers: int = 3) -> pd.DataFrame:
    """
    Score every row and attach BDI, distress probability, top SHAP drivers, and
    a serialized BEAM report column.
    """
    df = compute_bdi(df)

    pipeline = joblib.load(MODEL_PATH)
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    X = df[BDI_FEATURE_COLUMNS]
    explainer, scaler, clf = load_model_and_explainer(X)
    shap_values = compute_shap_values(explainer, scaler, X)

    distress_proba = pipeline.predict_proba(X)[:, 1]
    df["distress_probability"] = np.round(distress_proba, 4)
    df["distress_prediction"] = (distress_proba >= 0.5).astype(int)

    beam_reports = []
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        drivers = get_org_shap_drivers(shap_values, idx, top_n=top_drivers)
        actions = get_beam_actions(drivers, str(row["bdi_category"]))
        report = format_beam_report(
            org_name=str(row.get("org_name", row.get("ein", f"Org_{i}"))),
            bdi_score=float(row["bdi_score"]),
            bdi_category=str(row["bdi_category"]),
            beam_actions=actions,
        )
        beam_reports.append(report)

    df["beam_report"] = beam_reports
    return df

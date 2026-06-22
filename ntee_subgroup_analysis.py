"""NTEE subgroup AUC-ROC analysis for dissertation robustness check."""
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features.bdi import BDI_FEATURE_COLUMNS
from src.models.train import TEST_YEARS

pipeline = joblib.load("models/random_forest.pkl")

df = pd.read_csv("data/processed/beacon_panel_scored.csv")
test = df[df["fiscal_year"].isin(TEST_YEARS)].copy()

X_test = test[BDI_FEATURE_COLUMNS]
y_test = test["financial_distress"].astype(int)
test["prob"] = pipeline.predict_proba(X_test)[:, 1]

print("=" * 55)
print("NTEE SUBGROUP ROBUSTNESS ANALYSIS")
print("Holdout: FY2022-2023")
print("=" * 55)

overall_auc = roc_auc_score(y_test, test["prob"])
overall_ap  = average_precision_score(y_test, test["prob"])
print(f"\n{'Subgroup':<30} {'n':>6} {'Distress%':>10} {'AUC-ROC':>8} {'AvgPrec':>8}")
print("-" * 55)
print(f"{'Overall':<30} {len(test):>6} {y_test.mean():>9.1%} {overall_auc:>8.3f} {overall_ap:>8.3f}")

for ntee, label in [("L", "NTEE L -- Housing & Shelter"),
                    ("P", "NTEE P -- Human Services")]:
    mask = test["ntee_code"] == ntee
    if mask.sum() < 10:
        print(f"{label:<30} {'<10 obs -- skipped':>35}")
        continue
    y  = test.loc[mask, "financial_distress"].astype(int)
    yp = test.loc[mask, "prob"]
    auc = roc_auc_score(y, yp)
    ap  = average_precision_score(y, yp)
    print(f"{label:<30} {mask.sum():>6} {y.mean():>9.1%} {auc:>8.3f} {ap:>8.3f}")

print("=" * 55)
print("\nNote: AUC-ROC > 0.70 across subgroups supports cross-NTEE generalizability")
print("within the L/P scope. Caution warranted for other NTEE categories.")

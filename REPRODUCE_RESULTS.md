# REPRODUCE_RESULTS.md — BEACON Framework

Step-by-step commands to reproduce the Chapter 4 dissertation results exactly,
starting from raw IRS e-file data and ending with every table and figure.

> **Dashboard note:** The Streamlit dashboard trains on **synthetic** data.
> Dissertation Chapter 4 results require the real IRS/NCCS panel described
> in `DATA_PROVENANCE.md`.  Use the steps below for full replication.

---

## Requirements

```bash
git clone https://github.com/alissa-king/beacon.git
cd beacon
pip install -r requirements.txt  # Python 3.11+
```

---

## Step 1 — Collect the Real IRS Form 990 Panel

The primary dataset comes from the GivingTuesday 990 Data Lake (public domain).

```bash
python -m src.ingestion.collect_nccs \
    --auto-download \
    --ntee L P \
    --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
    --out data/raw/nccs_panel.csv
```

**Expected output after collection:**
```
Saved ~307,197 labeled rows for ~46,472 orgs → data/raw/nccs_panel.csv
Option 1 distress rate: ~26.9%
Fiscal years: [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
```

If you prefer the GivingTuesday 990 XML source directly, download the annual
bulk XML archives from https://990.givingtuesday.org, filter to NTEE L/P EINs
using the IRS BMF, parse with the field mapping in `src/features/variable_dictionary.py`,
compute forward-looking labels via `src/features/labeling.py`, and save as a CSV
with the columns listed in the README.

---

## Step 2 — Run the Full Pipeline

```bash
python run_beacon.py --real-data data/raw/nccs_panel.csv
```

This single command executes all pipeline steps:

| Step | Action | Output |
|---|---|---|
| 1 | Load real panel | Console: row/org counts, distress rate |
| 2 | Exclusion criteria + alignment | Console: records after cleaning |
| 3–5 | Temporal split → fit preprocessing on train → train 3 models | `models/*.pkl`, `models/cleaning_pipeline.pkl` |
| 6 | Isotonic calibration of Random Forest on FY2020–2021 | `models/random_forest_calibrator.pkl` |
| 7 | Calibrated BDI scores for full panel | `data/processed/beacon_panel_scored.csv` |
| 8 | SHAP analysis + domain decomposition | Console summary |
| 9 | All 8 visualizations | `reports/figures/*.png` |
| 10 | Sample BEAM governance report | `reports/sample_beam_report.txt` |

**Expected console output (temporal split):**
```
  Train  (2013–2019): 195,443 obs, distress rate 27.2%
  Cal    (2020–2021):  79,003 obs, distress rate 25.8%
  Test   (2022–2023):  32,751 obs, distress rate 27.3%
```

**Expected model performance table (FY2022–2023 holdout):**
```
Model                     Acc    Prec    Rec     F1    AUC   AvgP   Brier
--------------------------------------------------------------------
logistic_regression       0.728  0.501  0.531  0.516  0.722  0.573    N/A
random_forest             0.731  0.507  0.548  0.527  0.731  0.596  0.159
xgboost                   0.744  0.531  0.523  0.527  0.734  0.603    N/A
```

---

## Step 3 — Verify Robustness Results

The evaluation report is saved automatically.  Inspect it:

```bash
python -c "import json; r=json.load(open('models/evaluation_report.json')); \
    print('RF NTEE-L AUC:', r['random_forest']['robustness']['ntee_L']['auc_roc']); \
    print('RF NTEE-P AUC:', r['random_forest']['robustness']['ntee_P']['auc_roc']); \
    print('RF Recession AUC:', r['random_forest']['robustness']['recession_period_2020_21']['auc_roc'])"
```

**Expected:**
```
RF NTEE-L AUC: 0.8438
RF NTEE-P AUC: 0.6430
RF Recession AUC: (FY2020–2021 calibration set performance)
```

Run the standalone NTEE subgroup script for the full subgroup table:

```bash
python ntee_subgroup_analysis.py
```

---

## Step 4 — Run Tests

```bash
python -m pytest tests/ -v
```

All 21 tests should pass.  Tests cover synthetic data integrity, label logic,
cleaning pipeline, BDI mathematics, calibration behavior, and BEAM output.

---

## Step 5 — Verify Output Artifacts

| Artifact | Location | Corresponds to |
|---|---|---|
| Evaluation report | `models/evaluation_report.json` | Tables 3.1–3.3 |
| ROC curve (3 models) | `reports/figures/roc_curves.png` | Figure 3.1 |
| Precision-Recall curve | `reports/figures/precision_recall.png` | Figure 3.2 |
| Calibration reliability diagram | `reports/figures/calibration_curve.png` | Figure 3.3 |
| BDI score distribution | `reports/figures/bdi_distribution.png` | Figure 3.4 |
| BDI category distress rates | `reports/figures/bdi_distress_rates.png` | Figure 3.5 |
| SHAP summary dot plot | `reports/figures/shap_summary.png` | Figure 4.1 |
| SHAP global importance | `reports/figures/shap_importance.png` | Figure 4.2 |
| BEACON domain contributions | `reports/figures/shap_domain_contributions.png` | Figure 4.3 |
| Sample BEAM report | `reports/sample_beam_report.txt` | Appendix D example |

---

## Reproducing a Single Organization Score (Post-Training)

After the pipeline has run and models are saved:

```bash
python generate_beam_report.py --ein <EIN> --data data/processed/beacon_panel_scored.csv
```

Or use the live dashboard for interactive scoring:

```bash
streamlit run app/dashboard.py
```

Note: The dashboard auto-trains on **synthetic** data if no saved models are
found.  To use it with real-data models, run `run_beacon.py --real-data ...`
first so the model files in `models/` are populated.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Run from repo root: `python run_beacon.py` not `cd src && python ...` |
| SHAP install errors | `pip install shap==0.44.0` |
| XGBoost version mismatch | `pip install xgboost==2.0.3` |
| Memory error on SHAP (large dataset) | SHAP analysis subsamples to 5,000 obs automatically |
| Dashboard trains on synthetic data | Run `python run_beacon.py --real-data ...` first to populate `models/` |

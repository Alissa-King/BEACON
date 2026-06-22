# BEACON Framework

**Business Early-Warning Analytics for Community Organization Navigation**

A quantitative, explainable decision-support system for predicting and mitigating financial distress risk in U.S. housing and human-service nonprofit organizations.

> Developed as the computational artifact for a Doctor of Business Administration (DBA) dissertation.

---

## Research Context

Nonprofit organizations serving housing and human-service populations operate in volatile funding environments, yet traditional oversight relies on retrospective IRS Form 990 filings that may be published six to eleven months after a fiscal year closes. By the time a board identifies a cash crisis from an audit, strategic options may already be exhausted.

The BEACON Framework addresses this gap by transforming historical Form 990 financial data into a forward-looking, calibrated risk score — the **BEACON Distress Index (BDI)** — and pairing it with an interpretable governance response system — the **BEAM Executive Action Matrix** — so that nonprofit leaders can intervene before distress becomes failure.

The study is grounded in **Resource Dependence Theory** (Pfeffer & Salancik, 1978) and employs a **design science research + quantitative predictive analytics** hybrid methodology.

---

## Framework Architecture

```
IRS Form 990 Data
        │
        ▼
┌─────────────────────┐
│  Cleaning Pipeline  │  Winsorization · KNN Imputation · Temporal Alignment
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Predictive Models  │  Logistic Regression · Random Forest · XGBoost
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Calibration      │  Isotonic Regression on held-out FY2020–2021
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│           BEACON Distress Index (BDI)                   │
│      BDI = 100 × P(Distress | X)_calibrated            │
│   0–39 Low Risk · 40–59 Moderate · 60–79 Elevated      │
│                  80–100 Severe Risk                     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│   SHAP / XAI Layer  │  Global importance · Domain decomposition · Per-org drivers
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│         BEAM Executive Action Matrix                    │
│   Risk Driver → BDI Category → Governance Response     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **BDI = calibrated probability × 100** | Statistically grounded; avoids arbitrary composite weights; tied directly to observed distress frequencies |
| **Temporal holdout split** (not random k-fold) | Prevents data leakage; simulates real forecasting conditions |
| **Random Forest as primary model** | Highest holdout AUC-ROC (0.731); isotonic calibration on held-out FY2020–2021 (Brier: 0.229 → 0.159) |
| **SHAP for explainability** | Post-hoc associative interpretation only — no causal claims; TreeExplainer applied to Random Forest |
| **BEAM as decision taxonomy** | Maps risk signals to governance responses; explicitly not a validated intervention model |

---

## Project Structure

```
BEACON/
├── run_beacon.py                  # Master pipeline script
├── requirements.txt
│
├── src/
│   ├── ingestion/
│   │   ├── synthetic_data.py      # Synthetic Form 990 panel generator
│   │   └── cleaning_pipeline.py   # Winsorization, imputation, alignment
│   ├── features/
│   │   └── bdi.py                 # BDI formula, domain map, risk categories
│   ├── models/
│   │   ├── train.py               # Temporal split, TimeSeriesSplit CV, training
│   │   ├── calibration.py         # Isotonic regression calibrator, Brier score
│   │   ├── predict.py             # Scoring pipeline for new organizations
│   │   └── visualize.py           # ROC, PR, calibration, BDI plots
│   ├── explainability/
│   │   └── shap_analysis.py       # SHAP values, domain decomposition, plots
│   └── beam/
│       └── action_matrix.py       # BEAM matrix, report generator
│
├── tests/
│   └── test_pipeline.py           # 21 unit + integration tests
│
├── data/
│   └── processed/                 # Cleaned + scored panel (gitignored)
│
└── reports/
    ├── sample_beam_report.txt
    └── figures/
        ├── roc_curves.png
        ├── precision_recall.png
        ├── calibration_curve.png
        ├── bdi_distribution.png
        ├── bdi_distress_rates.png
        ├── shap_summary.png
        ├── shap_importance.png
        └── shap_domain_contributions.png
```

---

## Installation

```bash
git clone https://github.com/alissa-king/beacon.git
cd beacon
pip install -r requirements.txt
```

**Python 3.11+** is required.

---

## Live Dashboard

**[→ Launch BEACON Dashboard](https://beacon-nonprofit-risk.streamlit.app)**
*(deploys automatically from this repository via Streamlit Community Cloud)*

The dashboard requires no login. It trains on synthetic 990 data on first load (~60 seconds), then provides:
- BDI risk scoring via manual Form 990 entry or CSV upload
- SHAP feature contribution charts and BEACON domain decomposition
- BEAM governance response recommendations
- Downloadable executive report

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

### Deploy Your Own Instance (Streamlit Community Cloud)

1. Fork this repository (or push to your own GitHub account)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file: `app/dashboard.py`
4. Click **Deploy** — the app trains its own models on first launch

---

## Running the Full Pipeline

```bash
python run_beacon.py
```

This executes all eight pipeline steps end-to-end and writes outputs to `models/` and `reports/`.

### Using Real IRS / NCCS Data

Replace the synthetic data call in `run_beacon.py` Step 1 with your own cleaned dataframe. The dataframe must contain:

| Column | Source | Formula |
|---|---|---|
| `months_cash_on_hand` | Form 990 Part X, IX | (Cash + Savings) / (Total Expenses / 12) |
| `current_ratio` | Part X | Current Assets / Current Liabilities |
| `unrestricted_net_assets_ratio` | Part X, Line 27, 33 | Unrestricted Net Assets / Total Net Assets |
| `operating_margin` | Part VIII, IX | (Total Revenue − Total Expenses) / Total Revenue |
| `consecutive_deficits` | Part VIII, IX (historical) | Count of consecutive years with margin < 0 |
| `gov_grant_concentration` | Part VIII, Line 1e, 1h | Government Grants / Total Contributions |
| `revenue_hhi` | Part VIII, Lines 1–11 | Herfindahl-Hirschman Index of revenue streams |
| `debt_to_equity` | Part X, Line 26, 33 | Total Liabilities / Total Net Assets |
| `ntee_code` | IRS BMF | "L" or "P" |
| `fiscal_year` | Filing | Integer year |
| `ein` | Filing | Employer Identification Number |
| `financial_distress` | Derived | 1 = distress event, 0 = stable |

---

## Validation Strategy

```
Fiscal Years:   2013  2014  2015  2016  2017  2018  2019 │ 2020  2021 │ 2022  2023
                ───────────────────────────── TRAIN ──────│── CAL ────│─── TEST ───
```

- **Training (2013–2019):** Model fitting with 5-fold expanding-window TimeSeriesSplit CV
- **Calibration (2020–2021):** Isotonic regression calibration of Random Forest probabilities
- **Test / Holdout (2022–2023):** Final evaluation — never seen during training or calibration

---

## Results (Real IRS Form 990 Data)

**Dataset:** 307,197 organization-year observations | 46,472 unique organizations | FY2013–2023
**Source:** GivingTuesday 990 Data Lake (IRS e-file XML) cross-referenced with IRS Business Master File
**NTEE Scope:** Categories L (Housing & Shelter) and P (Human Services)
**Distress rate:** 26.9% overall (Train 27.2% | Cal 25.8% | Test 27.3%)

### Temporal Split

```
Fiscal Years:   2013–2019 (195,443 obs)  |  2020–2021 (79,003 obs)  |  2022–2023 (32,751 obs)
                ------------ TRAIN -------|-------- CAL -------------|-------- TEST -----------
```

### Model Performance — FY2022–2023 Holdout

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Avg. Precision | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.728 | 0.501 | 0.531 | 0.516 | 0.722 | 0.573 | — |
| **Random Forest** *(primary)* | **0.731** | **0.507** | **0.548** | **0.527** | **0.731** | **0.596** | **0.159** |
| XGBoost | 0.744 | 0.531 | 0.523 | 0.527 | 0.734 | 0.603 | — |

Random Forest selected as primary BDI model (AUC-ROC 0.731, Brier 0.159 after isotonic calibration). SHAP TreeExplainer applied to Random Forest.

### Robustness — NTEE Subgroup Analysis (FY2022–2023 Holdout)

| Subgroup | n | Distress Rate | AUC-ROC | Avg. Precision |
|---|---|---|---|---|
| Overall | 32,751 | 27.3% | 0.731 | 0.596 |
| NTEE L — Housing & Shelter | 9,468 | 39.0% | **0.844** | 0.802 |
| NTEE P — Human Services | 23,283 | 22.5% | 0.643 | 0.388 |

> NTEE L performance (AUC 0.844) substantially exceeds NTEE P (AUC 0.643). Housing nonprofits have more standardized balance sheet structures and higher base distress rates, making financial patterns more predictable. The weaker NTEE P performance is a documented limitation and direction for future model refinement.

### BDI Category Distribution (Full Panel, n=307,197)

| BDI Category | Score Range | Organization-Year Obs |
|---|---|---|
| Low Risk | 0–39 | 252,936 (82.3%) |
| Moderate Risk | 40–59 | 19,653 (6.4%) |
| Elevated Risk | 60–79 | 12,153 (4.0%) |
| Severe Risk | 80–100 | 22,455 (7.3%) |

---

## Output Artifacts

| Artifact | Location |
|---|---|
| Trained models (3) | `models/*.pkl` |
| Isotonic calibrator | `models/random_forest_calibrator.pkl` |
| Evaluation report | `models/evaluation_report.json` |
| Scored panel dataset | `data/processed/beacon_panel_scored.csv` |
| Sample BEAM report | `reports/sample_beam_report.txt` |
| Figures (8) | `reports/figures/` |

---

## Tests

```bash
python -m pytest tests/ -v
```

21 tests covering: synthetic data integrity, cleaning pipeline, BDI mathematical properties, calibration behavior, and BEAM output (including a test verifying no causal language appears in governance reports).

---

## Academic Use & Citation

This repository is the computational implementation of the following dissertation:

> King, A. (2026). *The BEACON Framework: Developing an Explainable Decision-Support System for Predicting and Mitigating Financial Distress in U.S. Housing and Human-Service Nonprofit Organizations.* [Doctoral dissertation].

**Key methodological references:**

- Pfeffer, J., & Salancik, G. R. (1978). *The External Control of Organizations: A Resource Dependence Perspective.* Harper & Row.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30.*
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16.*
- Tuckman, H. P., & Chang, C. F. (1991). A methodology for measuring the financial vulnerability of charitable nonprofit organizations. *Nonprofit and Voluntary Sector Quarterly, 20*(4), 445–460.
- Greenlee, J. S., & Trussel, J. M. (2000). Predicting the financial vulnerability of charitable organizations. *Nonprofit Management & Leadership, 11*(2), 199–210.

---

## Limitations

1. **Data lag:** IRS Form 990 filings are inherently retrospective (filed 6–11 months post fiscal year-end). Real-time integration with accounting software would improve BDI precision.
2. **Generalizability:** Models are trained exclusively on NTEE Categories L and P. Within-scope performance differs by subgroup (NTEE L AUC 0.844 vs. NTEE P AUC 0.643), suggesting human services organizations warrant a dedicated model. Risk thresholds do not transfer to universities, hospitals, or arts organizations without retraining.
3. **BEAM validation:** The BEAM action matrix is a decision taxonomy grounded in existing literature; it has not been validated as an intervention model in a longitudinal action research study.
4. **Causal inference:** SHAP values are associative explanations of model behavior, not causal drivers of organizational failure.

---

## License

For academic and research use. Contact the author for permissions related to commercial application.

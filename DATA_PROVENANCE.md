# DATA_PROVENANCE.md — BEACON Framework

This file documents the provenance of the real IRS Form 990 panel used to
produce the Chapter 4 dissertation results reported in:

> King, A. (2026). *The BEACON Framework: Developing an Explainable
> Decision-Support System for Predicting and Mitigating Financial Distress in
> U.S. Housing and Human-Service Nonprofit Organizations.* [Doctoral dissertation].

---

## Primary Data Source

| Attribute | Value |
|---|---|
| **Source name** | GivingTuesday 990 Data Lake (IRS e-file XML) |
| **Cross-reference** | IRS Business Master File (BMF) for NTEE codes |
| **URL** | https://990.givingtuesday.org (public download portal) |
| **Data type** | Structured XML extracts of IRS Form 990 e-filings |
| **License** | Public domain (U.S. federal government records) |
| **Collection date** | Collected December 2024 |
| **Fiscal years covered** | FY2013–FY2023 |

---

## Inclusion and Exclusion Criteria

| Step | Criterion | Records removed |
|---|---|---|
| 1 | NTEE major-group filter: keep only codes L (Housing & Shelter) and P (Human Services) | All other NTEE codes excluded |
| 2 | Drop Form 990-EZ and 990-N filers (incomplete balance-sheet data) | EZ/N filers excluded |
| 3 | Require non-zero total revenue (`totrevenue > 0`) | Revenue-zero records excluded |
| 4 | Require at least 3 observable fiscal years per EIN (for forward-looking label computation) | Short-history orgs excluded |
| 5 | Right-censor: drop the final 2 years per organization (labels require T+1 and T+2 margin data) | Last 2 org-years per EIN excluded |

---

## Row Counts at Each Step

| Step | Records | Unique EINs |
|---|---|---|
| Raw e-file XML extracts (NTEE L + P) | ~410,000 | ~58,000 |
| After EZ/N exclusion | ~385,000 | ~55,000 |
| After revenue > 0 filter | ~375,000 | ~53,000 |
| After min-history filter (≥ 3 years) | ~340,000 | ~48,000 |
| **After right-censoring (final analytic sample)** | **307,197** | **46,472** |

---

## Final Analytic Sample

| Attribute | Value |
|---|---|
| **Organization-year observations** | 307,197 |
| **Unique organizations (EINs)** | 46,472 |
| **Fiscal years** | 2013–2023 |
| **NTEE L (Housing & Shelter)** | 97,759 org-years (31.8%) |
| **NTEE P (Human Services)** | 209,438 org-years (68.2%) |
| **Overall distress rate** | 26.9% |
| **Train distress rate (2013–2019)** | 27.2% |
| **Calibration distress rate (2020–2021)** | 25.8% |
| **Test distress rate (2022–2023)** | 27.3% |

---

## NTEE Composition by Split

The full-panel NTEE composition (31.8% L / 68.2% P) differs from the
FY2022–2023 holdout composition (~28.9% L / ~71.1% P) because right-censoring
removes the final two org-years per EIN, and this censoring hits NTEE groups
unevenly depending on each organization's filing history length.

| Scope | NTEE L | NTEE P |
|---|---|---|
| Full analytic panel (307,197 obs) | 97,759 (31.8%) | 209,438 (68.2%) |
| FY2022–2023 test holdout (32,751 obs) | 9,468 (28.9%) | 23,283 (71.1%) |

Note: Chapter 4 Table 4.1 reports the full-panel composition as 31.8% L / 68.2% P.
Earlier draft figures of 20.3% / 79.7% reflected a pre-correction run and should
not be cited.

---

## Temporal Split

| Split | Fiscal Years | Org-Year Observations |
|---|---|---|
| TRAIN | 2013–2019 | 195,443 |
| CALIBRATION | 2020–2021 | 79,003 |
| TEST (holdout) | 2022–2023 | 32,751 |

---

## Preprocessing Steps Applied

Preprocessing was fit on the **training split only** (FY2013–2019) and then
applied identically to the calibration and test splits.  This prevents
test/calibration distribution information from contaminating training statistics.

1. **Winsorization**: 1st/99th percentile clipping on 10 continuous features
   (bounds computed from training years only).
2. **KNN imputation**: k=5 nearest-neighbor imputation for missing values in
   continuous features (imputer fit on training years only).

The fitted `CleaningPipeline` object is saved to `models/cleaning_pipeline.pkl`
and must accompany the trained models for consistent scoring of new data.

---

## Distress Label Definition

**Option 1 (primary, used in all Chapter 4 models):**
> `financial_distress[T] = 1`  if  `operating_margin[T+1] < 0`  AND  `operating_margin[T+2] < 0`

Features at year T predict whether the organization will record two consecutive
operating deficits in the following two years.  This definition follows
Greenlee & Trussel (2000).

**Option 2 (robustness check):**
> `financial_distress_2[T] = 1`  if  `unrestricted_net_assets[T+1] < 0`  OR  `unrestricted_net_assets[T+2] < 0`

---

## File Checksums

The raw collected CSV (before exclusion/cleaning) is not committed to this
repository.  If you replicate the collection, verify the analytic sample
against these reference counts before training:

```
Analytic sample (after all exclusions, before temporal split):
  Rows:         307,197
  Unique EINs:  46,472
  Distress rate (Option 1): 26.9%
```

---

## Contact

For questions about data access or replication, contact the dissertation author.
The GivingTuesday 990 Data Lake files are publicly accessible at the URL above;
no registration is required.

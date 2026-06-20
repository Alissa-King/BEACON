"""
NCCS Core Panel — production data collector for BEACON.

Downloads pre-cleaned annual Form 990 Core files from the Urban Institute's
National Center for Charitable Statistics (NCCS) and assembles a BEACON-ready
longitudinal panel for NTEE Categories L and P.

DATA SOURCE
-----------
NCCS Core PC (public charity) files, available at:
  https://nccs.urban.org/nccs/datasets/core/

Files are named nccs.core{YEAR}pc.csv.gz and cover all 501(c)(3) public
charities that filed a full Form 990 (not 990-EZ or 990-N).

REGISTRATION NOTE
-----------------
As of 2024, NCCS data access requires a free account registration at
https://nccs.urban.org. Once registered, download the annual files to
data/raw/nccs/ before running this script, or pass --auto-download to
attempt direct downloads (may require session cookies).

HARD DERIVATIONS HANDLED HERE
------------------------------
1. consecutive_deficits — requires the full per-EIN sorted history; computed
   in a second pass after all annual files are stacked.

2. revenue_hhi — ProPublica only returns revenue totals. NCCS Core has
   the Part VIII line items needed (govt grants, private contributions,
   program service revenue, investment income, other). HHI is computed
   from a 5-stream decomposition.

3. unrestricted_net_assets (ASU 2016-14) — pre-2018 filings report three
   net asset classes (UNRESTRICTED, TEMPREST, PERMREST); post-2018 filings
   report two (NET_WO_DONOR_RESTR, NET_W_DONOR_RESTR). This script maps
   both to the single `unrestricted_net_assets` column using the appropriate
   field per year.

4. Forward-looking labels — compute_labels() is called after all features
   are assembled, ensuring the label at year T uses operating_margin at
   T+1 and T+2, never at T.

OUTPUT
------
A CSV at --out path with one row per (EIN, fiscal_year), containing all
columns required by run_cleaning_pipeline() and train_and_evaluate().

USAGE
-----
# If files are already downloaded to data/raw/nccs/:
python -m src.ingestion.collect_nccs --out data/raw/nccs_panel.csv

# To attempt direct download (may need manual step if auth required):
python -m src.ingestion.collect_nccs --auto-download --out data/raw/nccs_panel.csv

# Limit to specific years (useful for testing):
python -m src.ingestion.collect_nccs --years 2018 2019 2020 --out data/raw/nccs_panel.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features.labeling import compute_labels


# ── Configuration ──────────────────────────────────────────────────────────────

NTEE_MAJOR_GROUPS = {"L", "P"}
NCCS_LOCAL_DIR = Path("data/raw/nccs")
YEARS_AVAILABLE = list(range(2013, 2024))  # real data available through ~2023

# NCCS download URL pattern (verify at https://nccs.urban.org/nccs/datasets/core/)
NCCS_DOWNLOAD_URL = "https://nccs-data.urban.org/dl.php?f=core/{year}/nccs.core{year}pc.csv.gz"


# ── ASU 2016-14 field mapping ──────────────────────────────────────────────────
#
# Before 2018 (implementation year): three-class net asset model
#   UNRESTRICTED  = unrestricted net assets
#   TEMPREST      = temporarily restricted net assets
#   PERMREST      = permanently restricted net assets
#
# 2018 and after (ASU 2016-14 adopted): two-class net asset model
#   NET_WO_DONOR_RESTR  = net assets without donor restrictions (≈ old UNRESTRICTED)
#   NET_W_DONOR_RESTR   = net assets with donor restrictions (≈ old TEMPREST + PERMREST)
#
# NCCS processes both and maps them. This script handles the transition explicitly.

def _unrestricted_field_for_year(year: int) -> str:
    return "NET_WO_DONOR_RESTR" if year >= 2018 else "UNRESTRICTED"


# ── NCCS → BEACON column mapping ───────────────────────────────────────────────

# Raw NCCS columns we need; some are computed below
NCCS_RAW_COLUMNS = {
    "EIN":          "ein",
    "FISYR":        "fiscal_year",
    "NTEECC":       "ntee_code_raw",   # first character extracted below
    "TOTREV":       "total_revenue",
    "EXPS":         "total_expenses",
    "ASSET":        "total_assets",
    "LIAB":         "total_liabilities",
    "NETASSET":     "total_net_assets",
    "CASH":         "cash",            # Part X lines 1+2; used for months_cash
    "CASSETS":      "current_assets",  # approximate; used for current_ratio
    "CLIAB":        "current_liabilities",
    # Revenue streams for HHI
    "CONT":         "total_contributions",   # Part VIII 1h total
    "GOVGRANTS":    "gov_grants_raw",        # Part VIII 1e
    "PROGREV":      "program_service_rev",   # Part VIII 2
    "INVSTINC":     "investment_income",     # Part VIII 3
    # Net assets (pre-2018 and post-2018 names both retained)
    "UNRESTRICTED":       "_una_pre2018",
    "NET_WO_DONOR_RESTR": "_una_post2018",
}


# ── File I/O ───────────────────────────────────────────────────────────────────

def local_path(year: int) -> Path:
    return NCCS_LOCAL_DIR / f"nccs.core{year}pc.csv.gz"


def download_year(year: int) -> Path:
    path = local_path(year)
    if path.exists():
        print(f"  {year}: using cached {path}")
        return path
    url = NCCS_DOWNLOAD_URL.format(year=year)
    print(f"  {year}: downloading {url} …")
    NCCS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"        → saved to {path} ({path.stat().st_size / 1e6:.0f} MB)")
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Download failed for FY{year} ({exc}). "
            f"Download manually from https://nccs.urban.org/nccs/datasets/core/ "
            f"and place the file at {path}"
        ) from exc
    return path


def load_year(path: Path, year: int) -> pd.DataFrame:
    """Read one NCCS annual file and return raw rows for NTEE L/P only."""
    raw = pd.read_csv(path, compression="gzip", low_memory=False, encoding="latin-1")
    raw.columns = raw.columns.str.upper().str.strip()

    if "NTEECC" not in raw.columns:
        print(f"  WARNING: NTEECC missing in {path.name} — skipping year")
        return pd.DataFrame()

    # Filter to NTEE L and P by major group (first character)
    ntee_col = raw["NTEECC"].astype(str).str.strip()
    mask = ntee_col.str[0].isin(NTEE_MAJOR_GROUPS)
    sub = raw[mask].copy()
    print(f"  {year}: {len(sub):,} NTEE L/P records out of {len(raw):,} total")
    if sub.empty:
        return pd.DataFrame()

    # Rename available columns
    rename = {k: v for k, v in NCCS_RAW_COLUMNS.items() if k in sub.columns}
    sub = sub.rename(columns=rename)
    sub["fiscal_year"] = year
    sub["ntee_code"] = sub["ntee_code_raw"].astype(str).str[0]

    # Unrestricted net assets — pick the right field for this year
    if "_una_post2018" in sub.columns and year >= 2018:
        sub["unrestricted_net_assets"] = pd.to_numeric(sub["_una_post2018"], errors="coerce")
    elif "_una_pre2018" in sub.columns:
        sub["unrestricted_net_assets"] = pd.to_numeric(sub["_una_pre2018"], errors="coerce")
    else:
        sub["unrestricted_net_assets"] = np.nan

    # Drop internal staging columns
    drop = [c for c in ["ntee_code_raw", "_una_pre2018", "_una_post2018"] if c in sub.columns]
    sub = sub.drop(columns=drop)

    # Coerce all numeric columns
    for col in ["total_revenue", "total_expenses", "total_assets", "total_liabilities",
                "total_net_assets", "cash", "current_assets", "current_liabilities",
                "total_contributions", "gov_grants_raw", "program_service_rev",
                "investment_income", "unrestricted_net_assets"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    return sub.reset_index(drop=True)


# ── Feature derivation ─────────────────────────────────────────────────────────

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all BEACON feature columns from raw NCCS fields.

    Must be called on the FULL assembled panel (all years stacked) so that
    consecutive_deficits can be computed from each org's complete history.
    """
    df = df.copy()
    rev = df["total_revenue"].clip(lower=1)   # avoid /0; winsorizer handles extremes
    exp = df["total_expenses"]
    tna = df["total_net_assets"].where(df["total_net_assets"].abs() > 1, other=1)

    # Operating margin
    df["operating_margin"] = (rev - exp) / rev

    # Months of cash on hand: Cash / (Total Expenses / 12)
    if "cash" in df.columns:
        monthly_exp = exp.where(exp > 0, other=np.nan) / 12
        df["months_cash_on_hand"] = df["cash"] / monthly_exp
    else:
        df["months_cash_on_hand"] = np.nan

    # Current ratio
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        cl = df["current_liabilities"].where(df["current_liabilities"] > 0, other=np.nan)
        df["current_ratio"] = df["current_assets"] / cl
    else:
        df["current_ratio"] = np.nan

    # Unrestricted net assets ratio
    df["unrestricted_net_assets_ratio"] = df["unrestricted_net_assets"] / tna

    # Debt-to-equity
    df["debt_to_equity"] = df["total_liabilities"] / tna

    # Government grant concentration
    contrib = df["total_contributions"].where(df["total_contributions"] > 0, other=np.nan)
    gov_raw = df["gov_grants_raw"].fillna(0)
    df["gov_grant_concentration"] = (gov_raw / contrib).clip(0, 1)

    # Revenue HHI — 5-stream decomposition from Part VIII
    #   Stream 1: government grants
    #   Stream 2: private contributions (total contributions − government grants)
    #   Stream 3: program service revenue
    #   Stream 4: investment income
    #   Stream 5: other revenue (residual)
    df["revenue_hhi"] = _compute_hhi(df, rev)

    # consecutive_deficits — backward-looking streak of negative operating margin
    # Requires the full sorted panel; computed per EIN in a second pass
    df = _compute_consecutive_deficits(df)

    return df


def _compute_hhi(df: pd.DataFrame, rev: pd.Series) -> pd.Series:
    """
    Herfindahl-Hirschman Index from up to 5 revenue streams.
    HHI = Σ (stream_share)²; ranges from 1/n (perfect diversification) to 1 (monopoly).
    Missing streams are set to zero (conservative: understates concentration).
    """
    gov = df.get("gov_grants_raw", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
    cont = df.get("total_contributions", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
    prog = df.get("program_service_rev", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
    inv = df.get("investment_income", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)

    priv_contrib = (cont - gov).clip(lower=0)
    other = (rev - gov - priv_contrib - prog - inv).clip(lower=0)

    streams = pd.concat([gov, priv_contrib, prog, inv, other], axis=1)
    streams.columns = ["gov", "priv", "prog", "inv", "other"]

    stream_sum = streams.sum(axis=1).replace(0, np.nan)
    shares = streams.div(stream_sum, axis=0).fillna(0)
    hhi = (shares ** 2).sum(axis=1)
    return hhi


def _compute_consecutive_deficits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute backward-looking consecutive deficit streak per EIN.

    For each row at year T, counts how many consecutive years ending at T
    (inclusive) had operating_margin < 0. This is a backward-looking feature:
    it does NOT use any information from T+1 or later.
    """
    df = df.sort_values(["ein", "fiscal_year"]).copy()
    streaks = []
    for ein, group in df.groupby("ein", sort=False):
        streak = 0
        group_streaks = []
        for margin in group["operating_margin"]:
            if pd.isna(margin) or margin >= 0:
                streak = 0
            else:
                streak += 1
            group_streaks.append(streak)
        streaks.extend(group_streaks)
    df["consecutive_deficits"] = streaks
    return df


# ── Main assembly function ─────────────────────────────────────────────────────

def build_panel(
    years: list[int],
    out_path: Path,
    auto_download: bool = False,
    min_years_per_org: int = 3,
) -> pd.DataFrame:
    """
    Assemble a BEACON-ready panel from NCCS Core annual files.

    Parameters
    ----------
    years            : fiscal years to include
    out_path         : where to write the output CSV
    auto_download    : attempt to download files if not cached locally
    min_years_per_org: drop orgs with fewer than this many years (sparse histories
                       make forward-looking labels and consecutive_deficits unreliable)
    """
    frames = []
    for year in sorted(years):
        path = local_path(year)
        if not path.exists():
            if auto_download:
                try:
                    path = download_year(year)
                except RuntimeError as exc:
                    print(f"  SKIP FY{year}: {exc}")
                    continue
            else:
                print(
                    f"  SKIP FY{year}: {path} not found. "
                    f"Download from https://nccs.urban.org/nccs/datasets/core/ "
                    f"or re-run with --auto-download"
                )
                continue
        try:
            df_year = load_year(path, year)
            frames.append(df_year)
        except Exception as exc:
            print(f"  ERROR FY{year}: {exc}")

    if not frames:
        raise RuntimeError(
            "No NCCS data loaded. Download annual files to data/raw/nccs/ "
            "from https://nccs.urban.org/nccs/datasets/core/"
        )

    print(f"\nStacking {len(frames)} annual files…")
    panel = pd.concat(frames, ignore_index=True)
    panel["ein"] = panel["ein"].astype(str).str.strip().str.zfill(9)

    # Drop orgs with too few years for reliable features and labels
    org_counts = panel.groupby("ein")["fiscal_year"].count()
    keep_eins = org_counts[org_counts >= min_years_per_org].index
    panel = panel[panel["ein"].isin(keep_eins)].copy()
    print(f"  {len(panel):,} rows for {panel['ein'].nunique():,} orgs "
          f"(>= {min_years_per_org} years each)")

    # Derive all BEACON features
    print("Deriving BEACON feature columns…")
    panel = derive_features(panel)

    # Forward-looking distress labels (same logic as synthetic pipeline)
    print("Computing forward-looking distress labels…")
    panel = compute_labels(panel)

    # Keep only columns the cleaning pipeline and models need
    keep_cols = [
        "ein", "ntee_code", "fiscal_year",
        "total_revenue", "total_expenses",
        "months_cash_on_hand", "current_ratio",
        "unrestricted_net_assets", "unrestricted_net_assets_ratio",
        "total_net_assets", "total_liabilities",
        "operating_margin", "consecutive_deficits",
        "gov_grant_concentration", "revenue_hhi", "debt_to_equity",
        "financial_distress", "financial_distress_2",
        "label_year_t1", "label_year_t2",
    ]
    out_cols = [c for c in keep_cols if c in panel.columns]
    panel = panel[out_cols].sort_values(["ein", "fiscal_year"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    rate1 = panel["financial_distress"].mean()
    print(f"\nSaved {len(panel):,} labeled rows for {panel['ein'].nunique():,} orgs → {out_path}")
    print(f"Option 1 distress rate: {rate1:.1%}")
    if "financial_distress_2" in panel.columns and panel["financial_distress_2"].notna().any():
        rate2 = panel["financial_distress_2"].mean()
        print(f"Option 2 distress rate: {rate2:.1%}")
    print(f"Fiscal years: {sorted(panel['fiscal_year'].unique())}")
    return panel


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build BEACON panel from NCCS Core annual files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=YEARS_AVAILABLE,
        help="Fiscal years to include (default: 2013–2023)",
    )
    parser.add_argument(
        "--out", default="data/raw/nccs_panel.csv",
        help="Output path for assembled panel CSV",
    )
    parser.add_argument(
        "--auto-download", action="store_true",
        help="Attempt to download NCCS files if not cached locally",
    )
    parser.add_argument(
        "--min-years", type=int, default=3,
        help="Minimum years per org to retain (default: 3)",
    )
    args = parser.parse_args()
    build_panel(
        years=args.years,
        out_path=Path(args.out),
        auto_download=args.auto_download,
        min_years_per_org=args.min_years,
    )

"""
NCCS Core Panel — data collection script for BEACON.

Downloads pre-cleaned annual Form 990 panel files from the Urban Institute's
National Center for Charitable Statistics (NCCS) and assembles a longitudinal
dataset for NTEE Categories L and P spanning FY2013–2023.

NCCS Core files are available at:
  https://nccs-data.urban.org/data.php?ds=core

Each annual file (e.g., nccs.core2019pc.csv.gz) covers 501(c)(3) public
charities and contains ~200 financial variables derived from Part VIII, IX, X.

Usage:
    python -m src.ingestion.collect_nccs \
        --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
        --out data/raw/nccs_panel.csv

Note: NCCS files can be large (200–400 MB per year). The script downloads
them to data/raw/nccs/ and caches locally so re-runs are fast.
"""

import argparse
from pathlib import Path

import pandas as pd
import requests


NCCS_BASE = "https://nccs-data.urban.org/dl.php?f=core"
NCCS_LOCAL_DIR = Path("data/raw/nccs")

NTEE_MAJOR_GROUPS = {"L", "P"}

# NCCS column → BEACON column mapping
COLUMN_MAP = {
    "EIN": "ein",
    "NTEECC": "ntee_code",
    "FISYR": "fiscal_year",
    # Revenue & expenses (Part VIII / IX)
    "TOTREV": "total_revenue",
    "EXPS": "total_expenses",
    # Balance sheet (Part X)
    "ASSET": "total_assets",
    "LIAB": "total_liabilities",
    "NETASSET": "total_net_assets",
    # Unrestricted net assets (Part X Line 27)
    "UNRESTRICTED": "unrestricted_net_assets",
    # Government grants (Part VIII Line 1e + 1h)
    "GOVGRANTS": "gov_grant_concentration",  # will be converted to ratio below
    "CONT": "total_contributions",
    # Cash (Part X Lines 1–2 combined)
    "CASH": "cash",
    # Current assets / liabilities (Part X Lines 1–5 / 17)
    "CASSETS": "current_assets",
    "CLIAB": "current_liabilities",
}

YEARS_AVAILABLE = list(range(2013, 2024))


def _nccs_file_url(year: int) -> str:
    return f"{NCCS_BASE}{year}pc.csv.gz"


def _nccs_local_path(year: int) -> Path:
    return NCCS_LOCAL_DIR / f"nccs_core_{year}.csv.gz"


def download_year(year: int) -> Path:
    local = _nccs_local_path(year)
    if local.exists():
        print(f"  {year}: using cached {local}")
        return local
    url = _nccs_file_url(year)
    print(f"  {year}: downloading {url} …")
    NCCS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(local, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"        → saved to {local}")
    return local


def load_and_filter_year(local_path: Path, year: int) -> pd.DataFrame:
    """Read one NCCS annual file, keep NTEE L/P, rename columns, compute ratios."""
    raw = pd.read_csv(local_path, compression="gzip", low_memory=False)
    raw.columns = raw.columns.str.upper()

    # Filter to NTEE L and P (first character of NTEECC major group)
    if "NTEECC" not in raw.columns:
        print(f"  WARNING: NTEECC column missing in {local_path.name}")
        return pd.DataFrame()

    mask = raw["NTEECC"].astype(str).str[0].isin(NTEE_MAJOR_GROUPS)
    sub = raw[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    # Rename to BEACON column names
    rename = {k: v for k, v in COLUMN_MAP.items() if k in sub.columns}
    sub = sub.rename(columns=rename)

    # Derive ntee_code major group (single letter)
    sub["ntee_code"] = sub["ntee_code"].astype(str).str[0]

    # Fiscal year
    if "fiscal_year" not in sub.columns:
        sub["fiscal_year"] = year

    # Compute operating margin
    if "total_revenue" in sub.columns and "total_expenses" in sub.columns:
        rev = pd.to_numeric(sub["total_revenue"], errors="coerce")
        exp = pd.to_numeric(sub["total_expenses"], errors="coerce")
        sub["operating_margin"] = (rev - exp) / rev.replace(0, float("nan"))

    # Convert government grants to concentration ratio
    if "gov_grant_concentration" in sub.columns and "total_contributions" in sub.columns:
        gov = pd.to_numeric(sub["gov_grant_concentration"], errors="coerce")
        cont = pd.to_numeric(sub["total_contributions"], errors="coerce")
        sub["gov_grant_concentration"] = gov / cont.replace(0, float("nan"))

    # Derive current ratio
    if "current_assets" in sub.columns and "current_liabilities" in sub.columns:
        ca = pd.to_numeric(sub["current_assets"], errors="coerce")
        cl = pd.to_numeric(sub["current_liabilities"], errors="coerce")
        sub["current_ratio"] = ca / cl.replace(0, float("nan"))

    # Months of cash on hand: Cash / (Total Expenses / 12)
    if "cash" in sub.columns and "total_expenses" in sub.columns:
        cash = pd.to_numeric(sub["cash"], errors="coerce")
        exp = pd.to_numeric(sub["total_expenses"], errors="coerce")
        monthly_exp = exp / 12
        sub["months_cash_on_hand"] = cash / monthly_exp.replace(0, float("nan"))

    # Unrestricted net assets ratio
    if "unrestricted_net_assets" in sub.columns and "total_net_assets" in sub.columns:
        una = pd.to_numeric(sub["unrestricted_net_assets"], errors="coerce")
        tna = pd.to_numeric(sub["total_net_assets"], errors="coerce")
        sub["unrestricted_net_assets_ratio"] = una / tna.replace(0, float("nan"))

    # Debt-to-equity
    if "total_liabilities" in sub.columns and "total_net_assets" in sub.columns:
        liab = pd.to_numeric(sub["total_liabilities"], errors="coerce")
        tna = pd.to_numeric(sub["total_net_assets"], errors="coerce")
        sub["debt_to_equity"] = liab / tna.replace(0, float("nan"))

    # Keep only columns relevant to BEACON (plus EIN/year identifiers)
    keep = [
        "ein", "ntee_code", "fiscal_year",
        "total_revenue", "total_expenses", "operating_margin",
        "months_cash_on_hand", "current_ratio",
        "unrestricted_net_assets", "unrestricted_net_assets_ratio",
        "total_net_assets", "total_liabilities",
        "gov_grant_concentration", "debt_to_equity",
        "cash", "current_assets", "current_liabilities",
    ]
    available = [c for c in keep if c in sub.columns]
    return sub[available].reset_index(drop=True)


def collect(
    years: list[int],
    out_path: Path,
) -> pd.DataFrame:
    frames = []
    for year in sorted(years):
        print(f"Processing FY{year}…")
        try:
            local = download_year(year)
            df_year = load_and_filter_year(local, year)
            print(f"  FY{year}: {len(df_year):,} NTEE L/P records")
            frames.append(df_year)
        except Exception as exc:
            print(f"  FY{year}: FAILED — {exc}")

    if not frames:
        print("No data collected.")
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ein", "fiscal_year"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    print(
        f"\nSaved {len(panel):,} records for {panel['ein'].nunique():,} organizations "
        f"across {panel['fiscal_year'].nunique()} fiscal years to {out_path}"
    )
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect NCCS Core panel data for BEACON")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS_AVAILABLE)
    parser.add_argument("--out", default="data/raw/nccs_panel.csv")
    args = parser.parse_args()
    collect(years=args.years, out_path=Path(args.out))

"""
GivingTuesday 990 Data Lake collector for BEACON.

Pulls real IRS e-file 990 data for NTEE L (Housing & Shelter) and
P (Human Services) organizations from the GivingTuesday public S3 bucket.
No AWS account or credentials required (--no-sign-request).

HOW IT WORKS
------------
1. Download the GivingTuesday index (parquet) — lists every e-filed 990
   with EIN, TaxYear, FormType, and a direct URL to the XML.
2. Cross-reference with the IRS Business Master File (BMF) to get NTEE
   codes, since NTEE is not embedded in the XML files themselves.
3. Filter index to FormType=990, TaxYear 2013–2025, NTEE L or P.
4. Stream-parse each XML to extract BEACON feature fields.
5. Compute consecutive_deficits (second pass) and forward-looking
   labels via compute_labels().

BMF SOURCE
----------
The IRS publishes the Exempt Organizations Business Master File as a set
of CSV files (one per state/region) at:
  https://www.irs.gov/charities-non-profits/exempt-organizations-business-master-file-extract-eo-bmf

Download all ~6 files manually from that page and pass them via
--bmf-files, OR let this script download them automatically (default).
The combined file is ~300MB and is filtered in-memory to NTEE L/P.

XML FIELD MAPPING (IRS e-file schema 2013+)
-------------------------------------------
All monetary amounts are nested in groups:
  CashNonInterestBearingGrp/EOYAmt   — Part X Line 1
  SavingsAndTempCashInvstGrp/EOYAmt  — Part X Line 2
  NoDonorRestrictionNetAssetsGrp/EOYAmt  — post-2018 (ASU 2016-14)
  UnrestrictedNetAssetsAmt            — pre-2018 (direct field)
  AccountsPayableAccrExpnssGrp/EOYAmt — Part X Line 17
  DeferredRevenueGrp/EOYAmt          — Part X Line 19

USAGE
-----
# Full collection (downloads BMF automatically, ~2-4 hrs for all NTEE L/P):
python -m src.ingestion.collect_gt990 \\
    --out data/raw/gt990_panel.csv

# Limit to N organizations for testing:
python -m src.ingestion.collect_gt990 \\
    --max-orgs 500 \\
    --out data/raw/gt990_sample.csv

# Use pre-downloaded BMF files:
python -m src.ingestion.collect_gt990 \\
    --bmf-files data/raw/bmf/eo1.csv data/raw/bmf/eo2.csv \\
    --out data/raw/gt990_panel.csv

# Then pass to BEACON pipeline:
python run_beacon.py --real-data data/raw/gt990_panel.csv
"""

import argparse
import io
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.features.labeling import compute_labels


# ── Constants ─────────────────────────────────────────────────────────────────

S3_INDEX = (
    "https://gt990datalake-rawdata.s3.amazonaws.com/Indices/990xmls/"
    "index_all_years_efiledata_xmls_created_on_2026-06-04.parquet"
)

# IRS BMF — hosted in the GivingTuesday data lake (same S3 bucket, no auth needed)
S3_BMF = (
    "s3://gt990datalake-rawdata/EfileData/BMF/IRS_BMF_2025_12_23_raw.csv"
)
S3_BMF_HTTP = (
    "https://gt990datalake-rawdata.s3.amazonaws.com/EfileData/BMF/"
    "IRS_BMF_2025_12_23_raw.csv"
)

IRS_NS = "http://www.irs.gov/efile"  # XML namespace for all IRS e-file returns

NTEE_TARGET = {"L", "P"}
FORM_TYPE = "990"
MIN_YEAR = 2013
MAX_YEAR = 2025

# Workers for parallel XML downloads — stay well below S3 rate limits
MAX_WORKERS = 8
REQUEST_DELAY = 0.1   # seconds between requests per worker


# ── BMF loading ───────────────────────────────────────────────────────────────

def load_bmf(bmf_files: list[str] | None = None, bmf_cache: str | None = None) -> pd.DataFrame:
    """
    Return DataFrame with columns [ein, ntee_code] for NTEE L and P only.

    Parameters
    ----------
    bmf_files : local CSV path(s) to pre-downloaded BMF file(s). If None,
                downloads the consolidated BMF from the GivingTuesday S3 bucket
                (same public bucket as the 990 data, no credentials needed).
    bmf_cache : local path to cache the downloaded BMF (avoids re-download).
    """
    if bmf_files:
        print(f"Loading {len(bmf_files)} local BMF file(s)…")
        frames = []
        for f in bmf_files:
            try:
                df = pd.read_csv(f, usecols=["EIN", "NTEE_CD"], dtype=str)
                df.columns = ["ein", "ntee_code"]
                frames.append(df)
            except Exception as exc:
                print(f"  Warning: {f}: {exc}")
        if not frames:
            raise RuntimeError(f"Could not load any BMF files from: {bmf_files}")
        bmf_raw = pd.concat(frames, ignore_index=True)
    else:
        cache_path = Path(bmf_cache) if bmf_cache else None
        if cache_path and cache_path.exists():
            print(f"Loading BMF from cache: {cache_path}")
            bmf_raw = pd.read_csv(cache_path, usecols=["EIN", "NTEE_CD"], dtype=str)
        else:
            print(f"Downloading IRS BMF from GivingTuesday S3 (~317 MB, one-time)…")
            print(f"  {S3_BMF_HTTP}")
            import subprocess
            dest = str(cache_path) if cache_path else "/tmp/beacon_bmf.csv"
            result = subprocess.run(
                ["aws", "s3", "cp", S3_BMF, dest, "--no-sign-request"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"BMF download failed: {result.stderr}\n"
                    "Install AWS CLI: pip install awscli\n"
                    "Or pass a local file via --bmf-files."
                )
            bmf_raw = pd.read_csv(dest, usecols=["EIN", "NTEE_CD"], dtype=str)
        bmf_raw.columns = ["ein", "ntee_code"]

    bmf_raw["ein"] = bmf_raw["ein"].astype(str).str.strip().str.zfill(9)
    bmf_raw["ntee_major"] = bmf_raw["ntee_code"].fillna("").str[0].str.upper()
    bmf = bmf_raw[bmf_raw["ntee_major"].isin(NTEE_TARGET)].drop_duplicates("ein").copy()
    print(f"BMF: {len(bmf):,} NTEE L/P organizations "
          f"(L={( bmf['ntee_major']=='L').sum():,}, P={(bmf['ntee_major']=='P').sum():,}).")
    return bmf[["ein", "ntee_major"]].rename(columns={"ntee_major": "ntee_code"})


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index(local_path: str | None = None) -> pd.DataFrame:
    """
    Load the GivingTuesday index parquet and filter to full 990 filers,
    years 2013–2025. Returns DataFrame with columns [EIN, TaxYear, URL].
    """
    if local_path and Path(local_path).exists():
        print(f"Loading index from {local_path}…")
        idx = pd.read_parquet(local_path, columns=["EIN", "TaxYear", "FormType", "URL"])
    else:
        print(f"Downloading GivingTuesday index (~1.4 GB parquet)…")
        print(f"  Source: {S3_INDEX}")
        print("  This is a one-time download. Use --index-cache to save locally.")
        idx = pd.read_parquet(S3_INDEX, columns=["EIN", "TaxYear", "FormType", "URL"])

    idx = idx[
        (idx["FormType"] == FORM_TYPE) &
        (idx["TaxYear"].between(str(MIN_YEAR), str(MAX_YEAR)))
    ].copy()
    idx["EIN"] = idx["EIN"].astype(str).str.strip().str.zfill(9)
    idx["TaxYear"] = idx["TaxYear"].astype(int)
    print(f"Index: {len(idx):,} full-990 filings, {MIN_YEAR}–{MAX_YEAR}.")
    return idx


# ── XML parser ────────────────────────────────────────────────────────────────

def _tag(name: str) -> str:
    return f"{{{IRS_NS}}}{name}"


def _txt(node, *path: str) -> str | None:
    """Navigate a chain of child tags and return text, or None."""
    cur = node
    for p in path:
        if cur is None:
            return None
        cur = cur.find(_tag(p))
    return cur.text if cur is not None else None


def _grp_eoy(irs990: ET.Element, group: str) -> float | None:
    """Return EOYAmt from a Part X balance-sheet group, or None."""
    grp = irs990.find(_tag(group))
    if grp is None:
        return None
    node = grp.find(_tag("EOYAmt"))
    return _safe_float(node.text) if node is not None else None


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_pos(val) -> float:
    v = _safe_float(val)
    return max(v, 0.0) if v is not None else 0.0


def parse_xml(ein: str, ntee_code: str, xml_bytes: bytes) -> dict | None:
    """
    Parse one IRS e-file 990 XML and return a BEACON feature record.
    Returns None if the record is unusable (no revenue, parse error).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    header = root.find(_tag("ReturnHeader"))
    rd     = root.find(_tag("ReturnData"))
    if rd is None:
        return None
    irs990 = rd.find(_tag("IRS990"))
    if irs990 is None:
        return None

    # Fiscal year from header
    tax_yr = _txt(header, "TaxYr")
    year = _safe_float(tax_yr)
    if year is None:
        return None
    year = int(year)

    # ── Revenue & expenses (Part VIII / IX) ──────────────────────────────────
    total_revenue  = _safe_float(_txt(irs990, "CYTotalRevenueAmt"))
    total_expenses = _safe_float(_txt(irs990, "CYTotalExpensesAmt"))

    if total_revenue is None or total_revenue <= 0:
        return None

    operating_margin = (
        (total_revenue - total_expenses) / total_revenue
        if total_expenses is not None else None
    )

    # ── Balance sheet (Part X) ────────────────────────────────────────────────
    total_assets      = _safe_float(_txt(irs990, "TotalAssetsEOYAmt"))
    total_liabilities = _safe_float(_txt(irs990, "TotalLiabilitiesEOYAmt"))
    total_net_assets  = _safe_float(_txt(irs990, "NetAssetsOrFundBalancesEOYAmt"))
    if total_net_assets is None and total_assets is not None and total_liabilities is not None:
        total_net_assets = total_assets - total_liabilities

    # Unrestricted net assets — ASU 2016-14 aware
    # Post-2018: NoDonorRestrictionNetAssetsGrp/EOYAmt
    # Pre-2018:  UnrestrictedNetAssetsAmt (direct field)
    una = _grp_eoy(irs990, "NoDonorRestrictionNetAssetsGrp")
    if una is None:
        una = _safe_float(_txt(irs990, "UnrestrictedNetAssetsAmt"))

    # ── Liquidity (Part X current items) ─────────────────────────────────────
    cash   = _grp_eoy(irs990, "CashNonInterestBearingGrp")
    saving = _grp_eoy(irs990, "SavingsAndTempCashInvstGrp")
    cash_total = (cash or 0.0) + (saving or 0.0)

    # Current assets = cash + savings + pledges receivable + accounts receivable + prepaid
    pledges     = _grp_eoy(irs990, "PledgesAndGrantsReceivableNetGrp")
    ar          = _grp_eoy(irs990, "AccountsReceivableNetGrp")
    prepaid     = _grp_eoy(irs990, "PrepaidExpensesDefrdChargesGrp")
    current_assets = (
        (cash or 0.0) + (saving or 0.0) +
        (pledges or 0.0) + (ar or 0.0) + (prepaid or 0.0)
    ) or None

    # Current liabilities = accounts payable + deferred revenue + grants payable
    ap       = _grp_eoy(irs990, "AccountsPayableAccrExpnssGrp")
    deferred = _grp_eoy(irs990, "DeferredRevenueGrp")
    grants_p = _grp_eoy(irs990, "GrantsPayableGrp")
    current_liabilities = (
        (ap or 0.0) + (deferred or 0.0) + (grants_p or 0.0)
    ) or None

    months_cash_on_hand = None
    if cash_total > 0 and total_expenses and total_expenses > 0:
        months_cash_on_hand = cash_total / (total_expenses / 12)

    current_ratio = None
    if current_assets and current_liabilities and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities

    # ── Revenue concentration HHI (Part VIII) ────────────────────────────────
    gov_grants        = _safe_pos(_txt(irs990, "GovernmentGrantsAmt"))
    total_contribs    = _safe_pos(_txt(irs990, "CYContributionsGrantsAmt"))
    prog_rev          = _safe_pos(_txt(irs990, "CYProgramServiceRevenueAmt"))
    invest_inc        = _safe_pos(_txt(irs990, "CYInvestmentIncomeAmt"))
    other_rev         = _safe_pos(_txt(irs990, "CYOtherRevenueAmt"))
    priv_contrib      = max(total_contribs - gov_grants, 0.0)
    # Recompute other so streams sum to total_revenue
    other_rev = max(total_revenue - gov_grants - priv_contrib - prog_rev - invest_inc, 0.0)

    stream_total = gov_grants + priv_contrib + prog_rev + invest_inc + other_rev
    if stream_total > 0:
        shares = [s / stream_total for s in
                  [gov_grants, priv_contrib, prog_rev, invest_inc, other_rev]]
        revenue_hhi = sum(s ** 2 for s in shares)
    else:
        revenue_hhi = None

    gov_grant_concentration = (
        gov_grants / total_contribs if total_contribs > 0 else None
    )

    # ── Ratios ────────────────────────────────────────────────────────────────
    tna_denom = total_net_assets if total_net_assets and abs(total_net_assets) > 1 else None
    una_ratio = (
        una / tna_denom if una is not None and tna_denom else None
    )
    debt_to_equity = (
        total_liabilities / tna_denom
        if total_liabilities is not None and tna_denom else None
    )

    # Program expense ratio (Part IX)
    prog_exp = _safe_float(_txt(irs990, "TotalProgramServiceExpensesAmt"))
    program_expense_ratio = (
        prog_exp / total_expenses
        if prog_exp is not None and total_expenses and total_expenses > 0 else None
    )

    total_revenue_log = float(np.log(max(total_revenue, 1)))

    return {
        "ein":                          ein,
        "ntee_code":                    ntee_code,
        "fiscal_year":                  year,
        "total_revenue":                total_revenue,
        "total_expenses":               total_expenses,
        "operating_margin":             operating_margin,
        "total_net_assets":             total_net_assets,
        "total_liabilities":            total_liabilities,
        "unrestricted_net_assets":      una,
        "unrestricted_net_assets_ratio": una_ratio,
        "months_cash_on_hand":          months_cash_on_hand,
        "current_ratio":                current_ratio,
        "gov_grant_concentration":      gov_grant_concentration,
        "revenue_hhi":                  revenue_hhi,
        "debt_to_equity":               debt_to_equity,
        "program_expense_ratio":        program_expense_ratio,
        "total_revenue_log":            total_revenue_log,
    }


# ── XML downloader ────────────────────────────────────────────────────────────

def _fetch_url(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "BEACON/1.0 academic research"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def _process_filing(row: dict) -> dict | None:
    """Download and parse one filing. Called in thread pool."""
    xml_bytes = _fetch_url(row["url"])
    if xml_bytes is None:
        return None
    time.sleep(REQUEST_DELAY)
    return parse_xml(row["ein"], row["ntee_code"], xml_bytes)


# ── Consecutive deficits (second pass) ───────────────────────────────────────

def _compute_consecutive_deficits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ein", "fiscal_year"]).copy()
    streaks = []
    for _, group in df.groupby("ein", sort=False):
        streak = 0
        for margin in group["operating_margin"]:
            if pd.isna(margin) or margin >= 0:
                streak = 0
            else:
                streak += 1
            streaks.append(streak)
    df["consecutive_deficits"] = streaks
    return df


# ── Main collection function ──────────────────────────────────────────────────

def collect(
    out_path: Path,
    bmf_files: list[str] | None = None,
    bmf_cache: str | None = None,
    index_cache: str | None = None,
    max_orgs: int | None = None,
    max_workers: int = MAX_WORKERS,
    checkpoint: bool = True,
) -> pd.DataFrame:
    """
    Collect IRS e-file 990 data for NTEE L/P and build a BEACON-ready panel.

    Parameters
    ----------
    out_path    : destination CSV path
    bmf_files   : local IRS BMF CSV file(s); if None, downloads from S3
    bmf_cache   : local path to cache the downloaded BMF (avoids re-download)
    index_cache : local path to save/load the GT index parquet (~1.4 GB)
    max_orgs    : cap on number of unique EINs to process (None = all)
    max_workers : parallel XML download threads
    checkpoint  : save progress every 500 records so a restart can resume
    """
    checkpoint_path = Path(str(out_path).replace(".csv", "_checkpoint.csv"))

    # Step 1: load BMF → EIN→NTEE mapping
    bmf = load_bmf(bmf_files, bmf_cache)

    # Step 2: load GT index → list of 990 filings with URLs
    idx = load_index(index_cache)

    # Step 3: merge to get only NTEE L/P filings
    idx = idx.merge(bmf, left_on="EIN", right_on="ein", how="inner")
    print(f"After NTEE filter: {len(idx):,} filings for {idx['EIN'].nunique():,} orgs.")

    # Optional org cap (take all filings for the selected orgs)
    if max_orgs is not None:
        selected_eins = idx["EIN"].unique()[:max_orgs]
        idx = idx[idx["EIN"].isin(selected_eins)]
        print(f"Capped to {max_orgs:,} orgs -> {len(idx):,} filings.")

    # Step 4: resume from checkpoint if one exists
    rows = []
    done_urls: set[str] = set()
    if checkpoint and checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path, dtype=str)
        rows = prior.to_dict("records")
        done_urls = set(prior.get("_url", pd.Series(dtype=str)).dropna())
        print(f"Resuming from checkpoint: {len(rows):,} records already collected, "
              f"{len(done_urls):,} URLs skipped.")

    tasks = [
        {"ein": row["EIN"], "ntee_code": row["ntee_code"], "url": row["URL"]}
        for _, row in idx.iterrows()
        if row["URL"] not in done_urls
    ]

    n_tasks = len(tasks)
    n_total = len(idx)
    print(f"\nDownloading and parsing {n_tasks:,} XML files "
          f"({n_total - n_tasks:,} already done, {max_workers} workers)...")

    def _process_with_url(task: dict) -> dict | None:
        result = _process_filing(task)
        if result:
            result["_url"] = task["url"]
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_with_url, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                rows.append(result)
            if i % 500 == 0 or i == n_tasks:
                print(f"  {i:,}/{n_tasks:,} done — {len(rows):,} valid records")
                if checkpoint and rows:
                    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    if not rows:
        print("No records collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Drop the internal tracking column before saving
    df = df.drop(columns=["_url"], errors="ignore")
    df["ein"] = df["ein"].astype(str).str.strip().str.zfill(9)

    # Deduplicate — keep most complete row per (EIN, year)
    df = (
        df.sort_values("operating_margin", na_position="last")
        .drop_duplicates(subset=["ein", "fiscal_year"], keep="first")
    )

    # Second pass: consecutive_deficits requires full EIN history
    df = _compute_consecutive_deficits(df)

    # Forward-looking distress labels
    df = compute_labels(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Remove checkpoint now that final file is written
    if checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    rate1 = df["financial_distress"].mean()
    print(f"\nSaved {len(df):,} labeled rows for {df['ein'].nunique():,} orgs -> {out_path}")
    print(f"Option 1 distress rate: {rate1:.1%}")
    if df["financial_distress_2"].notna().any():
        print(f"Option 2 distress rate: {df['financial_distress_2'].mean():.1%}")
    print(f"Fiscal years: {sorted(df['fiscal_year'].unique())}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect GivingTuesday 990 data lake for BEACON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", default="data/raw/gt990_panel.csv",
        help="Output CSV path (default: data/raw/gt990_panel.csv)",
    )
    parser.add_argument(
        "--bmf-files", nargs="+", metavar="CSV",
        help="Local IRS BMF CSV file(s). If omitted, downloads consolidated BMF "
             "from the GivingTuesday S3 bucket automatically (~317 MB).",
    )
    parser.add_argument(
        "--bmf-cache", metavar="CSV",
        help="Local path to cache/load the downloaded BMF CSV (avoids re-download).",
    )
    parser.add_argument(
        "--index-cache", metavar="PARQUET",
        help="Local path to cache/load the GivingTuesday index parquet (~1.4 GB).",
    )
    parser.add_argument(
        "--max-orgs", type=int, default=None,
        help="Cap on unique orgs to process (useful for testing).",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Parallel download workers (default: {MAX_WORKERS}).",
    )
    args = parser.parse_args()

    collect(
        out_path=Path(args.out),
        bmf_files=args.bmf_files,
        bmf_cache=args.bmf_cache,
        index_cache=args.index_cache,
        max_orgs=args.max_orgs,
        max_workers=args.workers,
    )

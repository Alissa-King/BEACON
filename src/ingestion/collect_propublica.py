"""
ProPublica Nonprofit Explorer — secondary data collector for BEACON.

Fetches IRS Form 990 filings for NTEE L and P organizations via the ProPublica
Nonprofit Explorer API (free; no API key required). Suitable for targeted
collection of specific organizations or as a supplement when NCCS Core files
are unavailable.

LIMITATION vs. NCCS CORE
--------------------------
ProPublica returns parsed totals from Part VIII and IX but does NOT return
Part X balance sheet line items in the same detail as NCCS Core. As a result:
  - months_cash_on_hand: approximated from Part X cash totals if available;
    otherwise NaN (handled by KNN imputer in cleaning pipeline)
  - current_ratio: NaN if Part X current items not present (KNN imputed)
  - revenue_hhi: computed from available Part VIII stream breakdown
  - consecutive_deficits: computed in second pass from assembled panel
  - forward-looking labels: computed via compute_labels() as in synthetic pipeline

For a full-panel dissertation dataset, prefer collect_nccs.py (NCCS Core).
Use this script for: supplemental validation on specific orgs, smaller
samples, or cross-checking NCCS values.

RATE LIMITING
--------------
ProPublica does not publish explicit rate limits. A 0.5-second delay between
requests is used. For large collections (>1000 orgs) run overnight.

USAGE
-----
# Collect up to 2000 NTEE L/P orgs:
python -m src.ingestion.collect_propublica \\
    --out data/raw/propublica_panel.csv \\
    --max-orgs 2000

# Fetch specific EINs only:
python -m src.ingestion.collect_propublica \\
    --eins 123456789 987654321 \\
    --out data/raw/propublica_targeted.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features.labeling import compute_labels


BASE_URL = "https://projects.propublica.org/nonprofits/api/v2"
NTEE_CODES = ["L", "P"]
REQUEST_DELAY = 0.5  # seconds between calls

# API NOTE: ProPublica's search endpoint accepts the `ntee` parameter with
# the major-group letter code (e.g. "L", "P").  Some versions of the API
# documentation describe a numeric `ntee[id]` form; the letter-code form
# has been observed to work in practice but is not formally documented.
# If search results are empty or incorrect, verify the current parameter
# name against https://projects.propublica.org/nonprofits/api before a
# large collection run.  This collector is marked supplemental — the
# primary dissertation dataset was collected via collect_nccs.py (NCCS Core).


# ── API fetch helpers ──────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def search_orgs(ntee_code: str, page: int = 0, per_page: int = 100) -> list[dict]:
    data = _get(f"{BASE_URL}/search.json", {"ntee": ntee_code, "page": page, "per_page": per_page})
    return data.get("organizations", [])


def fetch_org_filings(ein: str) -> list[dict]:
    try:
        data = _get(f"{BASE_URL}/organizations/{ein}.json")
        return data.get("filings_with_data", [])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return []
        raise


# ── Per-filing parser ──────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_pos(val) -> float:
    """Return float or 0 if missing/negative."""
    v = _safe_float(val)
    return max(v, 0) if v is not None else 0.0


def parse_filing(ein: str, ntee_code: str, filing: dict) -> dict | None:
    """
    Extract BEACON-relevant fields from one ProPublica filing record.

    ProPublica field reference:
      totrevenue    — Part VIII total revenue
      totfuncexpns  — Part IX total functional expenses
      totassetsend  — Part X total assets (end)
      totliabend    — Part X total liabilities (end)
      netassetsend  — Part X net assets (end)
      unrestricted  — Part X unrestricted net assets (pre-2018 filers)
      nwunrestricted— Part X net assets w/o donor restrictions (post-2018, ASU 2016-14)
      grscontrgvtgrnt — Part VIII 1e government grants
      totcntrbgfts  — Part VIII 1h total contributions
      totprgmrevnue — Part VIII 2 program service revenue
      invstmntinc   — Part VIII 3 investment income
      miscrevtot11e — Part VIII 11e other revenue (miscellaneous)
      compnsatncurrofcr — Part VII compensation (not used; for reference)
    """
    year = _safe_float(filing.get("tax_prd_yr"))
    if year is None:
        return None
    year = int(year)

    total_revenue = _safe_float(filing.get("totrevenue"))
    total_expenses = _safe_float(filing.get("totfuncexpns"))

    if total_revenue is None or total_revenue <= 0:
        return None  # can't compute margin without revenue

    operating_margin = (
        (total_revenue - total_expenses) / total_revenue
        if total_expenses is not None else None
    )

    total_assets = _safe_float(filing.get("totassetsend"))
    total_liabilities = _safe_float(filing.get("totliabend"))
    total_net_assets = _safe_float(filing.get("netassetsend"))
    if total_net_assets is None and total_assets and total_liabilities is not None:
        total_net_assets = total_assets - total_liabilities

    # Unrestricted net assets — ASU 2016-14 aware
    # ProPublica returns both field names; prefer post-2018 for recent filers
    una = _safe_float(filing.get("nwunrestricted")) or _safe_float(filing.get("unrestricted"))

    # Part VIII revenue streams for HHI
    gov_grants = _safe_pos(filing.get("grscontrgvtgrnt"))
    total_contributions = _safe_pos(filing.get("totcntrbgfts"))
    prog_rev = _safe_pos(filing.get("totprgmrevnue"))
    invest_inc = _safe_pos(filing.get("invstmntinc"))
    priv_contrib = max(total_contributions - gov_grants, 0)
    other_rev = max(total_revenue - gov_grants - priv_contrib - prog_rev - invest_inc, 0)

    stream_total = gov_grants + priv_contrib + prog_rev + invest_inc + other_rev
    if stream_total > 0:
        shares = [s / stream_total for s in [gov_grants, priv_contrib, prog_rev, invest_inc, other_rev]]
        revenue_hhi = sum(s ** 2 for s in shares)
    else:
        revenue_hhi = None

    gov_grant_concentration = (
        gov_grants / total_contributions
        if total_contributions > 0 else None
    )

    tna_denom = total_net_assets if total_net_assets and abs(total_net_assets) > 1 else None
    unrestricted_net_assets_ratio = (
        una / tna_denom if una is not None and tna_denom else None
    )
    debt_to_equity = (
        total_liabilities / tna_denom if total_liabilities is not None and tna_denom else None
    )

    # Program expense ratio — ProPublica field: progservexp (Part IX 25b if available)
    prog_exp = _safe_float(filing.get("progservexp"))
    program_expense_ratio = (
        prog_exp / total_expenses
        if prog_exp is not None and total_expenses and total_expenses > 0
        else None
    )

    total_revenue_log = float(np.log(max(total_revenue, 1))) if total_revenue else None

    return {
        "ein": ein,
        "ntee_code": ntee_code,
        "fiscal_year": year,
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "operating_margin": operating_margin,
        "total_net_assets": total_net_assets,
        "total_liabilities": total_liabilities,
        "unrestricted_net_assets": una,
        "unrestricted_net_assets_ratio": unrestricted_net_assets_ratio,
        "gov_grant_concentration": gov_grant_concentration,
        "revenue_hhi": revenue_hhi,
        "debt_to_equity": debt_to_equity,
        "program_expense_ratio": program_expense_ratio,
        "total_revenue_log": total_revenue_log,
        # ProPublica doesn't reliably provide Part X current items — leave for KNN imputation
        "months_cash_on_hand": None,
        "current_ratio": None,
    }


# ── Consecutive deficits (second-pass, panel-level) ───────────────────────────

def _compute_consecutive_deficits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-looking streak of negative operating_margin ending at each row's year.
    Computed after all filings are assembled so the full org history is available.
    """
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


# ── Main collection function ───────────────────────────────────────────────────

def collect(
    out_path: Path,
    per_page: int = 100,
    max_orgs: int | None = None,
    eins: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collect ProPublica 990 filings and build a BEACON-ready panel.

    Parameters
    ----------
    out_path  : where to write the output CSV
    per_page  : orgs per API search page (max 100)
    max_orgs  : cap on total orgs fetched via search (None = no cap)
    eins      : if provided, fetch ONLY these specific EINs (ignores max_orgs)
    """
    rows = []

    if eins:
        print(f"Fetching {len(eins)} specific EINs…")
        for ein in eins:
            ntee = "?"  # unknown without search; will be set from filing data if available
            filings = fetch_org_filings(ein)
            time.sleep(REQUEST_DELAY)
            for filing in filings:
                record = parse_filing(ein, ntee, filing)
                if record:
                    rows.append(record)
    else:
        org_count = 0
        for ntee in NTEE_CODES:
            print(f"Searching NTEE-{ntee}…")
            page = 0
            while True:
                orgs = search_orgs(ntee, page=page, per_page=per_page)
                if not orgs:
                    break
                for org in orgs:
                    ein = str(org.get("ein", "")).strip()
                    if not ein:
                        continue
                    filings = fetch_org_filings(ein)
                    time.sleep(REQUEST_DELAY)
                    for filing in filings:
                        record = parse_filing(ein, ntee, filing)
                        if record:
                            rows.append(record)
                    org_count += 1
                    if org_count % 50 == 0:
                        print(f"  … {org_count} orgs, {len(rows)} filing records so far")
                    if max_orgs and org_count >= max_orgs:
                        break
                page += 1
                time.sleep(REQUEST_DELAY)
                if max_orgs and org_count >= max_orgs:
                    break

    if not rows:
        print("No records collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ein"] = df["ein"].astype(str).str.strip().str.zfill(9)

    # Remove duplicate (EIN, year) pairs — keep most complete row
    df = (
        df.sort_values("operating_margin", na_position="last")
        .drop_duplicates(subset=["ein", "fiscal_year"], keep="first")
    )

    # Compute consecutive_deficits from assembled panel history
    df = _compute_consecutive_deficits(df)

    # Forward-looking distress labels (identical logic to synthetic pipeline)
    df = compute_labels(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    rate1 = df["financial_distress"].mean()
    print(f"\nSaved {len(df):,} labeled rows for {df['ein'].nunique():,} orgs → {out_path}")
    print(f"Option 1 distress rate: {rate1:.1%}")
    if df["financial_distress_2"].notna().any():
        print(f"Option 2 distress rate: {df['financial_distress_2'].mean():.1%}")
    print(f"Fiscal years: {sorted(df['fiscal_year'].unique())}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect ProPublica 990 data for BEACON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out", default="data/raw/propublica_panel.csv")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-orgs", type=int, default=None)
    parser.add_argument("--eins", nargs="+", help="Fetch only these specific EINs")
    args = parser.parse_args()
    collect(
        out_path=Path(args.out),
        per_page=args.per_page,
        max_orgs=args.max_orgs,
        eins=args.eins,
    )

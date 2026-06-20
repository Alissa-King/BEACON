"""
ProPublica Nonprofit Explorer API — data collection script for BEACON.

Fetches IRS Form 990 filings for NTEE Category L (Housing & Shelter) and
Category P (Human Services) organizations via the ProPublica API (no API key
required). Builds a longitudinal panel of raw financial fields that the
cleaning pipeline can subsequently process.

API reference: https://projects.propublica.org/nonprofits/api/
Rate limits:   No published hard limit; conservative 0.5 s delay between calls.

Usage:
    python -m src.ingestion.collect_propublica \
        --out data/raw/propublica_panel.csv \
        --per-page 100 \
        --max-orgs 5000

The output CSV contains one row per (EIN, fiscal_year) and includes the
columns required by the BEACON cleaning pipeline.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://projects.propublica.org/nonprofits/api/v2"
NTEE_CODES = ["L", "P"]
REQUEST_DELAY = 0.5  # seconds between API calls


def search_orgs(ntee_code: str, page: int = 0, per_page: int = 100) -> list[dict]:
    """Return a page of organization stubs for the given NTEE major group."""
    url = f"{BASE_URL}/search.json"
    params = {"ntee": ntee_code, "page": page, "per_page": per_page}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("organizations", [])


def fetch_org_filings(ein: str) -> list[dict]:
    """Return all available filings (partial 990 JSON) for a single EIN."""
    url = f"{BASE_URL}/organizations/{ein}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("filings_with_data", [])


def parse_filing(ein: str, ntee_code: str, filing: dict) -> dict | None:
    """
    Extract BEACON feature fields from a single ProPublica filing record.

    ProPublica returns pre-parsed totals from Part VIII and IX.  Fields not
    directly available (e.g., current assets/liabilities, unrestricted net
    assets breakdown) are left as NaN for the KNN imputer to handle.
    """
    year = filing.get("tax_prd_yr")
    if year is None:
        return None

    total_revenue = _safe_float(filing.get("totrevenue"))
    total_expenses = _safe_float(filing.get("totfuncexpns"))

    if total_revenue is None or total_revenue <= 0:
        return None

    operating_margin = (
        (total_revenue - total_expenses) / total_revenue
        if total_expenses is not None else None
    )

    total_assets = _safe_float(filing.get("totassetsend"))
    total_liabilities = _safe_float(filing.get("totliabend"))
    total_net_assets = (
        total_assets - total_liabilities
        if total_assets is not None and total_liabilities is not None
        else None
    )

    # Government grants: Part VIII Line 1e (federal) + 1h (state/local)
    gov_grants = _safe_sum(
        filing.get("grscontrgvtgrnt"),
        filing.get("grntstogovt"),
    )
    total_contributions = _safe_float(filing.get("totcntrbgfts"))
    gov_grant_concentration = (
        gov_grants / total_contributions
        if gov_grants is not None and total_contributions and total_contributions > 0
        else None
    )

    return {
        "ein": ein,
        "ntee_code": ntee_code,
        "fiscal_year": int(year),
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "operating_margin": operating_margin,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_net_assets": total_net_assets,
        "gov_grant_concentration": gov_grant_concentration,
    }


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_sum(*vals) -> float | None:
    floats = [_safe_float(v) for v in vals if _safe_float(v) is not None]
    return sum(floats) if floats else None


def collect(
    out_path: Path,
    per_page: int = 100,
    max_orgs: int | None = None,
) -> pd.DataFrame:
    rows = []
    org_count = 0

    for ntee in NTEE_CODES:
        print(f"Fetching NTEE-{ntee} organizations…")
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
                if max_orgs and org_count >= max_orgs:
                    break
            page += 1
            time.sleep(REQUEST_DELAY)
            if max_orgs and org_count >= max_orgs:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        print("No records collected.")
        return df

    df = df.sort_values(["ein", "fiscal_year"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"Saved {len(df):,} filing records for {df['ein'].nunique():,} "
        f"organizations to {out_path}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect ProPublica 990 data for BEACON")
    parser.add_argument("--out", default="data/raw/propublica_panel.csv")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-orgs", type=int, default=None)
    args = parser.parse_args()
    collect(
        out_path=Path(args.out),
        per_page=args.per_page,
        max_orgs=args.max_orgs,
    )

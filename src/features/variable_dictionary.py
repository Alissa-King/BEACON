"""
BEACON Variable Dictionary — Appendix B / Chapter 3 reference.

Defines every predictor variable and the outcome variable used in BEACON,
including: Form 990 source, formula, theoretical justification, expected
direction of association with distress, and BEACON domain assignment.

This module is the authoritative specification. All other modules derive
feature lists and domain assignments from the structures defined here.

Theoretical grounding:
  Tuckman & Chang (1991)  — four indicators of financial vulnerability
  Greenlee & Trussel (2000) — sustained operating deficit as distress predictor
  Pfeffer & Salancik (1978) — Resource Dependence Theory (revenue concentration)
  Keating & Frumkin (2003) — program efficiency and financial health
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableSpec:
    name: str                 # Python column name
    label: str                # Human-readable label for tables/plots
    form_990_source: str      # Part and line number(s)
    formula: str              # Algebraic definition
    construct: str            # Theoretical construct being measured
    theoretical_basis: str    # Citation(s)
    expected_direction: str   # "higher = more risk" or "lower = more risk" or "nonlinear"
    beacon_domain: str        # Domain for SHAP aggregation
    unit: str = ""            # Units or range note
    availability: str = "High"  # High / Medium / Low (data availability in practice)
    notes: str = ""


# ── Outcome variable ──────────────────────────────────────────────────────────

DISTRESS_OUTCOME = VariableSpec(
    name="financial_distress",
    label="Financial Distress (Option 1 — Primary)",
    form_990_source="Part VIII; Part IX",
    formula="1 if operating_margin < 0 in BOTH year T+1 AND year T+2; else 0",
    construct="Sustained Operating Deficit",
    theoretical_basis=(
        "Greenlee & Trussel (2000): two consecutive years of negative operating margin "
        "operationalizes financial distress; widely replicated in nonprofit finance literature. "
        "Forward-looking (T+1, T+2) to prevent self-prediction at year T."
    ),
    expected_direction="Outcome variable (binary)",
    beacon_domain="N/A",
    unit="Binary {0, 1}",
    availability="High",
    notes=(
        "Option 1 is the primary label. Option 2 (unrestricted_net_assets < 0 in T+1 or T+2) "
        "serves as a robustness check. Rows where T+1 or T+2 is unobservable are "
        "right-censored and excluded."
    ),
)

DISTRESS_OUTCOME_2 = VariableSpec(
    name="financial_distress_2",
    label="Financial Distress (Option 2 — Robustness)",
    form_990_source="Part X, Line 27 (pre-2018) / Net assets without donor restrictions (post-2018)",
    formula="1 if unrestricted_net_assets < 0 in year T+1 OR year T+2; else 0",
    construct="Net Asset Insolvency",
    theoretical_basis=(
        "Tuckman & Chang (1991): negative equity (net assets) indicates genuine solvency risk "
        "beyond cash-flow stress. Post-ASU 2016-14 (effective 2018), 'net assets without donor "
        "restrictions' is the equivalent field."
    ),
    expected_direction="Outcome variable (binary, robustness check)",
    beacon_domain="N/A",
    unit="Binary {0, 1}",
    availability="Medium",
    notes=(
        "Rarer than Option 1 (lower base rate → heavier class imbalance). "
        "Available when unrestricted_net_assets is present; NaN otherwise."
    ),
)


# ── Predictor variables ────────────────────────────────────────────────────────

PREDICTORS: list[VariableSpec] = [

    # ── Financial Capacity domain ──────────────────────────────────────────

    VariableSpec(
        name="months_cash_on_hand",
        label="Months of Cash on Hand",
        form_990_source="Part X Lines 1–2 (cash + savings); Part IX Line 25 (total expenses)",
        formula="(Cash + Savings) / (Total Functional Expenses / 12)",
        construct="Liquidity — short-run operational cash coverage",
        theoretical_basis=(
            "Tuckman & Chang (1991): equity buffer / monthly expenses as primary vulnerability "
            "indicator. Months-of-cash is the nonprofit-sector standard for liquidity risk "
            "(Calabrese 2013). FASB ASC 958 requires disclosure of 'quantitative information "
            "about the availability of financial assets' including liquid reserves."
        ),
        expected_direction="Lower = more risk (fewer months of coverage → higher distress probability)",
        beacon_domain="Financial Capacity",
        unit="Months (continuous, winsorized at 1st/99th percentile)",
        availability="High",
    ),

    VariableSpec(
        name="current_ratio",
        label="Current Ratio",
        form_990_source="Part X Lines 1–5 (current assets); Part X Lines 17–20 (current liabilities)",
        formula="Current Assets / Current Liabilities",
        construct="Liquidity — ability to meet short-term obligations",
        theoretical_basis=(
            "Standard accounting ratio; ratio < 1.0 indicates current liabilities exceed current "
            "assets. Widely used in nonprofit financial health studies (Greenlee & Trussel 2000; "
            "Nunnenkamp & Öhler 2012). Complements months-of-cash by capturing balance-sheet "
            "liquidity rather than cash flow."
        ),
        expected_direction="Lower = more risk",
        beacon_domain="Financial Capacity",
        unit="Ratio (continuous); benchmark ≥ 1.0",
        availability="Medium",
        notes="Part X current sub-totals are not always separately reported; KNN imputed if missing.",
    ),

    VariableSpec(
        name="unrestricted_net_assets_ratio",
        label="Unrestricted Net Assets Ratio",
        form_990_source=(
            "Part X Line 27 / Line 33 (pre-2018); "
            "Net assets w/o donor restrictions / Total net assets (post-2018, ASU 2016-14)"
        ),
        formula="Unrestricted Net Assets / Total Net Assets",
        construct="Financial slack — proportion of net assets available without donor restriction",
        theoretical_basis=(
            "Tuckman & Chang (1991): equity ratio as vulnerability indicator. "
            "Organizations with low unrestricted ratios cannot access donor-restricted funds "
            "for operational shortfalls, limiting management flexibility under Resource Dependence "
            "Theory (Pfeffer & Salancik 1978). Ratio < 0 indicates accumulated deficits exceed "
            "unrestricted net assets."
        ),
        expected_direction="Lower (especially negative) = more risk",
        beacon_domain="Financial Capacity",
        unit="Ratio (can be negative); benchmark > 0",
        availability="Medium",
        notes=(
            "ASU 2016-14 (FASB 2016) reclassified three net asset classes into two. "
            "Pre-2018: UNRESTRICTED / TOTAL_NET_ASSETS. "
            "2018+: NET_WO_DONOR_RESTR / TOTAL_NET_ASSETS. "
            "BEACON treats both as equivalent with explicit year-conditional mapping."
        ),
    ),

    # ── Financial Sustainability domain ────────────────────────────────────

    VariableSpec(
        name="operating_margin",
        label="Operating Margin",
        form_990_source="Part VIII Line 12 (total revenue); Part IX Line 25 (total expenses)",
        formula="(Total Revenue − Total Functional Expenses) / Total Revenue",
        construct="Profitability / surplus generation",
        theoretical_basis=(
            "Greenlee & Trussel (2000): operating margin is the primary signal of financial "
            "vulnerability. Persistent negative margins exhaust reserves and precede organizational "
            "failure. This variable at year T is backward-looking; the DISTRESS label at T is "
            "based on T+1 and T+2 margins (no overlap — prevents self-prediction)."
        ),
        expected_direction="Lower (negative) = more risk",
        beacon_domain="Financial Sustainability",
        unit="Ratio (−∞ to 1); benchmark > 0",
        availability="High",
    ),

    VariableSpec(
        name="consecutive_deficits",
        label="Consecutive Deficit Years",
        form_990_source="Part VIII; Part IX (computed from historical panel)",
        formula="Count of consecutive years ending at T with operating_margin < 0",
        construct="Deficit persistence — evidence of structural rather than transitory imbalance",
        theoretical_basis=(
            "Greenlee & Trussel (2000): deficit persistence distinguishes structural financial "
            "weakness from temporary shortfalls. Requires a sorted per-EIN longitudinal panel "
            "before labeling; computed in a second pass to ensure the streak at T reflects only "
            "years ≤ T (strictly backward-looking)."
        ),
        expected_direction="Higher = more risk",
        beacon_domain="Financial Sustainability",
        unit="Integer (0, 1, 2, …); winsorized at 99th percentile",
        availability="High",
        notes=(
            "High correlation with the Option 1 label is expected and intentional: past deficit "
            "streaks are the strongest autocorrelation predictor of future deficit continuation. "
            "This is a valid predictive signal, not circular — the feature covers years ≤ T, "
            "the label covers T+1 and T+2."
        ),
    ),

    VariableSpec(
        name="program_expense_ratio",
        label="Program Expense Ratio",
        form_990_source="Part IX Line 25 Column B (program service expenses); Part IX Line 25 (total)",
        formula="Program Service Expenses / Total Functional Expenses",
        construct="Mission efficiency — proportion of spending directed toward programmatic activities",
        theoretical_basis=(
            "Keating & Frumkin (2003): organizations that chronically underfund programs relative "
            "to administrative costs face donor erosion and revenue decline, a precursor to "
            "financial distress. Also a proxy for organizational maturity and management quality. "
            "GuideStar and Charity Navigator use this ratio as a primary efficiency metric."
        ),
        expected_direction="Nonlinear — very low (<0.5) suggests administrative bloat; very high "
                           "(>0.95) may indicate underinvestment in capacity (management risk)",
        beacon_domain="Financial Sustainability",
        unit="Ratio [0, 1]; benchmark 0.65–0.85",
        availability="High",
        notes=(
            "Part IX Column B (program expenses) is reliably available for full 990 filers. "
            "Not available on 990-EZ. BEACON dataset restricted to full 990 filers."
        ),
    ),

    # ── Resource Dependence domain ─────────────────────────────────────────

    VariableSpec(
        name="gov_grant_concentration",
        label="Government Grant Concentration",
        form_990_source="Part VIII Lines 1e (federal grants) + 1g (state/local); Part VIII Line 1h (total contributions)",
        formula="Government Grants / Total Contributions",
        construct="Revenue dependency on government funding",
        theoretical_basis=(
            "Pfeffer & Salancik (1978) Resource Dependence Theory: organizations highly dependent "
            "on a single resource type (government grants) are vulnerable to policy changes, budget "
            "cycles, and contract termination. Nonprofits in NTEE L/P are often heavily "
            "government-funded, making this a particularly relevant risk dimension. "
            "Froelich (1999): government contract dependence correlates with mission drift and "
            "financial vulnerability."
        ),
        expected_direction="Higher = more risk (greater government dependency → greater exposure to funding cuts)",
        beacon_domain="Resource Dependence",
        unit="Ratio [0, 1]",
        availability="High",
    ),

    VariableSpec(
        name="revenue_hhi",
        label="Revenue Concentration (HHI)",
        form_990_source="Part VIII Lines 1–11 (all revenue streams)",
        formula=(
            "Σ (stream_i / total_revenue)² across streams: "
            "government grants, private contributions, program service, investment income, other"
        ),
        construct="Revenue diversification — concentration of revenue across funding sources",
        theoretical_basis=(
            "Herfindahl-Hirschman Index adapted for nonprofit revenue: Chang & Tuckman (1994) "
            "establish revenue concentration as a financial vulnerability indicator. "
            "HHI = 1.0 indicates a single revenue source (maximum concentration/fragility); "
            "HHI = 1/n indicates equal diversification across n streams. "
            "Resource Dependence Theory (Pfeffer & Salancik 1978): diversified revenue base "
            "reduces any single funder's leverage over the organization."
        ),
        expected_direction="Higher = more risk (more concentrated revenue = greater fragility)",
        beacon_domain="Resource Dependence",
        unit="Ratio [1/n, 1.0]; benchmark < 0.5",
        availability="High",
    ),

    # ── Organizational Risk domain ─────────────────────────────────────────

    VariableSpec(
        name="debt_to_equity",
        label="Debt-to-Equity Ratio",
        form_990_source="Part X Line 26 (total liabilities); Part X Line 33 (total net assets)",
        formula="Total Liabilities / Total Net Assets",
        construct="Leverage — extent of debt financing relative to organizational equity",
        theoretical_basis=(
            "Tuckman & Chang (1991): equity buffer indicator; high leverage reduces organizational "
            "resilience to revenue shocks. Trussel (2002): debt burden is a predictor of nonprofit "
            "financial distress. Ratio > 1 indicates liabilities exceed net assets (technical "
            "insolvency). Winsorized to manage extreme values when net assets approach zero."
        ),
        expected_direction="Higher = more risk",
        beacon_domain="Organizational Risk",
        unit="Ratio (continuous; can be negative if net assets negative); benchmark < 1.0",
        availability="High",
        notes="Denominator set to ±1 minimum to prevent division by zero; winsorized at 99th percentile.",
    ),

    VariableSpec(
        name="total_revenue_log",
        label="Organizational Scale (Log Revenue)",
        form_990_source="Part VIII Line 12 (total revenue)",
        formula="log(Total Revenue); Total Revenue in USD",
        construct="Organizational scale — size as a proxy for resource slack and resilience",
        theoretical_basis=(
            "Larger nonprofits tend to have more diverse revenue bases, greater reserves, and "
            "stronger governance infrastructure, making them more resilient to financial shocks "
            "(Helmut Anheier 2014). Log-transformation addresses right-skewed revenue distribution "
            "across nonprofit organizations. Included as a control to distinguish financial "
            "ratio effects from scale effects."
        ),
        expected_direction="Lower = more risk (smaller organizations more vulnerable)",
        beacon_domain="Organizational Risk",
        unit="Log(USD); continuous",
        availability="High",
        notes=(
            "Treated as a control variable for organizational size. Not used in primary BDI "
            "formula weights but included in model feature set to prevent scale confounding."
        ),
    ),
]


# ── Convenience accessors ──────────────────────────────────────────────────────

#: Ordered list of predictor column names — source of truth for BDI_FEATURE_COLUMNS
PREDICTOR_NAMES: list[str] = [v.name for v in PREDICTORS]

#: Domain → feature list mapping — source of truth for SHAP_DOMAIN_MAP
DOMAIN_MAP: dict[str, list[str]] = {}
for v in PREDICTORS:
    DOMAIN_MAP.setdefault(v.beacon_domain, []).append(v.name)

#: Human-readable label lookup
FEATURE_LABELS: dict[str, str] = {v.name: v.label for v in PREDICTORS}


def print_data_dictionary(include_notes: bool = True) -> None:
    """Print a committee-ready text table of all variables."""
    divider = "=" * 100
    print(divider)
    print("BEACON VARIABLE DICTIONARY")
    print("Appendix B — Operational Definitions and Form 990 Sources")
    print(divider)

    print("\nOUTCOME VARIABLES\n" + "-" * 60)
    for outcome in [DISTRESS_OUTCOME, DISTRESS_OUTCOME_2]:
        print(f"\n{outcome.label}")
        print(f"  Column     : {outcome.name}")
        print(f"  Source     : {outcome.form_990_source}")
        print(f"  Formula    : {outcome.formula}")
        print(f"  Construct  : {outcome.construct}")
        print(f"  Basis      : {outcome.theoretical_basis}")
        if include_notes and outcome.notes:
            print(f"  Notes      : {outcome.notes}")

    print(f"\n\nPREDICTOR VARIABLES ({len(PREDICTORS)} features)\n" + "-" * 60)
    for v in PREDICTORS:
        print(f"\n[{v.beacon_domain}] {v.label}")
        print(f"  Column     : {v.name}")
        print(f"  Source     : {v.form_990_source}")
        print(f"  Formula    : {v.formula}")
        print(f"  Construct  : {v.construct}")
        print(f"  Direction  : {v.expected_direction}")
        print(f"  Basis      : {v.theoretical_basis}")
        print(f"  Unit       : {v.unit}")
        print(f"  Availability: {v.availability}")
        if include_notes and v.notes:
            print(f"  Notes      : {v.notes}")

    print(f"\n{divider}")
    print(f"Total predictor variables: {len(PREDICTORS)}")
    print(f"Domains: {list(DOMAIN_MAP.keys())}")
    print(f"Variables per domain: { {d: len(fs) for d, fs in DOMAIN_MAP.items()} }")
    print(divider)


if __name__ == "__main__":
    print_data_dictionary()

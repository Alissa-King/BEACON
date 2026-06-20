"""
BEAM Executive Action Matrix (Appendix E).

BEAM is a rule-based interpretive mapping system that links statistically
identified risk features (via SHAP) to structured governance response
categories. It translates XAI outputs into practitioner-facing actions
grounded in nonprofit financial management literature.

Important limitations (Section 3.10):
  - BEAM does NOT imply causality between risk features and distress.
  - BEAM is a decision taxonomy, not a validated intervention model.
  - The study does not evaluate whether BEAM interventions reduce failure
    rates; that requires longitudinal action research (future work).
  - Governance teams should apply professional judgment when selecting
    among recommended actions.
"""

from dataclasses import dataclass


@dataclass
class BeamAction:
    risk_driver: str
    bdi_colours: list[str]        # colour codes matching bdi_colour column
    executive_action: str
    board_action: str
    threshold_description: str = ""


BEAM_MATRIX: list[BeamAction] = [
    BeamAction(
        risk_driver="months_cash_on_hand",
        bdi_colours=["Orange", "Red"],
        threshold_description="< 1.5 months of operating expenses",
        executive_action=(
            "Enact 90-day cash flow forecasting; draw on lines of credit; "
            "freeze non-essential hiring and capital expenditures."
        ),
        board_action=(
            "Mandate bi-weekly liquidity updates from the CFO; review reserve "
            "policies; form an emergency finance committee."
        ),
    ),
    BeamAction(
        risk_driver="gov_grant_concentration",
        bdi_colours=["Yellow", "Orange"],
        threshold_description="> 75% of total revenue from government sources",
        executive_action=(
            "Launch a multi-year strategic initiative to build unrestricted "
            "individual giving and corporate sponsorships."
        ),
        board_action=(
            "Require management to tie executive performance metrics to "
            "unrestricted revenue growth targets."
        ),
    ),
    BeamAction(
        risk_driver="consecutive_deficits",
        bdi_colours=["Red"],
        threshold_description=">= 2 consecutive years of negative operating margin",
        executive_action=(
            "Conduct immediate programmatic profitability analysis; "
            "sunset chronically underfunded programs."
        ),
        board_action=(
            "Demand a balanced budget recovery plan within 30 days; restrict "
            "usage of board-designated reserves to cover structural deficits."
        ),
    ),
    BeamAction(
        risk_driver="unrestricted_net_assets_ratio",
        bdi_colours=["Orange"],
        threshold_description="Declining trend over 2+ fiscal years",
        executive_action=(
            "Reduce administrative overhead; renegotiate vendor contracts; "
            "launch targeted operational fundraising."
        ),
        board_action=(
            "Review long-term strategic plan viability; consider feasibility "
            "of strategic partnerships or mergers."
        ),
    ),
    BeamAction(
        risk_driver="revenue_hhi",
        bdi_colours=["Yellow", "Orange", "Red"],
        threshold_description="HHI > 0.50 (highly concentrated revenue)",
        executive_action=(
            "Develop a revenue diversification plan targeting at least 3 "
            "independent revenue streams within 18 months."
        ),
        board_action=(
            "Commission a revenue sustainability audit; set board-level "
            "diversification targets as a KPI in CEO evaluation."
        ),
    ),
    BeamAction(
        risk_driver="debt_to_equity",
        bdi_colours=["Orange", "Red"],
        threshold_description="> 2.0 total liabilities-to-net-assets ratio",
        executive_action=(
            "Renegotiate debt terms; prioritize debt reduction in the annual "
            "budget; defer capital projects until ratio improves."
        ),
        board_action=(
            "Establish a debt ceiling policy; require CFO to report "
            "debt-to-equity quarterly alongside standard financials."
        ),
    ),
    BeamAction(
        risk_driver="operating_margin",
        bdi_colours=["Yellow", "Orange", "Red"],
        threshold_description="< 0% (operating deficit)",
        executive_action=(
            "Conduct immediate expense review; identify cost reduction "
            "opportunities; present a break-even recovery timeline."
        ),
        board_action=(
            "Require management to submit a 90-day operational recovery plan; "
            "pause major strategic initiatives until margin stabilizes."
        ),
    ),
    BeamAction(
        risk_driver="current_ratio",
        bdi_colours=["Orange", "Red"],
        threshold_description="< 1.0 current assets-to-current liabilities",
        executive_action=(
            "Accelerate receivables collection; negotiate extended payables "
            "terms; explore a short-term bridge facility."
        ),
        board_action=(
            "Review and update the organization's liquidity policy; "
            "set a minimum current ratio covenant of 1.2."
        ),
    ),
]

_DRIVER_INDEX: dict[str, BeamAction] = {a.risk_driver: a for a in BEAM_MATRIX}


def get_beam_actions(risk_drivers: list[dict], bdi_colour: str) -> list[dict]:
    """
    Return applicable BEAM governance actions for a given organization.

    Parameters
    ----------
    risk_drivers : top-N SHAP driver dicts from get_org_shap_drivers()
    bdi_colour   : "Green" | "Yellow" | "Orange" | "Red"
                   (bdi_colour column, derived from bdi_category)

    Returns
    -------
    list of action dicts — only actions whose colour threshold is met
    """
    actions = []
    for driver in risk_drivers:
        feature = driver["feature"]
        if feature not in _DRIVER_INDEX:
            continue
        action = _DRIVER_INDEX[feature]
        if bdi_colour in action.bdi_colours or bdi_colour == "Red":
            actions.append({
                "risk_driver": action.risk_driver,
                "label": driver["label"],
                "threshold_description": action.threshold_description,
                "shap_value": driver["shap_value"],
                "direction": driver["direction"],
                "executive_action": action.executive_action,
                "board_action": action.board_action,
            })
    return actions


def format_beam_report(
    org_name: str,
    bdi_score: float,
    bdi_category: str,
    beam_actions: list[dict],
) -> str:
    """Render a plain-text BEAM executive report."""
    lines = [
        "=" * 70,
        "BEACON FINANCIAL RESILIENCE REPORT",
        "=" * 70,
        f"Organization : {org_name}",
        f"BDI Score    : {bdi_score:.1f} / 100  "
        f"(0 = Low Risk -> 100 = Severe Risk)",
        f"Category     : {bdi_category.upper()}",
        "",
        "NOTE: BDI reflects predicted probability of financial distress",
        "based on Form 990 financial characteristics. SHAP drivers below",
        "are associative predictors of model output, not causal factors.",
        "",
        "PRIMARY RISK DRIVERS (SHAP Explanatory Decomposition)",
        "-" * 40,
    ]
    for i, a in enumerate(beam_actions, 1):
        sign = "+" if a["shap_value"] > 0 else ""
        lines.append(
            f"  {i}. {a['label']} "
            f"[SHAP: {sign}{a['shap_value']:.3f}] "
            f"— {a['direction']} distress risk"
        )

    lines += [
        "",
        "BEAM GOVERNANCE RESPONSE TAXONOMY",
        "(decision support — professional judgment required)",
        "-" * 40,
    ]
    for i, a in enumerate(beam_actions, 1):
        lines += [
            f"  [{i}] Risk Association: {a['label']}",
            f"      Executive: {a['executive_action']}",
            f"      Board    : {a['board_action']}",
            "",
        ]
    lines.append("=" * 70)
    return "\n".join(lines)

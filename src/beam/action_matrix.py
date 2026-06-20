"""
BEAM Executive Action Matrix (Appendix E).

Maps SHAP-identified risk drivers + BDI category to governance interventions.
"""

from dataclasses import dataclass, field


@dataclass
class BeamAction:
    risk_driver: str
    bdi_statuses: list[str]
    executive_action: str
    board_action: str
    threshold_description: str = ""


BEAM_MATRIX: list[BeamAction] = [
    BeamAction(
        risk_driver="months_cash_on_hand",
        bdi_statuses=["Orange", "Red"],
        threshold_description="< 1.5 months",
        executive_action=(
            "Enact 90-day cash flow forecasting; draw on lines of credit; "
            "freeze non-essential hiring and capital expenditures."
        ),
        board_action=(
            "Mandate bi-weekly liquidity updates from the CFO; review reserve "
            "policies; form emergency finance committee."
        ),
    ),
    BeamAction(
        risk_driver="gov_grant_concentration",
        bdi_statuses=["Yellow", "Orange"],
        threshold_description="> 75% of revenue",
        executive_action=(
            "Launch a 3-year strategic initiative to build unrestricted individual "
            "giving and corporate sponsorships."
        ),
        board_action=(
            "Require management to tie executive performance metrics to "
            "unrestricted revenue growth targets."
        ),
    ),
    BeamAction(
        risk_driver="consecutive_deficits",
        bdi_statuses=["Red"],
        threshold_description=">= 2 consecutive years",
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
        bdi_statuses=["Orange"],
        threshold_description="Declining trend",
        executive_action=(
            "Reduce administrative overhead; renegotiate vendor contracts; "
            "launch targeted operational fundraising."
        ),
        board_action=(
            "Review long-term strategic plan viability; consider feasibility of "
            "strategic partnerships or mergers."
        ),
    ),
    BeamAction(
        risk_driver="revenue_hhi",
        bdi_statuses=["Yellow", "Orange", "Red"],
        threshold_description="HHI > 0.50",
        executive_action=(
            "Develop a revenue diversification plan targeting at least 3 independent "
            "revenue streams within 18 months."
        ),
        board_action=(
            "Commission a revenue sustainability audit; set board-level diversification "
            "targets as a KPI in the CEO evaluation."
        ),
    ),
    BeamAction(
        risk_driver="debt_to_equity",
        bdi_statuses=["Orange", "Red"],
        threshold_description="> 2.0 ratio",
        executive_action=(
            "Renegotiate debt terms; prioritize debt reduction in the annual budget; "
            "defer capital projects until ratio improves."
        ),
        board_action=(
            "Establish a debt ceiling policy; require CFO to report debt-to-equity "
            "quarterly alongside standard financials."
        ),
    ),
    BeamAction(
        risk_driver="operating_margin",
        bdi_statuses=["Yellow", "Orange", "Red"],
        threshold_description="< 0% (deficit)",
        executive_action=(
            "Conduct immediate expense review; identify cost reduction opportunities; "
            "present a break-even recovery timeline."
        ),
        board_action=(
            "Require management to submit a 90-day operational recovery plan; "
            "pause major strategic initiatives until margin stabilizes."
        ),
    ),
    BeamAction(
        risk_driver="current_ratio",
        bdi_statuses=["Orange", "Red"],
        threshold_description="< 1.0",
        executive_action=(
            "Accelerate receivables collection; negotiate extended payables terms; "
            "explore a short-term bridge facility."
        ),
        board_action=(
            "Review and update the organization's liquidity policy; "
            "set a minimum current ratio covenant of 1.2."
        ),
    ),
]

_DRIVER_INDEX: dict[str, BeamAction] = {a.risk_driver: a for a in BEAM_MATRIX}


def get_beam_actions(
    risk_drivers: list[dict], bdi_category: str
) -> list[dict]:
    """
    Given SHAP risk drivers and the BDI category, return applicable BEAM actions.

    Args:
        risk_drivers: list of dicts from get_org_shap_drivers()
        bdi_category: "Green" | "Yellow" | "Orange" | "Red"

    Returns:
        list of action dicts with executive_action and board_action fields
    """
    actions = []
    for driver in risk_drivers:
        feature = driver["feature"]
        if feature not in _DRIVER_INDEX:
            continue
        action = _DRIVER_INDEX[feature]
        # Only trigger if current BDI status is in the action's applicable statuses
        if bdi_category in action.bdi_statuses or bdi_category == "Red":
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
        f"BDI Score    : {bdi_score:.1f} / 100",
        f"Category     : {bdi_category.upper()}",
        "",
        "PRIMARY RISK DRIVERS (XAI / SHAP)",
        "-" * 40,
    ]
    for i, a in enumerate(beam_actions, 1):
        sign = "+" if a["shap_value"] > 0 else ""
        lines.append(f"  {i}. {a['label']} "
                     f"[SHAP: {sign}{a['shap_value']:.3f}] "
                     f"— {a['direction']} distress risk")

    lines += ["", "BEAM EXECUTIVE ACTIONS", "-" * 40]
    for i, a in enumerate(beam_actions, 1):
        lines += [
            f"  [{i}] Risk Driver: {a['label']}",
            f"      Executive: {a['executive_action']}",
            f"      Board    : {a['board_action']}",
            "",
        ]
    lines.append("=" * 70)
    return "\n".join(lines)

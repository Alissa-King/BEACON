"""
BEACON Dashboard — Streamlit app.

DBA artifact: interactive risk assessment tool for nonprofit finance officers
and board members. Accepts manual feature entry or CSV upload of Form 990
data and outputs:
  - BEACON Distress Index (BDI) score with colour-coded risk category
  - SHAP feature contribution waterfall chart
  - BEAM governance response recommendations
  - Downloadable executive report

Run locally:
    streamlit run app/dashboard.py

Deploy to Streamlit Community Cloud:
    1. Push repo to GitHub (public or private with access granted)
    2. Go to share.streamlit.io → New app → select this repo
    3. Main file path: app/dashboard.py
    4. Click Deploy

Models are trained automatically on first launch using synthetic 990 data
(2,000 organizations, 2013–2025) if pre-trained model files are not found.
Training takes ~60 seconds and is cached for the duration of the session.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path fix so imports work when launched from repo root or app/ ─────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.variable_dictionary import PREDICTORS, PREDICTOR_NAMES, FEATURE_LABELS, DOMAIN_MAP
from src.features.bdi import BDI_FEATURE_COLUMNS, compute_bdi, CATEGORY_TO_COLOUR
from src.ingestion.cleaning_pipeline import run_cleaning_pipeline
from src.beam.action_matrix import get_beam_actions, format_beam_report


# ── Constants ─────────────────────────────────────────────────────────────────

COLOUR_HEX = {
    "Green":  "#2e8b57",
    "Yellow": "#e6b800",
    "Orange": "#e07b00",
    "Red":    "#c0392b",
}

COLOUR_BG = {
    "Green":  "#eafaf1",
    "Yellow": "#fefde7",
    "Orange": "#fff3e0",
    "Red":    "#fdecea",
}

DOMAIN_COLOURS = {
    "Financial Capacity":      "#1f77b4",
    "Financial Sustainability": "#d62728",
    "Resource Dependence":     "#ff7f0e",
    "Organizational Risk":     "#9467bd",
}

# Feature defaults for the manual input form (median-ish realistic values)
FEATURE_DEFAULTS = {
    "months_cash_on_hand":          3.0,
    "current_ratio":                1.5,
    "unrestricted_net_assets_ratio": 0.6,
    "operating_margin":             0.02,
    "consecutive_deficits":         0,
    "program_expense_ratio":        0.78,
    "gov_grant_concentration":      0.45,
    "revenue_hhi":                  0.35,
    "debt_to_equity":               0.4,
    "total_revenue_log":            13.5,
}

FEATURE_RANGES = {
    "months_cash_on_hand":          (0.0, 24.0, 0.1),
    "current_ratio":                (0.0, 10.0, 0.01),
    "unrestricted_net_assets_ratio": (-1.0, 2.0, 0.01),
    "operating_margin":             (-1.0, 1.0, 0.001),
    "consecutive_deficits":         (0, 10, 1),
    "program_expense_ratio":        (0.0, 1.0, 0.01),
    "gov_grant_concentration":      (0.0, 1.0, 0.01),
    "revenue_hhi":                  (0.0, 1.0, 0.01),
    "debt_to_equity":               (-5.0, 20.0, 0.01),
    "total_revenue_log":            (8.0, 22.0, 0.1),
}


# ── Model loading / auto-training (cached) ────────────────────────────────────

# Use a writable temp directory for models when running on Streamlit Cloud
# (the repo root may be read-only in cloud environments)
_MODELS_DIR = ROOT / "models"
if not _MODELS_DIR.exists():
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        _MODELS_DIR = Path(tempfile.gettempdir()) / "beacon_models"
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = _MODELS_DIR / "random_forest.pkl"
CALIBRATOR_PATH = _MODELS_DIR / "random_forest_calibrator.pkl"


def _train_models_from_synthetic() -> tuple:
    """Train Random Forest + calibrator on 2,000 synthetic orgs. ~45–90 s."""
    import joblib
    import src.models.train as _train_mod
    from src.ingestion.synthetic_data import generate_synthetic_990
    from src.ingestion.cleaning_pipeline import run_cleaning_pipeline

    # Redirect the training module's MODEL_DIR to our writable directory
    _orig_model_dir = _train_mod.MODEL_DIR
    _train_mod.MODEL_DIR = _MODELS_DIR

    try:
        df_raw = generate_synthetic_990(n_orgs=2000, seed=42)
        df_clean = run_cleaning_pipeline(df_raw)
        _train_mod.train_and_evaluate(df_clean)
    finally:
        _train_mod.MODEL_DIR = _orig_model_dir

    pipeline = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH) if CALIBRATOR_PATH.exists() else None
    return pipeline, calibrator


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Return (pipeline, calibrator).  If saved models don't exist, train them
    on synthetic data and cache the result for the rest of the session.
    """
    import joblib
    if MODEL_PATH.exists():
        pipeline = joblib.load(MODEL_PATH)
        calibrator = joblib.load(CALIBRATOR_PATH) if CALIBRATOR_PATH.exists() else None
        return pipeline, calibrator

    # Auto-train with a visible progress message
    with st.spinner(
        "First launch: training BEACON models on 2,000 synthetic nonprofits "
        "(Random Forest + isotonic calibration). This takes ~60 seconds and "
        "won't repeat for this session…"
    ):
        try:
            return _train_models_from_synthetic()
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            return None, None


@st.cache_resource(show_spinner=False)
def load_shap_explainer(_pipeline):
    """Build SHAP TreeExplainer from the cached pipeline."""
    try:
        import shap
        clf = _pipeline.named_steps["clf"]
        return shap.TreeExplainer(clf)
    except Exception:
        return None


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_row(row_df: pd.DataFrame, pipeline, calibrator) -> dict:
    """
    Score a single-row (or multi-row) cleaned dataframe.
    Returns dict with bdi_score, bdi_category, bdi_colour, calibrated_prob, raw_prob.
    """
    X = row_df[BDI_FEATURE_COLUMNS]
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    X_scaled = scaler.transform(X)

    raw_probs = clf.predict_proba(X_scaled)[:, 1]
    if calibrator is not None:
        cal_probs = calibrator.predict(raw_probs)
    else:
        cal_probs = raw_probs

    result_df = compute_bdi(row_df.copy(), cal_probs)
    return result_df, cal_probs, raw_probs


def get_shap_values(pipeline, explainer, row_df: pd.DataFrame) -> np.ndarray | None:
    if explainer is None:
        return None
    X = row_df[BDI_FEATURE_COLUMNS]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    sv = explainer.shap_values(X_scaled)
    return sv if sv.ndim == 2 else sv[:, :, 1]


# ── Plotly charts ─────────────────────────────────────────────────────────────

def bdi_gauge(bdi_score: float, bdi_colour: str) -> go.Figure:
    colour_hex = COLOUR_HEX.get(bdi_colour, "#888888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bdi_score,
        number={"font": {"size": 52, "color": colour_hex}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 40, 60, 80, 100],
                "ticktext": ["0", "40", "60", "80", "100"],
                "tickfont": {"size": 12},
            },
            "bar": {"color": colour_hex, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40],  "color": "#d4efdf"},
                {"range": [40, 60], "color": "#fef9c3"},
                {"range": [60, 80], "color": "#fde8c8"},
                {"range": [80, 100], "color": "#fad3d0"},
            ],
            "threshold": {
                "line": {"color": colour_hex, "width": 4},
                "thickness": 0.75,
                "value": bdi_score,
            },
        },
        title={
            "text": "BEACON Distress Index",
            "font": {"size": 18, "color": "#333333"},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=280, margin={"t": 60, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="white",
    )
    return fig


def shap_waterfall(shap_vals: np.ndarray, base_value: float, idx: int = 0) -> go.Figure:
    row_shap = shap_vals[idx]
    labels = [FEATURE_LABELS.get(f, f) for f in BDI_FEATURE_COLUMNS]

    # Sort by magnitude descending, keep top 10
    order = np.argsort(np.abs(row_shap))[::-1]
    row_shap = row_shap[order]
    labels = [labels[i] for i in order]
    domains = [
        next((d for d, fs in DOMAIN_MAP.items() if BDI_FEATURE_COLUMNS[i] in fs), "Other")
        for i in order
    ]

    colours = [
        ("#c0392b" if v > 0 else "#2e8b57") for v in row_shap
    ]

    fig = go.Figure(go.Bar(
        x=row_shap,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.3f}" for v in row_shap],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#555")
    fig.update_layout(
        title={
            "text": "SHAP Feature Contributions to P(Distress)<br>"
                    "<sup>Red = increases risk · Green = decreases risk · "
                    "Associative, not causal</sup>",
            "font": {"size": 14},
        },
        xaxis_title="SHAP value (log-odds contribution)",
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        height=380,
        margin={"t": 80, "b": 40, "l": 200, "r": 80},
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )
    return fig


def domain_bar(shap_vals: np.ndarray, idx: int = 0) -> go.Figure:
    feat_idx = {f: i for i, f in enumerate(BDI_FEATURE_COLUMNS)}
    domain_totals = {}
    for domain, features in DOMAIN_MAP.items():
        idxs = [feat_idx[f] for f in features if f in feat_idx]
        domain_totals[domain] = float(shap_vals[idx, idxs].sum()) if idxs else 0.0

    domains = list(domain_totals.keys())
    values = list(domain_totals.values())
    colours = [COLOUR_HEX["Red"] if v > 0 else COLOUR_HEX["Green"] for v in values]

    fig = go.Figure(go.Bar(
        x=domains, y=values,
        marker_color=colours,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_width=1, line_color="#555")
    fig.update_layout(
        title={
            "text": "BEACON Domain Contributions<br>"
                    "<sup>Signed SHAP aggregation — explanatory decomposition layer</sup>",
            "font": {"size": 14},
        },
        yaxis_title="Cumulative SHAP (log-odds)",
        height=300,
        margin={"t": 80, "b": 60, "l": 60, "r": 30},
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )
    return fig


# ── UI rendering helpers ──────────────────────────────────────────────────────

def render_bdi_header(bdi_score: float, bdi_category: str, bdi_colour: str):
    hex_c = COLOUR_HEX[bdi_colour]
    bg_c  = COLOUR_BG[bdi_colour]
    st.markdown(
        f"""
        <div style="
            background:{bg_c}; border-left:8px solid {hex_c};
            border-radius:8px; padding:18px 24px; margin-bottom:12px;">
          <h2 style="margin:0; color:{hex_c}; font-size:1.6rem;">
            BDI Score: {bdi_score:.1f} / 100 — {bdi_category}
          </h2>
          <p style="margin:6px 0 0; color:#555; font-size:0.9rem;">
            0 = No Risk &nbsp;·&nbsp; 40 = Moderate &nbsp;·&nbsp;
            60 = Elevated &nbsp;·&nbsp; 80 = Severe
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_beam_actions(beam_actions: list[dict], bdi_colour: str):
    if not beam_actions:
        st.info("No BEAM governance actions triggered at this risk level.")
        return

    st.subheader("BEAM Governance Response Taxonomy")
    st.caption(
        "Decision support only — professional judgment required. "
        "BEAM links model-identified risk associations to structured governance "
        "responses. It does NOT imply causality between risk features and distress."
    )
    for a in beam_actions:
        sign = "+" if a["shap_value"] > 0 else ""
        with st.expander(
            f"**{a['label']}**  ·  SHAP {sign}{a['shap_value']:.3f}  ·  "
            f"{a['threshold_description']}",
            expanded=True,
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Executive Action**")
                st.write(a["executive_action"])
            with col2:
                st.markdown("**Board Action**")
                st.write(a["board_action"])


def render_feature_table(row_df: pd.DataFrame):
    display = []
    for feat in BDI_FEATURE_COLUMNS:
        spec = next((p for p in PREDICTORS if p.name == feat), None)
        val = row_df[feat].iloc[0]
        display.append({
            "Feature": FEATURE_LABELS.get(feat, feat),
            "Value": f"{val:.3f}" if isinstance(val, float) else str(val),
            "Domain": spec.beacon_domain if spec else "—",
            "Risk Direction": spec.expected_direction[:40] if spec else "—",
        })
    st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="BEACON — Nonprofit Financial Risk Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/lighthouse.png",
            width=72,
        )
        st.title("BEACON")
        st.caption(
            "**B**usiness **E**arly-Warning **A**nalytics for "
            "**C**ommunity **O**rganization **N**avigation"
        )
        st.divider()

        input_mode = st.radio(
            "Input mode",
            ["Manual entry", "CSV upload"],
            help="Manual entry: enter values for one organization. "
                 "CSV upload: score multiple organizations from a cleaned 990 panel.",
        )

        st.divider()
        st.markdown("**About BEACON**")
        st.caption(
            "Predicts financial distress for NTEE Category L (Housing & Shelter) "
            "and P (Human Services) nonprofits using 10 Form 990 features. "
            "Primary model: Random Forest (AUC 0.731) with isotonic calibration "
            "(Brier score 0.229 → 0.159)."
        )
        st.caption(
            "⚠️ BEACON is a research prototype. BDI scores are predictive "
            "associations, not audit opinions or legal determinations."
        )

    # ── Load models (auto-trains on first launch if not cached) ───────────────
    pipeline, calibrator = load_models()
    models_loaded = pipeline is not None

    # ── Explainer (lazy) ───────────────────────────────────────────────────────
    explainer = load_shap_explainer(pipeline) if models_loaded else None

    # ── MANUAL ENTRY MODE ──────────────────────────────────────────────────────
    if input_mode == "Manual entry":
        st.header("Organization Risk Assessment")

        with st.form("manual_form"):
            st.subheader("Form 990 Financial Indicators")
            st.caption(
                "Enter values for a single organization. "
                "Hover over ⓘ labels for variable definitions."
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Financial Capacity**")
                months_cash = st.number_input(
                    "Months of Cash on Hand ⓘ",
                    min_value=0.0, max_value=24.0, step=0.1,
                    value=float(FEATURE_DEFAULTS["months_cash_on_hand"]),
                    help="(Cash + Savings) / (Total Expenses / 12). "
                         "Part X Lines 1–2 ÷ Part IX Line 25. Benchmark ≥ 3 months.",
                )
                current_ratio = st.number_input(
                    "Current Ratio ⓘ",
                    min_value=0.0, max_value=10.0, step=0.01,
                    value=float(FEATURE_DEFAULTS["current_ratio"]),
                    help="Current Assets / Current Liabilities. Part X. Benchmark ≥ 1.0.",
                )
                una_ratio = st.number_input(
                    "Unrestricted Net Assets Ratio ⓘ",
                    min_value=-1.0, max_value=2.0, step=0.01,
                    value=float(FEATURE_DEFAULTS["unrestricted_net_assets_ratio"]),
                    help="Unrestricted Net Assets / Total Net Assets. "
                         "Part X Line 27 (or Net assets w/o donor restrictions post-2018). "
                         "Negative = accumulated deficits.",
                )

                st.markdown("**Financial Sustainability**")
                op_margin = st.number_input(
                    "Operating Margin ⓘ",
                    min_value=-1.0, max_value=1.0, step=0.001, format="%.3f",
                    value=float(FEATURE_DEFAULTS["operating_margin"]),
                    help="(Total Revenue − Total Expenses) / Total Revenue. "
                         "Part VIII Line 12 and Part IX Line 25. Benchmark > 0.",
                )
                consec_deficits = st.number_input(
                    "Consecutive Deficit Years ⓘ",
                    min_value=0, max_value=10, step=1,
                    value=int(FEATURE_DEFAULTS["consecutive_deficits"]),
                    help="Number of consecutive prior years with negative operating margin. "
                         "Backward-looking streak ending at the current year.",
                )
                prog_exp_ratio = st.number_input(
                    "Program Expense Ratio ⓘ",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(FEATURE_DEFAULTS["program_expense_ratio"]),
                    help="Program Service Expenses / Total Functional Expenses. "
                         "Part IX Column B ÷ Line 25. Benchmark 0.65–0.85.",
                )

            with col_b:
                st.markdown("**Resource Dependence**")
                gov_conc = st.number_input(
                    "Government Grant Concentration ⓘ",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(FEATURE_DEFAULTS["gov_grant_concentration"]),
                    help="Government Grants / Total Contributions. "
                         "Part VIII Lines 1e+1g ÷ 1h. High values indicate government dependency.",
                )
                rev_hhi = st.number_input(
                    "Revenue HHI ⓘ",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(FEATURE_DEFAULTS["revenue_hhi"]),
                    help="Herfindahl-Hirschman Index across 5 revenue streams. "
                         "0.2 = well diversified; 1.0 = single source. Benchmark < 0.5.",
                )

                st.markdown("**Organizational Risk**")
                debt_equity = st.number_input(
                    "Debt-to-Equity Ratio ⓘ",
                    min_value=-5.0, max_value=20.0, step=0.01,
                    value=float(FEATURE_DEFAULTS["debt_to_equity"]),
                    help="Total Liabilities / Total Net Assets. "
                         "Part X Lines 26 ÷ 33. Benchmark < 1.0.",
                )
                rev_log = st.number_input(
                    "Log Total Revenue ⓘ",
                    min_value=8.0, max_value=22.0, step=0.1,
                    value=float(FEATURE_DEFAULTS["total_revenue_log"]),
                    help="Natural log of Part VIII Line 12 (total revenue in USD). "
                         "ln(100k) ≈ 11.5 · ln(1M) ≈ 13.8 · ln(10M) ≈ 16.1",
                )

                st.markdown("**Organization identifier (optional)**")
                org_name = st.text_input("Organization name", value="My Nonprofit")
                org_ein  = st.text_input("EIN", value="")
                fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=2030,
                                              value=2023, step=1)

            submitted = st.form_submit_button("Compute BDI Score", type="primary")

        if submitted and models_loaded:
            row = {
                "months_cash_on_hand":          months_cash,
                "current_ratio":                current_ratio,
                "unrestricted_net_assets_ratio": una_ratio,
                "operating_margin":             op_margin,
                "consecutive_deficits":         float(consec_deficits),
                "program_expense_ratio":        prog_exp_ratio,
                "gov_grant_concentration":      gov_conc,
                "revenue_hhi":                  rev_hhi,
                "debt_to_equity":               debt_equity,
                "total_revenue_log":            rev_log,
                "ein":        org_ein or "000000000",
                "org_name":   org_name,
                "ntee_code":  "L",
                "fiscal_year": fiscal_year,
            }
            row_df = pd.DataFrame([row])

            with st.spinner("Scoring…"):
                scored_df, cal_probs, raw_probs = score_row(row_df, pipeline, calibrator)
                shap_vals = get_shap_values(pipeline, explainer, row_df)

            bdi_score    = float(scored_df["bdi_score"].iloc[0])
            bdi_category = str(scored_df["bdi_category"].iloc[0])
            bdi_colour   = str(scored_df["bdi_colour"].iloc[0])

            # ── BDI result ─────────────────────────────────────────────────────
            render_bdi_header(bdi_score, bdi_category, bdi_colour)

            col_gauge, col_meta = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(bdi_gauge(bdi_score, bdi_colour), use_container_width=True)
            with col_meta:
                st.markdown("### Score Summary")
                st.metric("BDI Score", f"{bdi_score:.1f} / 100")
                st.metric("Risk Category", bdi_category)
                st.metric("Calibrated P(Distress)", f"{cal_probs[0]:.1%}")
                st.metric("Raw P(Distress)", f"{raw_probs[0]:.1%}")
                st.caption(
                    "P(Distress) = predicted probability that this organization "
                    "will experience two consecutive operating deficits within 24 months "
                    "(Greenlee & Trussel 2000 definition)."
                )

            st.divider()

            # ── SHAP charts ────────────────────────────────────────────────────
            if shap_vals is not None:
                st.subheader("Risk Driver Analysis (SHAP)")
                st.caption(
                    "SHAP values explain **this model's output** for this organization. "
                    "They are associative predictors of BDI score, not causal determinants "
                    "of financial distress."
                )
                tab_waterfall, tab_domain = st.tabs(
                    ["Feature Contributions", "Domain Decomposition"]
                )
                with tab_waterfall:
                    st.plotly_chart(
                        shap_waterfall(shap_vals, base_value=0.0, idx=0),
                        use_container_width=True,
                    )
                with tab_domain:
                    st.plotly_chart(
                        domain_bar(shap_vals, idx=0),
                        use_container_width=True,
                    )
                st.divider()

                # BEAM actions
                top_drivers = [
                    {
                        "feature": BDI_FEATURE_COLUMNS[i],
                        "label": FEATURE_LABELS.get(BDI_FEATURE_COLUMNS[i], BDI_FEATURE_COLUMNS[i]),
                        "shap_value": round(float(shap_vals[0, i]), 4),
                        "direction": "increases" if shap_vals[0, i] > 0 else "decreases",
                    }
                    for i in np.argsort(np.abs(shap_vals[0]))[::-1][:5]
                ]
                beam_actions = get_beam_actions(top_drivers, bdi_colour)
                render_beam_actions(beam_actions, bdi_colour)

            else:
                st.info("SHAP explainer unavailable — install the `shap` package.")

            st.divider()

            # ── Input feature table ────────────────────────────────────────────
            with st.expander("Input feature values", expanded=False):
                render_feature_table(row_df)

            # ── Downloadable report ────────────────────────────────────────────
            st.subheader("Executive Report")
            beam_actions_for_report = get_beam_actions(top_drivers, bdi_colour) if shap_vals is not None else []
            report_text = format_beam_report(
                org_name=org_name,
                bdi_score=bdi_score,
                bdi_category=bdi_category,
                beam_actions=beam_actions_for_report,
            )
            st.text(report_text)
            st.download_button(
                "Download Executive Report (.txt)",
                data=report_text,
                file_name=f"BEACON_report_{org_ein or 'org'}_{fiscal_year}.txt",
                mime="text/plain",
            )

    # ── CSV UPLOAD MODE ────────────────────────────────────────────────────────
    else:
        st.header("Batch Scoring — CSV Upload")
        st.caption(
            "Upload a CSV containing Form 990 feature columns. Required columns: "
            + ", ".join(f"`{c}`" for c in BDI_FEATURE_COLUMNS)
            + ". Optional: `ein`, `org_name`, `fiscal_year`, `ntee_code`."
        )

        uploaded = st.file_uploader("Upload 990 panel CSV", type=["csv"])

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df):,} rows, {df['ein'].nunique() if 'ein' in df else '?'} organizations.")

                missing_cols = [c for c in BDI_FEATURE_COLUMNS if c not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.stop()

                if not models_loaded:
                    st.error("Models not loaded — cannot score. Run `python run_beacon.py` first.")
                    st.stop()

                # Add stub columns if not present
                if "ntee_code" not in df.columns:
                    df["ntee_code"] = "L"
                if "ein" not in df.columns:
                    df["ein"] = [f"{i:09d}" for i in range(len(df))]

                with st.spinner(f"Scoring {len(df):,} rows…"):
                    df_clean = run_cleaning_pipeline(df)
                    scored_df, cal_probs, _ = score_row(df_clean, pipeline, calibrator)
                    shap_vals = get_shap_values(pipeline, explainer, df_clean)

                st.success(f"Scored {len(scored_df):,} rows.")

                # ── Summary stats ──────────────────────────────────────────────
                st.subheader("Portfolio Risk Overview")
                cat_counts = scored_df["bdi_category"].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                for col, label in zip(
                    [col1, col2, col3, col4],
                    ["Low Risk", "Moderate Risk", "Elevated Risk", "Severe Risk"],
                ):
                    with col:
                        count = int(cat_counts.get(label, 0))
                        colour = COLOUR_HEX[CATEGORY_TO_COLOUR[label]]
                        st.markdown(
                            f"<div style='text-align:center; color:{colour};'>"
                            f"<h1>{count}</h1><p>{label}</p></div>",
                            unsafe_allow_html=True,
                        )

                # ── Score distribution ─────────────────────────────────────────
                fig_dist = go.Figure(go.Histogram(
                    x=scored_df["bdi_score"],
                    nbinsx=25,
                    marker_color="#3498db",
                    opacity=0.75,
                ))
                fig_dist.update_layout(
                    title="BDI Score Distribution",
                    xaxis_title="BDI Score",
                    yaxis_title="Count",
                    height=280,
                    margin={"t": 50, "b": 40, "l": 50, "r": 20},
                    paper_bgcolor="white",
                    plot_bgcolor="#fafafa",
                )
                for threshold, color in [(40, "gold"), (60, "orange"), (80, "red")]:
                    fig_dist.add_vline(
                        x=threshold, line_dash="dash",
                        line_color=color, line_width=1.5,
                        annotation_text=str(threshold), annotation_position="top right",
                    )
                st.plotly_chart(fig_dist, use_container_width=True)

                # ── High-risk org table ────────────────────────────────────────
                st.subheader("Elevated + Severe Risk Organizations")
                high_risk = scored_df[scored_df["bdi_score"] >= 60].copy()
                if high_risk.empty:
                    st.info("No organizations in Elevated or Severe Risk category.")
                else:
                    display_cols = (
                        ["org_name"] if "org_name" in high_risk.columns else []
                    ) + ["ein", "fiscal_year", "bdi_score", "bdi_category"] + BDI_FEATURE_COLUMNS[:5]
                    display_cols = [c for c in display_cols if c in high_risk.columns]
                    st.dataframe(
                        high_risk[display_cols].sort_values("bdi_score", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

                # ── Global SHAP if available ───────────────────────────────────
                if shap_vals is not None:
                    st.subheader("Global Feature Importance (SHAP)")
                    mean_abs = np.abs(shap_vals).mean(axis=0)
                    imp_df = (
                        pd.DataFrame({
                            "feature": [FEATURE_LABELS.get(f, f) for f in BDI_FEATURE_COLUMNS],
                            "mean_abs_shap": mean_abs,
                        })
                        .sort_values("mean_abs_shap")
                    )
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["mean_abs_shap"],
                        y=imp_df["feature"],
                        orientation="h",
                        marker_color="#e74c3c",
                        opacity=0.8,
                    ))
                    fig_imp.update_layout(
                        xaxis_title="Mean |SHAP Value|",
                        height=350,
                        margin={"t": 30, "b": 40, "l": 200, "r": 40},
                        paper_bgcolor="white", plot_bgcolor="#fafafa",
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                # ── Download scored CSV ────────────────────────────────────────
                output_cols = (
                    ["ein", "org_name", "fiscal_year", "ntee_code"]
                    + BDI_FEATURE_COLUMNS
                    + ["bdi_score", "bdi_category", "bdi_colour"]
                )
                output_cols = [c for c in output_cols if c in scored_df.columns]
                csv_bytes = scored_df[output_cols].to_csv(index=False).encode()
                st.download_button(
                    "Download Scored Results (.csv)",
                    data=csv_bytes,
                    file_name="beacon_scored_results.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(f"Error processing file: {exc}")
                st.exception(exc)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "BEACON v1.0 · Dissertation research prototype · "
        "Alissa King, DBA Candidate · "
        "Model: Random Forest (primary, AUC 0.731) + Isotonic Calibration · "
        "Features: 10 Form 990 indicators · "
        "Scope: NTEE Categories L (Housing & Shelter) and P (Human Services) · "
        "⚠️ Not for audit, legal, or regulatory use."
    )


if __name__ == "__main__":
    main()

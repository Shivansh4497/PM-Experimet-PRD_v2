# app.py
"""
Streamlit front-end for the A/B Test PRD Generator.
This is a complete replacement file that:
- preserves original flows (Intro -> Hypothesis -> PRD -> Calculations -> Review)
- adds an opt-in RAG toggle (disabled by default)
- integrates the lightweight persistence layer (utils.persistence) with a Save PRD button
- calls generate_content in a compatibility-safe way to support both old and new API handler signatures

Drop it into your repo root replacing the existing app.py.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import re
import json
from functools import partial
import streamlit.components.v1 as components
from typing import Any, Dict

# --- Import utilities (with graceful fallbacks) ---
# generate_content might live in utils.api_handler or root api_handler depending on your repo.
try:
    # Prefer a namespaced utils import if present
    from utils.api_handler import generate_content
    from utils.calculations import calculate_sample_size_proportion, calculate_sample_size_continuous, calculate_duration
    from utils.pdf_generator import create_pdf
except Exception:
    try:
        from api_handler import generate_content
        from calculations import calculate_sample_size_proportion, calculate_sample_size_continuous, calculate_duration
        from pdf_generator import create_pdf
    except Exception:
        # placeholder implementations (keeps app usable without utils)
        def generate_content(*args, **kwargs):
            content_type = args[-1] if args else kwargs.get("mode", "hypotheses")
            st.warning(f"LLM generate_content not available; using placeholder for {content_type}.")
            if content_type == "hypotheses":
                return {
                    "Hypothesis 1": {"Statement": "Statement 1", "Rationale": "Rationale 1", "Behavioral Basis": "Basis 1"},
                    "Hypothesis 2": {"Statement": "Statement 2", "Rationale": "Rationale 2", "Behavioral Basis": "Basis 2"},
                }
            if content_type == "enrich_hypothesis":
                return {"Statement": args[1].get("custom_hypothesis", ""), "Rationale": "Generated Rationale", "Behavioral Basis": "Generated Basis"}
            if content_type == "prd_sections":
                return {
                    "Problem_Statement": "This is the generated problem statement.",
                    "Goal_and_Success_Metrics": "This is the generated goal and success metrics section.",
                    "Implementation_Plan": ["Step 1", "Step 2"]
                }
            if content_type == "risks":
                return {"risks": [{"risk": "A potential risk.", "mitigation": "A potential mitigation."}]}
            return {"error": "Content generation utility is not available."}

        def calculate_sample_size_proportion(current_value, min_detectable_effect, confidence, power):
            return 1000

        def calculate_sample_size_continuous(mean, std_dev, min_detectable_effect, confidence, power):
            return 1200

        def calculate_duration(sample_size, daily_active_users, coverage):
            return 14

        def create_pdf(prd_data):
            return b"This is a placeholder PDF."

# persistence layer (utils.persistence)
try:
    from utils.persistence import init_db, save_prd, list_prds, get_prd, search_prds, get_audit_for_prd
    PERSISTENCE_AVAILABLE = True
except Exception:
    PERSISTENCE_AVAILABLE = False

# Keep compatibility if SciPy not installed; calculations file may handle it.
try:
    from scipy.stats import norm  # to detect availability
    CALCULATIONS_AVAILABLE = True
except Exception:
    CALCULATIONS_AVAILABLE = False

# --- Page config and CSS ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarNavCollapseButton"] { display: none !important; }
    div[data-testid="stAppViewContainer"] { margin-top: -6rem; }
    .app-header { text-align: center; padding: 1rem 0; margin-bottom: 1rem; }
    .app-header h1 { font-size: 2.5rem; font-weight: 700; color: #e0e0e0; margin: 0; }
    .app-header p { font-size: 1.1rem; color: #8b949e; margin: 0; }
    .top-nav { display:flex; justify-content:center; padding:10px 0; border-bottom:1px solid #30363d; margin-bottom:2rem; }
    .nav-button { background:transparent; border:2px solid #30363d; color:#8b949e; font-weight:bold; padding:8px 16px; margin:0 5px; border-radius:8px; }
    .nav-button.complete-stage { border-color:#216d33; color:#c9d1d9; }
    .nav-button.active-stage { background-color:#216d33; color:white; border-color:#2ea043; }
    html, body, [class*="st-"] { font-family: 'Roboto', sans-serif; }
    .stButton > button { background-color: #216d33; color: white; border-radius: 8px; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# --- Session state defaults ---
def _init_session_state():
    if "stage" not in st.session_state:
        st.session_state.stage = "Intro"  # Intro -> Hypothesis -> PRD -> Calculations -> Review
    if "prd_data" not in st.session_state:
        st.session_state.prd_data = {
            "intro_data": {},
            "hypothesis": {},
            "prd_sections": {},
            "calculations": {},
            "risks": []
        }
    if "editing_section" not in st.session_state:
        st.session_state.editing_section = None
    if "editing_risk" not in st.session_state:
        st.session_state.editing_risk = None
    if "hypotheses_selected" not in st.session_state:
        st.session_state.hypotheses_selected = False
    if "hypotheses" not in st.session_state:
        st.session_state.hypotheses = {}
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False
    if "saved_prd_info" not in st.session_state:
        st.session_state.saved_prd_info = None
    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False

_init_session_state()

STAGES = ["Intro", "Hypothesis", "PRD", "Calculations", "Review"]

# --- Navigation helpers ---
def next_stage():
    idx = STAGES.index(st.session_state.stage)
    if idx < len(STAGES) - 1:
        st.session_state.stage = STAGES[idx + 1]
        st.session_state.scroll_to_top = True

def prev_stage():
    idx = STAGES.index(st.session_state.stage)
    if idx > 0:
        st.session_state.stage = STAGES[idx - 1]
        st.session_state.scroll_to_top = True

def scroll_to_top():
    components.html("<script>window.scrollTo(0,0)</script>")

# --- Basic utilities used across the app ---
def format_content_for_display(content):
    if isinstance(content, list):
        return "\n".join([f"- {item}" for item in content])
    else:
        return str(content)

def set_editing_section(section_title):
    st.session_state.editing_section = section_title
    st.session_state.editing_risk = None

def set_editing_risk(risk_index):
    st.session_state.editing_risk = risk_index
    st.session_state.editing_section = None

def save_edit(section_title):
    edited_text = st.session_state.get(f"text_area_{section_title}", "")
    original_content = st.session_state.prd_data["prd_sections"].get(section_title, "")
    if isinstance(original_content, list):
        st.session_state.prd_data["prd_sections"][section_title] = [line.strip("- ").strip() for line in edited_text.split('\n') if line.strip()]
    else:
        st.session_state.prd_data["prd_sections"][section_title] = edited_text
    st.session_state.editing_section = None
    st.success(f"Changes to '{section_title.replace('_',' ').title()}' saved!")

def save_risk_edit(risk_index):
    edited_risk = st.session_state.get(f"text_area_risk_{risk_index}", "")
    edited_mitigation = st.session_state.get(f"text_area_mitigation_{risk_index}", "")
    st.session_state.prd_data["risks"][risk_index] = {"risk": edited_risk, "mitigation": edited_mitigation}
    st.session_state.editing_risk = None
    st.success(f"Changes to Risk {risk_index+1} saved!")

def save_summary_edit():
    st.session_state.prd_data["intro_data"]["business_goal"] = st.session_state.summary_business_goal
    st.session_state.prd_data["hypothesis"]["Statement"] = st.session_state.summary_hypothesis
    st.session_state.prd_data["intro_data"]["user_persona"] = st.session_state.summary_user_persona
    st.session_state.prd_data["intro_data"]["app_description"] = st.session_state.summary_app_description
    st.session_state.editing_section = None
    st.success("Executive Summary updated!")

# --- Dialogs (Streamlit dialogs) ---
@st.dialog("Edit Section")
def edit_section_dialog(section_title):
    content = st.session_state.prd_data["prd_sections"].get(section_title, "")
    cleaned_label = section_title.replace("_", " ").title()
    st.text_area(f"Edit {cleaned_label}", value=format_content_for_display(content), height=300, key=f"text_area_{section_title}")
    st.caption("You can use Markdown for formatting (e.g., **bold**, *italics*, - lists).")
    if st.button("Save Changes", key=f"save_dialog_{section_title}"):
        save_edit(section_title)
        st.rerun()

@st.dialog("Edit Risk")
def edit_risk_dialog(risk_index):
    risk_item = st.session_state.prd_data["risks"][risk_index]
    st.text_area("Risk", value=risk_item.get('risk',''), height=100, key=f"text_area_risk_{risk_index}")
    st.text_area("Mitigation", value=risk_item.get('mitigation',''), height=100, key=f"text_area_mitigation_{risk_index}")
    if st.button("Save Changes", key=f"save_dialog_risk_{risk_index}"):
        save_risk_edit(risk_index)
        st.rerun()

@st.dialog("Edit Executive Summary")
def edit_summary_dialog():
    prd = st.session_state.prd_data
    st.text_input("Business Goal", value=prd['intro_data'].get('business_goal', ''), key="summary_business_goal")
    st.text_area("Hypothesis", value=prd['hypothesis'].get('Statement', ''), key="summary_hypothesis")
    st.text_area("Target User Persona (Optional)", value=prd['intro_data'].get('user_persona', ''), key="summary_user_persona")
    st.text_area("App Description (Optional)", value=prd['intro_data'].get('app_description', ''), key="summary_app_description")
    if st.button("Save Changes", key="save_summary_dialog"):
        save_summary_edit()
        st.rerun()

# --- Header / Topbar ---
def render_header():
    st.markdown("""
        <div class="app-header">
            <h1>A/B Test PRD Generator</h1>
            <p>Create AI-powered PRDs and experiment plans, fast.</p>
        </div>
    """, unsafe_allow_html=True)

def render_topbar():
    current_stage_index = STAGES.index(st.session_state.stage)
    button_html_list = []
    for i, stage in enumerate(STAGES):
        class_list = "nav-button"
        if i == current_stage_index:
            class_list += " active-stage"
        elif i < current_stage_index:
            class_list += " complete-stage"
        button_html_list.append(f'<div class="{class_list}">{stage}</div>')
    all_buttons_html = "".join(button_html_list)
    st.markdown(f'<div class="top-nav">{all_buttons_html}</div>', unsafe_allow_html=True)

# --- Wrapper to call generate_content in a compatibility-safe way ---
def _call_generate_content(user_inputs: Dict[str, Any], mode: str):
    """
    Try calling the newer generate_content signature first:
      generate_content(mode=..., user_inputs=..., use_rag=..., ...)
    Fallback to older signature generate_content(api_key, data, mode)
    """
    use_rag = st.session_state.get("use_rag", False)
    # Common args for new signature
    try:
        # Try new-style call (supported by the updated api_handler replacement)
        resp = generate_content(
            mode=mode,
            user_inputs=user_inputs,
            use_rag=use_rag,
        )
        # If the wrapper returns the older shape (dict with 'error'), keep it as-is
        return resp
    except TypeError:
        # Fallback to legacy signature: generate_content(api_key, data, mode)
        try:
            # If streamlit secrets contain key, pass it; otherwise pass None
            api_key = None
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                api_key = None
            return generate_content(api_key, user_inputs, mode)
        except Exception as e:
            return {"error": f"generate_content call failed: {e}"}
    except Exception as e:
        return {"error": f"generate_content unexpected error: {e}"}

# --- Page renderers (Intro, Hypothesis, PRD, Calculations, Review) ---
def render_intro_page():
    st.header("Step 1: The Basics 📝")
    st.info("Provide high-level details about your A/B test. Better context → better outputs.")

    # RAG toggle in the intro step (opt-in; default remains off)
    st.checkbox("Enable RAG (use previous PRDs to ground suggestions)", value=st.session_state.use_rag, key="use_rag")

    if "GROQ_API_KEY" not in st.secrets:
        st.warning("Groq API key not found in Streamlit secrets. LLM features will fallback to placeholders.")

    def process_intro_form():
        st.session_state.prd_data["intro_data"]["business_goal"] = st.session_state.intro_business_goal
        st.session_state.prd_data["intro_data"]["key_metric"] = st.session_state.intro_key_metric
        st.session_state.prd_data["intro_data"]["product_area"] = st.session_state.intro_product_area
        st.session_state.prd_data["intro_data"]["metric_type"] = st.session_state.intro_metric_type
        st.session_state.prd_data["intro_data"]["current_value"] = st.session_state.intro_current_value
        st.session_state.prd_data["intro_data"]["target_value"] = st.session_state.intro_target_value
        st.session_state.prd_data["intro_data"]["dau"] = st.session_state.intro_dau
        st.session_state.prd_data["intro_data"]["product_type"] = st.session_state.intro_product_type
        st.session_state.prd_data["intro_data"]["user_persona"] = st.session_state.intro_user_persona
        st.session_state.prd_data["intro_data"]["app_description"] = st.session_state.intro_app_description

        if st.session_state.get("intro_metric_type") == "Continuous":
            st.session_state.prd_data["intro_data"]["std_dev"] = st.session_state.get("intro_std_dev")

        required_fields = ["business_goal", "key_metric", "metric_type", "current_value", "product_area", "target_value", "dau", "product_type"]
        if st.session_state.prd_data["intro_data"]["metric_type"] == "Continuous":
            required_fields.append("std_dev")

        if all(st.session_state.prd_data["intro_data"].get(field) for field in required_fields):
            with st.spinner("Generating hypotheses..."):
                resp = _call_generate_content(st.session_state.prd_data["intro_data"], "hypotheses")
                if isinstance(resp, dict) and resp.get("error"):
                    st.error(resp.get("error"))
                else:
                    # Accept multiple response shapes:
                    # - Old-style: a dict of hypotheses
                    # - New-style: a dict with keys like {ok, parsed, raw_text}
                    if isinstance(resp, dict) and resp.get("parsed"):
                        st.session_state.hypotheses = resp.get("parsed")
                    else:
                        st.session_state.hypotheses = resp
                    next_stage()
        else:
            st.error("Please fill out all required fields to continue.")

    with st.form("intro_form"):
        st.subheader("Business & Product Details")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Business Goal", placeholder="e.g., Increase new user activation", key="intro_business_goal")
            st.text_input("Key Metric", placeholder="e.g., Signup -> First Action conversion", key="intro_key_metric")
            st.selectbox("Metric Type", ["Proportion", "Continuous"], key="intro_metric_type")
            st.number_input("Current Metric Value", min_value=0.0, value=50.0, key="intro_current_value")
            if st.session_state.get("intro_metric_type") == "Continuous":
                st.number_input("Standard Deviation", min_value=0.0, value=10.0, key="intro_std_dev", help="Std dev for continuous metric")
        with col2:
            st.text_input("Product Area", placeholder="e.g., New user onboarding flow", key="intro_product_area")
            st.number_input("Target Metric Value", min_value=0.0, value=55.0, key="intro_target_value")
            st.number_input("Daily Active Users (DAU)", min_value=100, value=10000, key="intro_dau")
            st.selectbox("Product Type", ["SaaS Product", "Mobile App", "Web Platform", "Other"], index=1, key="intro_product_type")

        st.subheader("Optional Context")
        st.text_area("Target User Persona (Optional)", placeholder="e.g., Tech-savvy millennials...", key="intro_user_persona")
        st.text_area("App Description (Optional)", placeholder="Short description of the product", key="intro_app_description")

        st.form_submit_button("Generate Hypotheses", on_click=process_intro_form)

def render_hypothesis_page():
    st.header("Step 2: Hypotheses 🧠")
    st.info("Select a suggested hypothesis or write your own and have the system enrich it.")

    def select_hypothesis(hypothesis_data):
        st.session_state.prd_data["hypothesis"] = hypothesis_data
        st.session_state.hypotheses_selected = True
        st.success(f"Selected: {hypothesis_data.get('Statement', str(hypothesis_data)[:120])}")
        next_stage()

    def generate_from_custom():
        custom_hypothesis = st.session_state.get("custom_hypothesis_input", "")
        if not custom_hypothesis:
            st.error("Please write a custom hypothesis first.")
            return
        with st.spinner("Enriching your hypothesis..."):
            context = {"custom_hypothesis": custom_hypothesis, **st.session_state.prd_data["intro_data"]}
            resp = _call_generate_content(context, "enrich_hypothesis")
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
            else:
                # normalize different shapes
                if isinstance(resp, dict) and resp.get("parsed"):
                    enriched = resp.get("parsed")
                else:
                    enriched = resp
                st.session_state.custom_hypothesis_generated = enriched
                st.success("Custom hypothesis enriched!")

    def lock_custom_hypothesis():
        enriched = st.session_state.get("custom_hypothesis_generated")
        if enriched:
            st.session_state.prd_data["hypothesis"] = enriched
            st.session_state.hypotheses_selected = True
            st.success("Custom hypothesis locked!")
            next_stage()
        else:
            st.error("No enriched hypothesis available to lock.")

    st.subheader("Write Your Own Hypothesis")
    st.text_area("Your Custom Hypothesis", placeholder="e.g., I hypothesize that...", key="custom_hypothesis_input")
    st.button("Generate from Custom", on_click=generate_from_custom)

    st.write("---")
    st.subheader("Or, Select from our suggestions")
    hypotheses = st.session_state.get("hypotheses", {})
    if isinstance(hypotheses, dict) and hypotheses:
        cols = st.columns(max(1, len(hypotheses)))
        for i, (k, v) in enumerate(hypotheses.items()):
            with cols[i]:
                st.subheader(f"Hypothesis {i+1}")
                st.markdown(f"**Statement:** {v.get('Statement', 'N/A')}")
                st.markdown(f"**Rationale:** {v.get('Rationale', 'N/A')}")
                st.markdown(f"**Behavioral Basis:** {v.get('Behavioral Basis', 'N/A')}")
                st.button("Select & Continue", key=f"select_{i}", on_click=select_hypothesis, args=(v,))
    else:
        st.info("No hypotheses available yet. Go back to Intro and re-generate.")

def render_prd_page():
    st.header("Step 3: PRD Draft ✍️")
    st.info("Review and edit the generated PRD sections.")

    if not st.session_state.prd_data.get("prd_sections"):
        with st.spinner("Drafting PRD sections..."):
            prd_context = {**st.session_state.prd_data["intro_data"], **(st.session_state.prd_data.get("hypothesis") or {})}
            resp = _call_generate_content(prd_context, "prd_sections")
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
            else:
                if isinstance(resp, dict) and resp.get("parsed"):
                    prd_sections = resp.get("parsed")
                else:
                    prd_sections = resp
                # Normalize Implementation_Plan list if string provided with newlines/hyphens
                for k, v in prd_sections.items():
                    if isinstance(v, str) and k == "Implementation_Plan":
                        prd_sections[k] = [line.strip("- ").strip() for line in v.splitlines() if line.strip()]
                st.session_state.prd_data["prd_sections"] = prd_sections

    prd_sections = st.session_state.prd_data.get("prd_sections", {})
    for key, content in prd_sections.items():
        cleaned_label = key.replace("_", " ").title()
        if st.session_state.editing_section == key:
            edit_section_dialog(key)
        with st.container():
            col1, col2 = st.columns([10, 1])
            with col1:
                st.subheader(cleaned_label)
                st.markdown(format_content_for_display(content))
            with col2:
                st.button("✏️", key=f"edit_{key}", on_click=set_editing_section, args=(key,))

    st.write("---")
    st.button("Save & Continue to Calculations", on_click=next_stage, key="to_calcs")

def render_calculations_page():
    st.header("Step 4: Experiment Calculations 📊")
    st.info("Verify inputs and calculate required sample size and duration.")

    intro = st.session_state.prd_data.get("intro_data", {})
    dau = intro.get("dau", 10000)
    current_value = intro.get("current_value", 50.0)
    metric_type = intro.get("metric_type", "Proportion")

    st.subheader("Key Metrics")
    st.markdown(f"**Key Metric:** {intro.get('key_metric', 'N/A')}")
    st.markdown(f"**Metric Type:** {metric_type}")
    st.markdown(f"**Current Value:** {current_value}")
    if metric_type == "Continuous":
        st.markdown(f"**Standard Deviation:** {intro.get('std_dev', 'N/A')}")

    st.subheader("Experiment Parameters")
    st.slider("Confidence Level (%)", 50, 99, 95, 1, key="calc_confidence")
    st.slider("Power Level (%)", 50, 99, 80, 1, key="calc_power")
    st.slider("Coverage (%)", 5, 100, 50, 5, key="calc_coverage")
    st.number_input("Minimum Detectable Effect (%)", min_value=0.1, value=5.0, step=0.1, key="calc_mde")

    def perform_calculations():
        try:
            st.session_state.prd_data["calculations"]["confidence"] = st.session_state.calc_confidence / 100
            st.session_state.prd_data["calculations"]["power"] = st.session_state.calc_power / 100
            st.session_state.prd_data["calculations"]["coverage"] = st.session_state.calc_coverage
            st.session_state.prd_data["calculations"]["min_detectable_effect"] = st.session_state.calc_mde

            if metric_type == "Proportion":
                sample_size = calculate_sample_size_proportion(current_value, st.session_state.calc_mde, st.session_state.calc_confidence / 100, st.session_state.calc_power / 100)
            else:
                sample_size = calculate_sample_size_continuous(current_value, intro.get("std_dev"), st.session_state.calc_mde, st.session_state.calc_confidence / 100, st.session_state.calc_power / 100)

            duration = calculate_duration(sample_size, dau, st.session_state.calc_coverage)
            st.session_state.prd_data["calculations"]["sample_size"] = int(sample_size)
            st.session_state.prd_data["calculations"]["duration"] = int(duration)
            st.success("Calculations complete!")
        except Exception as e:
            st.error(f"Error in calculations: {e}")

    st.button("Calculate", on_click=perform_calculations, key="calc_btn")

    if "sample_size" in st.session_state.prd_data.get("calculations", {}):
        st.subheader("Results")
        sample_size = st.session_state.prd_data['calculations']['sample_size']
        duration = st.session_state.prd_data['calculations']['duration']
        st.info(f"**Required Sample Size per Variant:** {sample_size:,}")
        st.info(f"**Estimated Experiment Duration:** {duration} days")
        st.button("Continue to Final Review", on_click=next_stage, key="to_review")

def render_final_review_page():
    st.header("Step 5: Final Review & Export 🎉")
    st.info("Review the PRD, polish text, generate risks, and export or persist.")

    prd = st.session_state.prd_data

    if st.session_state.editing_section == "executive_summary":
        edit_summary_dialog()

    with st.container():
        col1, col2 = st.columns([10, 1])
        with col1:
            st.subheader("🚀 Executive Summary")
            st.markdown(f"**Business Goal:** {prd['intro_data'].get('business_goal', 'N/A')}")
            st.markdown(f"**Hypothesis:** {prd['hypothesis'].get('Statement', 'N/A')}")
            st.markdown(f"**Success Criteria:** {prd['intro_data'].get('key_metric', 'N/A')} → {prd['intro_data'].get('target_value', 'N/A')}")
            if prd['intro_data'].get('user_persona'):
                st.markdown(f"**Target User Persona:** {prd['intro_data']['user_persona']}")
        with col2:
            st.button("✏️ Edit Summary", key="edit_summary", on_click=set_editing_section, args=("executive_summary",))

    st.subheader("PRD Sections")
    for key, content in prd.get('prd_sections', {}).items():
        display_label = key.replace("_", " ").title()
        if st.session_state.editing_section == key:
            edit_section_dialog(key)
        with st.container():
            col1, col2 = st.columns([10, 1])
            with col1:
                st.subheader(display_label)
                st.markdown(format_content_for_display(content))
            with col2:
                st.button("✏️", key=f"edit_review_{key}", on_click=set_editing_section, args=(key,))

    with st.container():
        st.subheader("Experiment Metrics Dashboard 📊")
        calc = prd.get("calculations", {})
        cols = st.columns(3)
        cols[0].metric("Confidence", f"{int(calc.get('confidence',0)*100)}%")
        cols[1].metric("Power", f"{int(calc.get('power',0)*100)}%")
        cols[2].metric("Min. Detectable Effect", f"{calc.get('min_detectable_effect','N/A')}%")

    with st.container():
        st.subheader("Risks & Next Steps ⚠️")

        def generate_risks():
            with st.spinner("Generating contextual risks..."):
                risk_data = {**prd['intro_data'], "hypothesis": prd['hypothesis'].get('Statement')}
                resp = _call_generate_content(risk_data, "risks")
                if isinstance(resp, dict) and resp.get("error"):
                    st.error(resp.get("error"))
                else:
                    if isinstance(resp, dict) and resp.get("parsed"):
                        risk_obj = resp.get("parsed")
                    else:
                        risk_obj = resp
                    if isinstance(risk_obj, dict) and "risks" in risk_obj:
                        st.session_state.prd_data["risks"] = risk_obj.get("risks", [])
                    else:
                        # handle case where LLM returned a list directly
                        if isinstance(risk_obj, list):
                            st.session_state.prd_data["risks"] = risk_obj
                        else:
                            st.error("Unexpected risk output format from LLM.")

        st.button("Generate Risks & Next Steps", on_click=generate_risks)

        for i, r in enumerate(st.session_state.prd_data.get("risks", [])):
            if st.session_state.editing_risk == i:
                edit_risk_dialog(i)
            with st.container():
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.subheader(f"Risk {i+1}")
                    st.markdown(f"**Description:** {r.get('risk','')}")
                    st.markdown(f"**Mitigation:** {r.get('mitigation','')}")
                with col2:
                    st.button("✏️", key=f"edit_risk_{i}", on_click=set_editing_risk, args=(i,))

    # Export & Persistence
    st.write("---")
    with st.container():
        col1, col2 = st.columns([3, 3])
        with col1:
            pdf_bytes = create_pdf(prd)
            st.download_button("📥 Download PRD as PDF", pdf_bytes, "AB_Testing_PRD.pdf", "application/pdf")
        with col2:
            if PERSISTENCE_AVAILABLE:
                def _save_prd_to_db():
                    try:
                        # Compose a clean PRD dict to save; include metadata
                        save_payload = {
                            "intro_data": prd.get("intro_data", {}),
                            "hypothesis": prd.get("hypothesis", {}),
                            "prd_sections": prd.get("prd_sections", {}),
                            "calculations": prd.get("calculations", {}),
                            "risks": prd.get("risks", []),
                            "saved_with_rag": st.session_state.get("use_rag", False),
                        }
                        prd_id = save_prd(save_payload, title=save_payload["intro_data"].get("business_goal", None), actor="streamlit_user")
                        st.session_state.saved_prd_info = {"id": prd_id, "title": save_payload["intro_data"].get("business_goal", "")}
                        st.success(f"PRD saved (id={prd_id})")
                    except Exception as e:
                        st.error(f"Failed to save PRD: {e}")
                st.button("💾 Save PRD to DB", on_click=_save_prd_to_db)
                if st.session_state.saved_prd_info:
                    info = st.session_state.saved_prd_info
                    st.markdown(f"**Last saved PRD:** {info.get('title','')} (id: `{info.get('id')}`)")
                    if st.button("View Saved PRDs"):
                        try:
                            rows = list_prds(limit=50)
                            st.session_state._saved_prds_list = rows
                        except Exception as e:
                            st.error(f"Failed to load saved PRDs: {e}")
            else:
                st.info("Persistence layer not installed. Add utils/persistence.py to enable saving PRDs.")

# --- Main rendering logic ---
render_header()
render_topbar()

if st.session_state.stage == "Intro":
    render_intro_page()
elif st.session_state.stage == "Hypothesis":
    render_hypothesis_page()
elif st.session_state.stage == "PRD":
    render_prd_page()
elif st.session_state.stage == "Calculations":
    render_calculations_page()
elif st.session_state.stage == "Review":
    render_final_review_page()

if st.session_state.get("scroll_to_top"):
    scroll_to_top()
    st.session_state.scroll_to_top = False

# --- Optional: show saved PRDs list when requested (non-blocking) ---
if st.session_state.get("_saved_prds_list"):
    st.write("---")
    st.subheader("Saved PRDs")
    saved = st.session_state.get("_saved_prds_list", [])
    for r in saved:
        st.markdown(f"- **{r['title']}** — id: `{r['id']}` — created: {r['created_at']}")
        if st.button(f"Load {r['id']}", key=f"load_{r['id']}"):
            # Load PRD into session (safe)
            try:
                loaded = get_prd(r['id'])
                if loaded:
                    st.session_state.prd_data = loaded["prd"]
                    st.success(f"Loaded PRD {r['id']}")
                else:
                    st.error("Could not find PRD.")
            except Exception as e:
                st.error(f"Failed to load PRD: {e}")

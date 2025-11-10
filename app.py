# app.py
"""
Streamlit front-end for the A/B Test PRD Generator (UPDATED to optionally use FastAPI proxy).

This is a complete app.py replacement. Key points:
- Preserves all original flows (Intro -> Hypothesis -> PRD -> Calculations -> Review).
- If a FastAPI proxy is available (configured via env var FASTAPI_PROXY_URL or Streamlit secret), the app will
  call that proxy for generation and (optionally) for saving PRDs. Otherwise it falls back to local generation
  (existing generate_content function or placeholder).
- All features previously added remain: RAG toggle, Save PRD (local persistence), PDF export, editing dialogs.
- Non-blocking and defensive: if remote proxy fails, app falls back to local handlers and surfaces errors to the user.

Drop this file into the repo root and run `streamlit run app.py`.
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import streamlit as st
import streamlit.components.v1 as components

# Networking
import requests

# --- Try to import local utilities, fall back gracefully ---
try:
    from api_handler import generate_content  # may exist or be our modified api_handler replacement
    LOCAL_GENERATE_AVAILABLE = True
except Exception:
    generate_content = None
    LOCAL_GENERATE_AVAILABLE = False

try:
    from utils.persistence import init_db, save_prd as persistence_save_prd, list_prds, get_prd, search_prds, get_audit_for_prd
    PERSISTENCE_AVAILABLE = True
except Exception:
    persistence_save_prd = None
    list_prds = None
    get_prd = None
    search_prds = None
    get_audit_for_prd = None
    PERSISTENCE_AVAILABLE = False

try:
    from utils.pdf_generator import create_pdf
    PDF_AVAILABLE = True
except Exception:
    def create_pdf(prd):
        return b"PDF placeholder"
    PDF_AVAILABLE = False

# calculation functions (import or fallback)
try:
    from calculations import calculate_sample_size_proportion, calculate_sample_size_continuous, calculate_duration
except Exception:
    # simple placeholders to keep UI functional
    def calculate_sample_size_proportion(current_value, mde, confidence, power):
        return 1000
    def calculate_sample_size_continuous(mean, std_dev, mde, confidence, power):
        return 1200
    def calculate_duration(sample_size, dau, coverage):
        return max(1, int(sample_size / max(1, int(dau * coverage / 100))))

# --- Configuration: proxy detection ---
# Priority: environment var FASTAPI_PROXY_URL > streamlit secrets FASTAPI_PROXY_URL > None
_FASTAPI_PROXY_URL = os.environ.get("FASTAPI_PROXY_URL")
if not _FASTAPI_PROXY_URL:
    try:
        _FASTAPI_PROXY_URL = st.secrets.get("FASTAPI_PROXY_URL")
    except Exception:
        _FASTAPI_PROXY_URL = None

USE_PROXY_BY_DEFAULT = bool(_FASTAPI_PROXY_URL)
PROXY_URL = _FASTAPI_PROXY_URL

# Helper to call proxy endpoints
def _proxy_generate(mode: str, user_inputs: Dict[str, Any], use_rag: bool = False, k: int = 3, extra_instructions: Optional[str] = None, prompt_system_prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Call the FastAPI proxy /generate endpoint.
    Returns the proxy response dict mapped to the app's expected shape.
    """
    if not PROXY_URL:
        return {"error": "Proxy not configured."}
    url = PROXY_URL.rstrip("/") + "/generate"
    payload = {
        "mode": mode,
        "user_inputs": user_inputs,
        "use_rag": use_rag,
        "k": k,
        "extra_instructions": extra_instructions,
        "prompt_system_prefix": prompt_system_prefix
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.RequestException as e:
        return {"error": f"Proxy call failed: {e}"}
    except Exception as e:
        return {"error": f"Proxy parsing failed: {e}"}

def _proxy_save_prd(prd: Dict[str, Any], title: Optional[str] = None, tags: Optional[List[str]] = None, actor: str = "streamlit_user", index_now: bool = False) -> Dict[str, Any]:
    """
    Call the FastAPI proxy /save_prd endpoint to persist server-side.
    Returns {ok: bool, prd_id: str or None, error: str or None}
    """
    if not PROXY_URL:
        return {"ok": False, "error": "Proxy not configured."}
    url = PROXY_URL.rstrip("/") + "/save_prd"
    payload = {"prd": prd, "title": title, "tags": tags or [], "actor": actor, "index_now": index_now}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"ok": False, "error": f"Proxy save failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Proxy save parsing failed: {e}"}

# --- UI helpers & session state init ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
def _init_session_state():
    s = st.session_state
    if "stage" not in s:
        s.stage = "Intro"
    if "prd_data" not in s:
        s.prd_data = {"intro_data": {}, "hypothesis": {}, "prd_sections": {}, "calculations": {}, "risks": []}
    if "hypotheses" not in s:
        s.hypotheses = {}
    if "hypotheses_selected" not in s:
        s.hypotheses_selected = False
    if "editing_section" not in s:
        s.editing_section = None
    if "editing_risk" not in s:
        s.editing_risk = None
    if "use_rag" not in s:
        s.use_rag = USE_PROXY_BY_DEFAULT  # default to proxy preference
    if "saved_prd_info" not in s:
        s.saved_prd_info = None
    if "scroll_to_top" not in s:
        s.scroll_to_top = False

_init_session_state()

STAGES = ["Intro", "Hypothesis", "PRD", "Calculations", "Review"]

def next_stage():
    idx = STAGES.index(st.session_state.stage)
    if idx < len(STAGES)-1:
        st.session_state.stage = STAGES[idx+1]
        st.session_state.scroll_to_top = True

def prev_stage():
    idx = STAGES.index(st.session_state.stage)
    if idx > 0:
        st.session_state.stage = STAGES[idx-1]
        st.session_state.scroll_to_top = True

def scroll_to_top():
    components.html("<script>window.scrollTo(0,0)</script>")

# small formatting helpers
def format_content_for_display(content):
    if isinstance(content, list):
        return "\n".join([f"- {item}" for item in content])
    else:
        return str(content)

# --- Minimal CSS for polish ---
st.markdown("""
<style>
    html, body, [class*="st-"] { font-family: Inter, Roboto, sans-serif; }
    .stDownloadButton>button { background-color:#216d33; color:white; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# --- Generation wrapper that prefers proxy when configured ---
def _generate(mode: str, user_inputs: Dict[str, Any]):
    """
    Unified generation entrypoint used everywhere in the app.
    Behavior:
    - If proxy configured: call proxy /generate.
    - Else if local generate_content available: call it.
    - Else return a helpful error / placeholder.
    """
    use_proxy = bool(PROXY_URL)
    use_rag = st.session_state.get("use_rag", False)
    if use_proxy:
        resp = _proxy_generate(mode=mode, user_inputs=user_inputs, use_rag=use_rag)
        # proxy returns structure similar to GenerateResponse; adapt
        if isinstance(resp, dict) and resp.get("error"):
            # proxy failed — fallback to local if possible
            if LOCAL_GENERATE_AVAILABLE:
                try:
                    return generate_content(mode=mode, user_inputs=user_inputs, use_rag=use_rag)
                except Exception as e:
                    return {"error": f"Proxy failed and local fallback failed: {e}"}
            return {"error": resp.get("error")}
        # success — proxy returns keys: ok, mode, raw_text, parsed, errors, metadata
        return resp
    else:
        # local fallback
        if LOCAL_GENERATE_AVAILABLE:
            try:
                return generate_content(mode=mode, user_inputs=user_inputs, use_rag=st.session_state.get("use_rag", False))
            except Exception as e:
                return {"error": f"Local generation failed: {e}"}
        else:
            return {"error": "No generation backend available (configure FASTAPI_PROXY_URL or provide local api_handler.generate_content)."}

# --- UI pages ---
def render_header():
    st.title("A/B Test PRD Generator")
    if PROXY_URL:
        st.caption(f"Using remote generation proxy: {PROXY_URL}")
    else:
        st.caption("Using local generation (no proxy configured).")

def render_intro_page():
    st.header("Step 1 — Basics")
    st.checkbox("Enable RAG (use previous PRDs to ground suggestions)", value=st.session_state.use_rag, key="use_rag")
    if not PROXY_URL and not LOCAL_GENERATE_AVAILABLE:
        st.warning("LLM generation not available. Add GROQ key and api_handler or configure FASTAPI_PROXY_URL.")
    with st.form("intro_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Business Goal", key="intro_business_goal", placeholder="Increase onboarding completion")
            st.text_input("Key Metric", key="intro_key_metric", placeholder="onboarding_completion_rate")
            st.selectbox("Metric Type", ["Proportion", "Continuous"], key="intro_metric_type")
            st.number_input("Current Metric Value", min_value=0.0, value=50.0, key="intro_current_value")
        with col2:
            st.text_input("Product Area", key="intro_product_area", placeholder="Onboarding")
            st.number_input("Target Metric Value", min_value=0.0, value=55.0, key="intro_target_value")
            st.number_input("DAU", min_value=1, value=10000, key="intro_dau")
            st.selectbox("Product Type", ["SaaS Product", "Mobile App", "Web Platform", "Other"], key="intro_product_type")
        st.text_area("Target User Persona (optional)", key="intro_user_persona")
        st.text_area("App Description (optional)", key="intro_app_description")
        submitted = st.form_submit_button("Generate Hypotheses")
        if submitted:
            # validate minimal fields
            required = ["intro_business_goal", "intro_key_metric", "intro_metric_type", "intro_current_value", "intro_target_value", "intro_dau"]
            ok = all(st.session_state.get(k) is not None and st.session_state.get(k) != "" for k in required)
            if not ok:
                st.error("Fill required fields before generating hypotheses.")
            else:
                # prepare inputs
                intro_data = {
                    "business_goal": st.session_state.intro_business_goal,
                    "key_metric": st.session_state.intro_key_metric,
                    "metric_type": st.session_state.intro_metric_type,
                    "current_value": st.session_state.intro_current_value,
                    "target_value": st.session_state.intro_target_value,
                    "dau": st.session_state.intro_dau,
                    "product_area": st.session_state.intro_product_area,
                    "product_type": st.session_state.intro_product_type,
                    "user_persona": st.session_state.intro_user_persona,
                    "app_description": st.session_state.intro_app_description
                }
                st.session_state.prd_data["intro_data"] = intro_data
                with st.spinner("Generating hypotheses..."):
                    resp = _generate("hypotheses", intro_data)
                if isinstance(resp, dict) and resp.get("error"):
                    st.error(resp.get("error"))
                    # show raw when available
                    if resp.get("raw_text"):
                        st.code(resp.get("raw_text")[:4000])
                else:
                    # normalize responses: proxy returns parsed or raw_text
                    parsed = resp.get("parsed") if isinstance(resp, dict) else None
                    if parsed:
                        st.session_state.hypotheses = parsed
                    else:
                        # try raw_text -> parse JSON if possible
                        raw = resp.get("raw_text") if isinstance(resp, dict) else resp
                        try:
                            parsed_raw = json.loads(raw)
                            st.session_state.hypotheses = parsed_raw
                        except Exception:
                            # fallback: store raw text in a simple dict
                            st.session_state.hypotheses = {"suggestion_1": {"Statement": raw}}
                    next_stage()

def render_hypothesis_page():
    st.header("Step 2 — Hypotheses")
    st.info("Select a suggested hypothesis or write your own to enrich it.")

    st.text_area("Custom Hypothesis (optional)", key="custom_hypothesis_input", height=120)
    if st.button("Enrich Custom Hypothesis"):
        custom = st.session_state.get("custom_hypothesis_input", "").strip()
        if not custom:
            st.error("Write a custom hypothesis first.")
        else:
            payload = {"custom_hypothesis": custom, **st.session_state.prd_data.get("intro_data", {})}
            with st.spinner("Enriching hypothesis..."):
                resp = _generate("enrich_hypothesis", payload)
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
            else:
                parsed = resp.get("parsed") if isinstance(resp, dict) else None
                enriched = parsed or resp.get("raw_text") or resp
                st.session_state.prd_data["hypothesis"] = enriched
                st.success("Custom hypothesis enriched and saved to session. Continue to PRD stage.")
                next_stage()

    st.write("---")
    st.subheader("Suggested Hypotheses")
    hyps = st.session_state.get("hypotheses", {})
    if isinstance(hyps, dict) and hyps:
        for i, (k, v) in enumerate(hyps.items()):
            st.markdown(f"**Hypothesis {i+1}**")
            if isinstance(v, dict):
                st.markdown(f"- **Statement:** {v.get('Statement') or v.get('hypothesis') or str(v)[:200]}")
                if v.get("Rationale") or v.get("rationale"):
                    st.markdown(f"- **Rationale:** {v.get('Rationale') or v.get('rationale')}")
            else:
                st.markdown(f"- {v}")
            if st.button(f"Select {i+1}", key=f"select_hyp_{i}"):
                st.session_state.prd_data["hypothesis"] = v
                st.success("Hypothesis selected.")
                next_stage()
    else:
        st.info("No suggested hypotheses found. Go back and provide more context, or write a custom one.")

def render_prd_page():
    st.header("Step 3 — PRD Draft")
    prd_sections = st.session_state.prd_data.get("prd_sections") or {}
    if not prd_sections:
        with st.spinner("Drafting PRD sections..."):
            context = {**st.session_state.prd_data.get("intro_data", {}), **({"hypothesis": st.session_state.prd_data.get("hypothesis")} or {})}
            resp = _generate("prd_sections", context)
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
            else:
                parsed = resp.get("parsed") if isinstance(resp, dict) else None
                prd_sections = parsed or (resp.get("raw_text") if isinstance(resp, dict) else resp) or {}
                # crude normalization
                if isinstance(prd_sections, str):
                    prd_sections = {"Draft": prd_sections}
                st.session_state.prd_data["prd_sections"] = prd_sections

    for key, content in st.session_state.prd_data.get("prd_sections", {}).items():
        st.subheader(key.replace("_"," ").title())
        st.markdown(format_content_for_display(content))
        if st.button(f"Edit {key}", key=f"edit_{key}"):
            st.session_state.editing_section = key
            st.experimental_rerun()

    if st.session_state.editing_section:
        sec = st.session_state.editing_section
        val = st.session_state.prd_data["prd_sections"].get(sec, "")
        new_val = st.text_area(f"Edit section: {sec}", value=format_content_for_display(val), height=240, key=f"edit_area_{sec}")
        if st.button("Save section", key=f"save_sec_{sec}"):
            # write back: if original was list, convert lines
            orig = st.session_state.prd_data["prd_sections"].get(sec)
            if isinstance(orig, list):
                st.session_state.prd_data["prd_sections"][sec] = [line.strip("- ").strip() for line in new_val.splitlines() if line.strip()]
            else:
                st.session_state.prd_data["prd_sections"][sec] = new_val
            st.session_state.editing_section = None
            st.success("Section saved.")

    st.write("---")
    if st.button("Save & Continue"):
        next_stage()

def render_calculations_page():
    st.header("Step 4 — Calculations")
    intro = st.session_state.prd_data.get("intro_data", {})
    dau = intro.get("dau", 10000)
    current = intro.get("current_value", 50.0)
    metric_type = intro.get("metric_type", "Proportion")
    st.markdown(f"**Metric:** {intro.get('key_metric','N/A')} ({metric_type})")
    st.number_input("Confidence (%)", min_value=50, max_value=99, value=95, key="calc_confidence")
    st.number_input("Power (%)", min_value=50, max_value=99, value=80, key="calc_power")
    st.number_input("Coverage (%)", min_value=1, max_value=100, value=50, key="calc_coverage")
    st.number_input("Min Detectable Effect (%)", min_value=0.1, value=5.0, step=0.1, key="calc_mde")
    if st.button("Calculate"):
        confidence = st.session_state.calc_confidence/100.0
        power = st.session_state.calc_power/100.0
        coverage = st.session_state.calc_coverage
        mde = st.session_state.calc_mde
        try:
            if metric_type == "Proportion":
                sample = calculate_sample_size_proportion(current, mde, confidence, power)
            else:
                sample = calculate_sample_size_continuous(current, intro.get("std_dev", 1.0), mde, confidence, power)
            duration = calculate_duration(sample, dau, coverage)
            st.session_state.prd_data["calculations"] = {"sample_size": int(sample), "duration": int(duration), "confidence": confidence, "power": power, "coverage": coverage, "min_detectable_effect": mde}
            st.success("Calculations saved in session.")
        except Exception as e:
            st.error(f"Calculation error: {e}")
    if st.session_state.prd_data.get("calculations"):
        calc = st.session_state.prd_data["calculations"]
        st.info(f"Sample/variant: {calc.get('sample_size')}, Duration: {calc.get('duration')} days")
    if st.button("Continue to Review"):
        next_stage()

def render_review_page():
    st.header("Step 5 — Review & Export")
    prd = st.session_state.prd_data
    st.subheader("Executive summary")
    st.markdown(f"**Business Goal:** {prd.get('intro_data',{}).get('business_goal','')}")
    st.markdown(f"**Hypothesis:** {prd.get('hypothesis')}")
    st.subheader("PRD sections")
    for k,v in prd.get("prd_sections",{}).items():
        st.markdown(f"### {k.replace('_',' ').title()}")
        st.markdown(format_content_for_display(v))
    st.subheader("Calculations")
    st.write(prd.get("calculations", {}))
    st.subheader("Risks")
    for i, r in enumerate(prd.get("risks", [])):
        st.markdown(f"- **Risk:** {r.get('risk')} — **Mitigation:** {r.get('mitigation')}")
    # Generate risks via LLM
    if st.button("Generate Risks"):
        payload = {**prd.get("intro_data", {}), "hypothesis": prd.get("hypothesis")}
        with st.spinner("Generating risks..."):
            resp = _generate("risks", payload)
        if isinstance(resp, dict) and resp.get("error"):
            st.error(resp.get("error"))
        else:
            parsed = resp.get("parsed") if isinstance(resp, dict) else None
            risks = parsed or resp.get("raw_text") or resp
            # try to normalize risks to list
            if isinstance(risks, dict) and "risks" in risks:
                st.session_state.prd_data["risks"] = risks["risks"]
            elif isinstance(risks, list):
                st.session_state.prd_data["risks"] = risks
            else:
                st.info("Received risks; please review.")
    # PDF download
    pdf_bytes = create_pdf(prd)
    st.download_button("Download PRD as PDF", pdf_bytes, file_name="PRD.pdf", mime="application/pdf")
    # Save PRD: choose local persistence or proxy save
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if PERSISTENCE_AVAILABLE and not PROXY_URL:
            def _save_local():
                try:
                    save_payload = {
                        "intro_data": prd.get("intro_data", {}),
                        "hypothesis": prd.get("hypothesis", {}),
                        "prd_sections": prd.get("prd_sections", {}),
                        "calculations": prd.get("calculations", {}),
                        "risks": prd.get("risks", []),
                        "saved_with_rag": st.session_state.get("use_rag", False)
                    }
                    pid = persistence_save_prd(save_payload, title=save_payload["intro_data"].get("business_goal"))
                    st.session_state.saved_prd_info = {"id": pid, "title": save_payload["intro_data"].get("business_goal","")}
                    st.success(f"Saved locally with id: {pid}")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            st.button("Save PRD to local DB", on_click=_save_local)
        elif PROXY_URL:
            index_now = st.checkbox("Index now (add to RAG immediately)", value=False, key="index_now_flag")
            def _save_remote():
                save_payload = {
                    "intro_data": prd.get("intro_data", {}),
                    "hypothesis": prd.get("hypothesis", {}),
                    "prd_sections": prd.get("prd_sections", {}),
                    "calculations": prd.get("calculations", {}),
                    "risks": prd.get("risks", []),
                    "saved_with_rag": st.session_state.get("use_rag", False)
                }
                resp = _proxy_save_prd(save_payload, title=save_payload["intro_data"].get("business_goal"), index_now=index_now)
                if resp.get("ok"):
                    st.session_state.saved_prd_info = {"id": resp.get("prd_id"), "title": save_payload["intro_data"].get("business_goal","")}
                    st.success(f"Saved via proxy: {resp.get('prd_id')}")
                else:
                    st.error(f"Remote save failed: {resp.get('error')}")
            st.button("Save PRD via Proxy", on_click=_save_remote)
        else:
            st.info("Persistence not available. Add utils/persistence.py or configure proxy to enable saving.")
    with col2:
        if st.session_state.saved_prd_info:
            st.markdown(f"**Last saved PRD:** {st.session_state.saved_prd_info.get('title')} (id: {st.session_state.saved_prd_info.get('id')})")
            if st.button("View saved PRDs"):
                if PERSISTENCE_AVAILABLE:
                    rows = list_prds(limit=50)
                    st.session_state._saved_prds_list = rows
                else:
                    st.error("Local persistence not available.")

# --- Main render flow ---
render_header()
if st.session_state.stage == "Intro":
    render_intro_page()
elif st.session_state.stage == "Hypothesis":
    render_hypothesis_page()
elif st.session_state.stage == "PRD":
    render_prd_page()
elif st.session_state.stage == "Calculations":
    render_calculations_page()
elif st.session_state.stage == "Review":
    render_review_page()

if st.session_state.scroll_to_top:
    scroll_to_top()
    st.session_state.scroll_to_top = False

# show saved list if loaded
if st.session_state.get("_saved_prds_list"):
    st.write("---")
    st.subheader("Saved PRDs")
    for r in st.session_state._saved_prds_list:
        st.markdown(f"- **{r['title']}** — id: `{r['id']}` — created: {r['created_at']}")
        if st.button(f"Load {r['id']}", key=f"load_{r['id']}"):
            try:
                loaded = get_prd(r['id'])
                if loaded:
                    st.session_state.prd_data = loaded["prd"]
                    st.success("Loaded PRD into session.")
                else:
                    st.error("Could not load PRD.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

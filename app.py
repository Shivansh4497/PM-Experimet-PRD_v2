# app.py
"""
Streamlit front-end for the A/B Test PRD Generator (UPDATED: domain-based demo corpus + session-scoped RAG).

This is a complete, drop-in replacement for app.py.
Features added/ensured:
- PRD Domain dropdown in Intro (e.g., Growth, Retention, Monetization, Platform, UX, Engagement, Infrastructure).
- On domain selection, the app seeds a **session-scoped** vector store with the read-only demo PRDs from data/demo_prds.json for that domain.
- Uploaded user PRDs are **never** added to the demo vector store. They can optionally be included **ephemerally** in the prompt (per-session, opt-in).
- Provenance: after each RAG call the UI shows the top-k demo snippets used and labels whether ephemeral user docs were included.
- Fully defensive: if utils.rag_prompting or external modules are missing, the app falls back to safe behaviors (local placeholders, no persistent indexing).
- All original flows preserved: Intro -> Hypothesis -> PRD -> Calculations -> Review.
- Compatible with optional FastAPI proxy via FASTAPI_PROXY_URL (like previous app), but will use local generate_content if proxy not configured.
- Session-level isolation: vector stores live in st.session_state and are removed after session ends.

Drop this file into your repo root and run:
    streamlit run app.py
"""

from __future__ import annotations
import os
import json
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

# networking (optional proxy)
import requests

# --- Try to import existing utilities gracefully ---
# generate_content might live in api_handler or via proxy
try:
    from api_handler import generate_content as local_generate_content
    LOCAL_GENERATE_AVAILABLE = True
except Exception:
    local_generate_content = None
    LOCAL_GENERATE_AVAILABLE = False

# rag_prompting helpers (for vector store, chunking, retrieval)
try:
    from utils import rag_prompting  # expected functions: build_default_vector_store, docs_to_docchunks, generate_with_rag, _derive_full_text_from_prd
    RAG_PROMPTING_AVAILABLE = True
except Exception:
    rag_prompting = None
    RAG_PROMPTING_AVAILABLE = False

# persistence + pdf generator (optional)
try:
    from utils.persistence import save_prd as persistence_save_prd, list_prds
    PERSISTENCE_AVAILABLE = True
except Exception:
    persistence_save_prd = None
    list_prds = None
    PERSISTENCE_AVAILABLE = False

try:
    from utils.pdf_generator import create_pdf
    PDF_AVAILABLE = True
except Exception:
    def create_pdf(prd):
        # basic placeholder PDF bytes
        return b"%PDF-1.4\n%placeholder\n"
    PDF_AVAILABLE = False

# calculations (optional)
try:
    from calculations import calculate_sample_size_proportion, calculate_sample_size_continuous, calculate_duration
except Exception:
    def calculate_sample_size_proportion(current_value, mde, confidence, power):
        return int(1000)
    def calculate_sample_size_continuous(mean, std_dev, mde, confidence, power):
        return int(1200)
    def calculate_duration(sample_size, dau, coverage):
        return max(1, int(sample_size / max(1, int(dau * coverage / 100))))

# --- Proxy config (optional) ---
_FASTAPI_PROXY_URL = os.environ.get("FASTAPI_PROXY_URL") or (st.secrets.get("FASTAPI_PROXY_URL") if hasattr(st, "secrets") else None)
PROXY_URL = _FASTAPI_PROXY_URL

def _proxy_generate(mode: str, user_inputs: Dict[str, Any], use_rag: bool = False, k: int = 3, extra_instructions: Optional[str] = None, prompt_system_prefix: Optional[str] = None, ephemeral_docs: Optional[str] = None) -> Dict[str, Any]:
    if not PROXY_URL:
        return {"error": "proxy_not_configured"}
    try:
        resp = requests.post(
            PROXY_URL.rstrip("/") + "/generate",
            json={
                "mode": mode,
                "user_inputs": user_inputs,
                "use_rag": use_rag,
                "k": k,
                "extra_instructions": extra_instructions,
                "prompt_system_prefix": prompt_system_prefix,
                "ephemeral_docs": ephemeral_docs
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"proxy_error: {e}"}

# --- App config & constants ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
DEMO_PRDS_PATH = os.environ.get("DEMO_PRDS_PATH", "data/demo_prds.json")  # ensure this file exists and contains the improved demo PRDs
DOMAINS = ["Growth", "Retention", "Monetization", "Platform", "UX", "Engagement", "Infrastructure"]
DEFAULT_DOMAIN = "Growth"
TOP_K = 3  # number of demo snippets to show/use in prompt

# --- Session state initialization ---
def _init_session_state():
    s = st.session_state
    if "stage" not in s:
        s.stage = "Intro"
    if "prd_data" not in s:
        s.prd_data = {"intro_data": {}, "hypothesis": {}, "prd_sections": {}, "calculations": {}, "risks": []}
    if "editing_section" not in s:
        s.editing_section = None
    if "editing_risk" not in s:
        s.editing_risk = None
    if "use_rag" not in s:
        s.use_rag = True
    if "demo_domain" not in s:
        s.demo_domain = DEFAULT_DOMAIN
    if "ephemeral_uploads" not in s:
        # list of dicts: {"title":..., "text":...}
        s.ephemeral_uploads = []
    if "include_ephemeral_in_prompt" not in s:
        s.include_ephemeral_in_prompt = False
    if "vector_store" not in s:
        s.vector_store = None  # session-scoped vector store seeded from demo PRDs
    if "seeded_domain" not in s:
        s.seeded_domain = None
    if "provenance" not in s:
        s.provenance = []  # list of retrieved snippets metadata for last generation
    if "hypotheses" not in s:
        s.hypotheses = {}
    if "hypotheses_selected" not in s:
        s.hypotheses_selected = False
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

# --- Utility: load demo PRDs file ---
def _load_demo_prds() -> List[Dict[str, Any]]:
    if not os.path.exists(DEMO_PRDS_PATH):
        return []
    try:
        with open(DEMO_PRDS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
            return docs
    except Exception:
        return []

# --- RAG: session vector store management ---
def _build_session_vector_store(use_chroma: bool = False, embedder_model: str = "all-MiniLM-L6-v2"):
    """
    Build a session-scoped vector store. Prefer utils.rag_prompting if available.
    The returned object should implement:
      - add_documents(doc_chunks)
      - similarity_search(query, k=TOP_K) -> list of (chunk_text, meta)
    If rag_prompting is not available, return a simple in-memory lexical fallback.
    """
    if RAG_PROMPTING_AVAILABLE:
        try:
            vs = rag_prompting.build_default_vector_store(persist_dir=None, use_chroma=False, embedder_model=embedder_model)
            return vs
        except Exception:
            # fallback to in-memory wrapper below
            pass

    # Fallback simple in-memory store with naive embedding: store chunks and do substring/word-overlap scoring
    class SimpleInMemoryVS:
        def __init__(self):
            self.chunks = []  # each: {"id": id, "text": text, "meta": meta}
        def add_documents(self, doc_chunks: List[Any]):
            # accept either rag_prompting-like DocChunk or simple dict with 'text' and 'meta'
            for c in doc_chunks:
                try:
                    text = c.text if hasattr(c, "text") else c.get("text", str(c))
                    meta = c.meta if hasattr(c, "meta") else c.get("meta", {})
                    cid = getattr(c, "id", meta.get("id", str(len(self.chunks))))
                except Exception:
                    text = str(c)
                    meta = {}
                    cid = str(len(self.chunks))
                self.chunks.append({"id": cid, "text": text, "meta": meta})
        def similarity_search(self, query: str, k: int = TOP_K):
            # very naive scoring: count overlap of words
            qwords = set([w.lower() for w in query.split()])
            scored = []
            for c in self.chunks:
                cwords = set([w.lower() for w in c["text"].split()])
                score = len(qwords & cwords)
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [ (item[1]["text"], item[1]["meta"], score) for score, item in scored[:k] ]
            return results
    return SimpleInMemoryVS()

def _prepare_docchunks_from_demo_prds(docs: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert each demo PRD into doc chunks. Prefer rag_prompting.docs_to_docchunks if available.
    Each chunk should ideally contain text and meta: {'text':..., 'meta': {'title':..., 'source_id':...}}
    """
    chunks = []
    if RAG_PROMPTING_AVAILABLE and hasattr(rag_prompting, "docs_to_docchunks"):
        for d in docs:
            source_id = d.get("id") or d.get("title", "")[:32]
            title = d.get("title", "")
            # derive full text if utility exists; else serialize key fields
            if hasattr(rag_prompting, "_derive_full_text_from_prd"):
                text = rag_prompting._derive_full_text_from_prd(d)
            else:
                # Compose a readable text blob
                parts = []
                if d.get("business_context"):
                    parts.append(d["business_context"])
                if d.get("problem_statement"):
                    parts.append("Problem: " + d["problem_statement"])
                if d.get("hypothesis"):
                    parts.append("Hypothesis: " + (d["hypothesis"] if isinstance(d["hypothesis"], str) else d["hypothesis"].get("Statement","")))
                parts.append("Implementation: " + " ".join(d.get("implementation_plan", d.get("prd_sections", {}).get("Implementation_Plan", [])) if isinstance(d.get("implementation_plan", []), list) else [d.get("implementation_plan","")]))
                text = "\n\n".join(parts)
            # use rag_prompting.docs_to_docchunks to create smaller chunks if available
            try:
                c = rag_prompting.docs_to_docchunks(source_id=source_id, title=title, text=text)
                chunks.extend(c)
            except Exception:
                # fallback: single chunk dict
                chunks.append({"id": source_id + "_0", "text": text, "meta": {"title": title, "source_id": source_id}})
    else:
        # simple chunking: create one chunk per PRD from key fields
        for d in docs:
            source_id = d.get("id") or d.get("title","")[:32]
            title = d.get("title", "")
            text = []
            for fld in ("business_context", "problem_statement", "hypothesis", "implementation_plan"):
                if fld in d:
                    val = d[fld]
                    if isinstance(val, list):
                        text.append(" ".join(val))
                    else:
                        text.append(str(val))
            blob = "\n\n".join([t for t in text if t])
            chunks.append({"id": source_id + "_0", "text": blob, "meta": {"title": title, "source_id": source_id}})
    return chunks

def seed_session_vector_store_for_domain(domain: str):
    """
    Seed st.session_state.vector_store with demo PRDs for the selected domain.
    This is session-scoped and read-only (users cannot alter the demo corpus).
    """
    domain = domain or DEFAULT_DOMAIN
    if st.session_state.seeded_domain == domain and st.session_state.vector_store is not None:
        return  # already seeded
    demo_docs = _load_demo_prds()
    domain_docs = [d for d in demo_docs if d.get("domain") == domain]
    if not domain_docs:
        # nothing to seed: set empty vector store
        st.session_state.vector_store = _build_session_vector_store()
        st.session_state.seeded_domain = domain
        return
    vs = _build_session_vector_store()
    chunks = _prepare_docchunks_from_demo_prds(domain_docs)
    try:
        vs.add_documents(chunks)
    except Exception:
        # attempt to normalize and add
        normalized = []
        for c in chunks:
            try:
                text = c.text if hasattr(c, "text") else c.get("text", str(c))
                meta = c.meta if hasattr(c, "meta") else c.get("meta", {})
                normalized.append({"text": text, "meta": meta, "id": getattr(c, "id", meta.get("source_id"))})
            except Exception:
                normalized.append({"text": str(c), "meta": {}, "id": str(time.time())})
        try:
            vs.add_documents(normalized)
        except Exception:
            # swallow; vector store may be minimal
            pass
    st.session_state.vector_store = vs
    st.session_state.seeded_domain = domain

# --- Generation wrapper: integrates RAG, ephemeral docs, and proxy/local generation ---
def _call_generation(mode: str, user_inputs: Dict[str, Any], use_rag: bool = True, k: int = TOP_K):
    """
    Unified generation entrypoint.
    Behavior:
    - If PROXY_URL is set, call proxy with ephemeral_docs included if selected.
    - Else attempt to use local_generate_content if available (which may accept use_rag and vector_store).
    - If local generate does not accept vector_store, try a best-effort approach: retrieve demo snippets locally and append to user_inputs as 'rag_context'.
    - Always populate st.session_state.provenance with top-k demo snippets used (if any).
    """
    provenance = []
    ephemeral_text = None
    if st.session_state.include_ephemeral_in_prompt and st.session_state.ephemeral_uploads:
        # join ephemeral uploads into a single string (capped)
        parts = []
        for u in st.session_state.ephemeral_uploads:
            txt = u.get("text","")
            if len(txt) > 8000:
                txt = txt[:8000] + " [TRUNCATED]"
            parts.append(f"{u.get('title','uploaded_doc')}:\n{txt}")
        ephemeral_text = "\n\n".join(parts)

    # If proxy present, delegate (proxy is expected to support ephemeral_docs arg)
    if PROXY_URL:
        resp = _proxy_generate(mode=mode, user_inputs=user_inputs, use_rag=use_rag, k=k, ephemeral_docs=ephemeral_text)
        # Extract provenance if present
        st.session_state.provenance = resp.get("metadata", {}).get("rag_context", []) if isinstance(resp, dict) else []
        return resp

    # Try local generation usage
    # Preferred path: local_generate_content supports use_rag and vector_store args.
    if LOCAL_GENERATE_AVAILABLE:
        try:
            # Try to call with vector_store if signature supports it
            try:
                # new-style: generate_content(mode=..., user_inputs=..., use_rag=..., vector_store=..., k=..., ephemeral_docs=...)
                resp = local_generate_content(mode=mode, user_inputs=user_inputs, use_rag=use_rag, vector_store=st.session_state.vector_store, k=k, ephemeral_docs=ephemeral_text)
                # Try to get provenance from resp
                if isinstance(resp, dict):
                    st.session_state.provenance = resp.get("metadata", {}).get("rag_context", []) or []
                return resp
            except TypeError:
                # Older signature fallback: generate_content(api_key, data, mode)
                try:
                    resp = local_generate_content(None, user_inputs, mode)
                    st.session_state.provenance = []
                    return resp
                except Exception as e:
                    return {"error": f"local_generate_content_fallback_failed: {e}"}
        except Exception as e:
            return {"error": f"local_generate_failed: {e}"}
    # No proxy and no local generator -> emulate RAG by retrieving top-k demo snippets and return a placeholder response
    # Retrieve demo snippets from session vector store (if available)
    provenance = []
    if st.session_state.vector_store and use_rag:
        try:
            query_text = " ".join([str(v) for v in user_inputs.values()][:8])  # short query
            search_results = st.session_state.vector_store.similarity_search(query_text, k=k)
            # normalize results into list of dicts: {"text":..., "meta": {...}, "score": ...}
            normalized = []
            for item in search_results:
                if isinstance(item, tuple) and len(item) == 3:
                    text, meta, score = item
                elif isinstance(item, tuple) and len(item) == 2:
                    text, meta = item
                    score = None
                elif isinstance(item, dict):
                    text = item.get("text")
                    meta = item.get("meta", {})
                    score = item.get("score")
                else:
                    text = str(item)
                    meta = {}
                    score = None
                normalized.append({"text": text, "meta": meta, "score": score})
            provenance = normalized
            st.session_state.provenance = provenance
        except Exception:
            st.session_state.provenance = []
    # Build a very basic placeholder "generated" object using inputs and provenance to help the UI continue functioning
    placeholder_text = "Generated output (placeholder)."
    if st.session_state.provenance:
        placeholder_text += "\n\nGrounding snippets:\n"
        for p in st.session_state.provenance[:k]:
            snippet_preview = (p.get("text")[:400] + "...") if p.get("text") else ""
            title = p.get("meta", {}).get("title") or p.get("meta", {}).get("source_id", "demo")
            placeholder_text += f"- {title}: {snippet_preview}\n"
    if ephemeral_text and st.session_state.include_ephemeral_in_prompt:
        placeholder_text += "\n\nEphemeral user documents included in prompt."
    return {"ok": True, "mode": mode, "raw_text": placeholder_text, "parsed": None, "metadata": {"rag_context": st.session_state.provenance}}

# --- UI: small CSS polish ---
st.markdown("""
<style>
html, body, [class*="st-"] { font-family: Inter, Roboto, sans-serif; }
.stDownloadButton>button { background-color:#216d33; color:white; border-radius:8px; }
.small-muted { color: #6b6b6b; font-size: 0.9rem; }
.provenance-box { background:#0b1220; padding:12px; border-radius:8px; color: #e6eef8; }
</style>
""", unsafe_allow_html=True)

# --- UI pages ---
def render_header():
    st.title("A/B Test PRD Generator — Domain-grounded Demo")
    st.caption("Demo corpus is read-only. Uploaded docs remain ephemeral and never alter the demo corpus.")

def render_topbar():
    cols = st.columns([1,3,1])
    with cols[0]:
        if st.button("← Back") and st.session_state.stage != "Intro":
            prev_stage()
    with cols[1]:
        st.markdown(f"**Stage:** {st.session_state.stage}  •  **Domain:** {st.session_state.demo_domain}")
    with cols[2]:
        if st.button("Reset Session"):
            for k in list(st.session_state.keys()):
                # careful reset: keep static secrets but clear session-scoped items
                if k not in ("demo_domain", "seeded_domain"):
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass
            _init_session_state()
            st.experimental_rerun()

def render_intro_page():
    st.header("Step 1 — Basic context")
    st.info("Choose a PRD domain. The selected domain's read-only demo PRDs will ground generation for this session.")

    # Domain selector (seeding happens when domain changes)
    domain = st.selectbox("PRD domain", DOMAINS, index=DOMAINS.index(st.session_state.demo_domain) if st.session_state.demo_domain in DOMAINS else 0, help="Demo PRDs are read-only examples the assistant will use for grounding.")
    st.session_state.demo_domain = domain

    # Seed demo PRDs into session vector store if domain changed or not seeded
    seed_session_vector_store_for_domain(domain)

    # show small summary of demo PRDs available
    demo_docs = _load_demo_prds()
    domain_docs = [d for d in demo_docs if d.get("domain") == domain]
    st.markdown(f"**Demo PRDs available for {domain}:** {len(domain_docs)} (read-only)")
    if len(domain_docs) > 0:
        if st.button("View sample demo PRDs"):
            for d in domain_docs[:7]:
                st.subheader(d.get("title"))
                if d.get("business_context"):
                    st.markdown(f"*{d.get('business_context')}*")
                ps = d.get("problem_statement") or d.get("prd_sections", {}).get("Problem_Statement", "")
                if ps:
                    st.markdown(f"**Problem:** {ps[:400]}")

    st.write("---")
    st.subheader("Experiment Context")
    with st.form("intro_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Business Goal", key="intro_business_goal", placeholder="Increase onboarding completion")
            st.text_input("Key Metric", key="intro_key_metric", placeholder="activation_rate")
            st.selectbox("Metric Type", ["Proportion", "Continuous"], key="intro_metric_type")
            st.number_input("Current Metric Value", min_value=0.0, value=30.0, key="intro_current_value")
        with col2:
            st.text_input("Product Area", key="intro_product_area", placeholder=domain)
            st.number_input("Target Metric Value", min_value=0.0, value=40.0, key="intro_target_value")
            st.number_input("DAU", min_value=1, value=10000, key="intro_dau")
            st.selectbox("Product Type", ["SaaS Product", "Mobile App", "Web Platform", "Other"], key="intro_product_type")
        st.text_area("Target User Persona (optional)", key="intro_user_persona")
        st.text_area("App Description (optional)", key="intro_app_description")

        st.write("Ephemeral user uploads")
        st.markdown("You may upload or paste PRDs for this session only. They will **not** be added to the demo corpus or persist across sessions.")
        uploaded = st.file_uploader("Upload PRD files (optional, .json/.txt). These are ephemeral and will not affect the demo corpus.", accept_multiple_files=True, type=["json","txt"])
        pasted = st.text_area("Or paste PRD text here (optional)", key="ephemeral_paste", height=120)
        consent = st.checkbox("I confirm I have rights to use these documents for temporary session-only grounding (ephemeral).", value=False, key="ephemeral_consent")
        include_in_prompt = st.checkbox("Include my ephemeral uploads in the prompt (optional)", value=False, key="include_ephemeral")
        st.session_state.include_ephemeral_in_prompt = include_in_prompt

        submitted = st.form_submit_button("Generate Hypotheses")
        if submitted:
            # process ephemeral uploads into session.ephemeral_uploads
            st.session_state.ephemeral_uploads = []
            if uploaded:
                for f in uploaded:
                    try:
                        content = f.read().decode("utf-8")
                    except Exception:
                        try:
                            content = f.read().decode("latin-1")
                        except Exception:
                            content = ""
                    st.session_state.ephemeral_uploads.append({"title": getattr(f, "name", "upload"), "text": content})
            if pasted and pasted.strip():
                st.session_state.ephemeral_uploads.append({"title": "pasted_doc", "text": pasted})
            if st.session_state.ephemeral_uploads and not consent:
                st.error("You must confirm ownership/permission to use uploaded docs for ephemeral session grounding.")
                return
            # store intro data
            intro = {
                "business_goal": st.session_state.intro_business_goal,
                "key_metric": st.session_state.intro_key_metric,
                "metric_type": st.session_state.intro_metric_type,
                "current_value": st.session_state.intro_current_value,
                "target_value": st.session_state.intro_target_value,
                "dau": st.session_state.intro_dau,
                "product_area": st.session_state.intro_product_area,
                "product_type": st.session_state.intro_product_type,
                "user_persona": st.session_state.intro_user_persona,
                "app_description": st.session_state.intro_app_description,
                "domain": st.session_state.demo_domain
            }
            st.session_state.prd_data["intro_data"] = intro

            # call generation for hypotheses
            with st.spinner("Generating hypotheses..."):
                resp = _call_generation(mode="hypotheses", user_inputs=intro, use_rag=st.session_state.use_rag, k=TOP_K)
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
                if resp.get("raw_text"):
                    st.code(resp.get("raw_text")[:4000])
            else:
                # normalize into st.session_state.hypotheses
                parsed = resp.get("parsed") if isinstance(resp, dict) else None
                if parsed:
                    st.session_state.hypotheses = parsed
                else:
                    raw = resp.get("raw_text") if isinstance(resp, dict) else resp
                    try:
                        parsed_raw = json.loads(raw)
                        st.session_state.hypotheses = parsed_raw
                    except Exception:
                        st.session_state.hypotheses = {"suggestion_1": {"Statement": raw}}
                next_stage()

def render_hypothesis_page():
    st.header("Step 2 — Hypotheses")
    st.info("Select a suggested hypothesis or write your own to enrich it.")

    st.text_area("Custom Hypothesis (optional)", key="custom_hypothesis_input", height=120)
    if st.button("Enrich Custom Hypothesis"):
        custom = st.session_state.get("custom_hypothesis_input","").strip()
        if not custom:
            st.error("Write a custom hypothesis first.")
        else:
            payload = {"custom_hypothesis": custom, **st.session_state.prd_data.get("intro_data", {})}
            with st.spinner("Enriching hypothesis..."):
                resp = _call_generation(mode="enrich_hypothesis", user_inputs=payload, use_rag=st.session_state.use_rag, k=TOP_K)
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
            else:
                parsed = resp.get("parsed") if isinstance(resp, dict) else None
                enriched = parsed or resp.get("raw_text") or resp
                st.session_state.prd_data["hypothesis"] = enriched
                st.success("Custom hypothesis enriched and saved to session.")
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
    # If prd_sections empty, generate
    if not st.session_state.prd_data.get("prd_sections"):
        with st.spinner("Drafting PRD sections..."):
            context = {**st.session_state.prd_data.get("intro_data", {}), **({"hypothesis": st.session_state.prd_data.get("hypothesis")} or {})}
            resp = _call_generation(mode="prd_sections", user_inputs=context, use_rag=st.session_state.use_rag, k=TOP_K)
            if isinstance(resp, dict) and resp.get("error"):
                st.error(resp.get("error"))
            else:
                parsed = resp.get("parsed") if isinstance(resp, dict) else None
                prd_sections = parsed or (resp.get("raw_text") if isinstance(resp, dict) else resp) or {}
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
        val = st.session_state.prd_data["prd_sections"].get(sec,"")
        new_val = st.text_area(f"Edit section: {sec}", value=format_content_for_display(val), height=240, key=f"edit_area_{sec}")
        if st.button("Save section", key=f"save_sec_{sec}"):
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
                sample = calculate_sample_size_continuous(current, intro.get("std_dev",1.0), mde, confidence, power)
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
    for k,v in prd.get("prd_sections", {}).items():
        st.markdown(f"### {k.replace('_',' ').title()}")
        st.markdown(format_content_for_display(v))

    st.subheader("Calculations")
    st.write(prd.get("calculations", {}))

    st.subheader("Risks")
    for i, r in enumerate(prd.get("risks", [])):
        st.markdown(f"- **Risk:** {r.get('risk')} — **Mitigation:** {r.get('mitigation')}")

    st.write("---")
    st.subheader("Context & Provenance")
    st.markdown("The assistant used the following demo snippets from the selected domain (read-only):")
    prov = st.session_state.get("provenance", [])
    if prov:
        for i, p in enumerate(prov[:TOP_K]):
            title = p.get("meta", {}).get("title") or p.get("meta", {}).get("source_id", f"snippet_{i}")
            text = p.get("text") or p.get("raw") or ""
            st.markdown(f"**{i+1}. {title}** — {text[:400]}...")
    else:
        st.markdown("_No demo snippets were used or none available._")

    if st.session_state.include_ephemeral_in_prompt and st.session_state.ephemeral_uploads:
        st.markdown("**Ephemeral user uploads included in prompt:**")
        for u in st.session_state.ephemeral_uploads:
            st.markdown(f"- {u.get('title')}: {u.get('text')[:300]}...")

    # Generate risks via LLM
    if st.button("Generate Risks"):
        payload = {**prd.get("intro_data", {}), "hypothesis": prd.get("hypothesis")}
        with st.spinner("Generating risks..."):
            resp = _call_generation(mode="risks", user_inputs=payload, use_rag=st.session_state.use_rag, k=TOP_K)
        if isinstance(resp, dict) and resp.get("error"):
            st.error(resp.get("error"))
        else:
            parsed = resp.get("parsed") if isinstance(resp, dict) else None
            risks = parsed or resp.get("raw_text") or resp
            if isinstance(risks, dict) and "risks" in risks:
                st.session_state.prd_data["risks"] = risks["risks"]
            elif isinstance(risks, list):
                st.session_state.prd_data["risks"] = risks
            else:
                st.info("Received risks; please review.")

    pdf_bytes = create_pdf(prd)
    st.download_button("Download PRD as PDF", pdf_bytes, file_name="PRD.pdf", mime="application/pdf")

    # Save: local persistence or no-op if missing
    st.write("---")
    if PERSISTENCE_AVAILABLE:
        if st.button("Save PRD to DB"):
            try:
                save_payload = {
                    "intro_data": prd.get("intro_data", {}),
                    "hypothesis": prd.get("hypothesis", {}),
                    "prd_sections": prd.get("prd_sections", {}),
                    "calculations": prd.get("calculations", {}),
                    "risks": prd.get("risks", []),
                    "saved_with_rag": st.session_state.use_rag,
                    "seeded_domain": st.session_state.seeded_domain
                }
                pid = persistence_save_prd(save_payload, title=save_payload["intro_data"].get("business_goal"), actor="streamlit_user")
                st.session_state.saved_prd_info = {"id": pid, "title": save_payload["intro_data"].get("business_goal","")}
                st.success(f"Saved PRD id: {pid}")
            except Exception as e:
                st.error(f"Save failed: {e}")
    else:
        st.info("Persistence not available. Add utils/persistence.py to enable saving.")

# --- Helper functions ---
def format_content_for_display(content):
    if isinstance(content, list):
        return "\n".join([f"- {item}" for item in content])
    else:
        return str(content)

# --- Main render logic ---
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
    render_review_page()

if st.session_state.scroll_to_top:
    scroll_to_top()
    st.session_state.scroll_to_top = False

# optional: show seeded domain info and ephemeral uploads count in footer
with st.expander("Session info (for demo)"):
    st.markdown(f"- Seeded demo domain: **{st.session_state.seeded_domain}**")
    st.markdown(f"- Ephemeral uploads in session: **{len(st.session_state.ephemeral_uploads)}**")
    st.markdown(f"- Include ephemeral uploads in prompt: **{st.session_state.include_ephemeral_in_prompt}**")
    st.markdown(f"- RAG enabled: **{st.session_state.use_rag}**")


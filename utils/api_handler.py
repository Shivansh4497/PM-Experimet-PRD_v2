# api_handler.py
"""
API / LLM handler for the PRD generator project.

This file is a drop-in replacement for the original api handler and:
- Preserves original non-RAG behavior by default (no changes to callers required).
- Adds an opt-in RAG + adaptive prompting + schema validation pathway using utils.rag_prompting.
- Exposes the same basic wrapper to call Groq (send_prompt_to_groq) and a higher-level generate_content()
  used by the Streamlit app.
- Provides safe JSON parsing and defensive behavior so the app keeps working even if LLM output is messy.

Usage notes (non-breaking defaults):
- If you call generate_content(..., use_rag=False) it behaves like the original handler (simple prompt -> Groq).
- To enable RAG + schema validation, call generate_content(..., use_rag=True, schema=..., mode=..., vector_store=...).
  The integration expects the `utils/rag_prompting` module to be present, but will gracefully fall back if it's missing.
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import requests

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Attempt to import optional rag_prompting helper (graceful fallback) ---
try:
    from utils import rag_prompting
    RAG_AVAILABLE = True
except Exception as e:
    rag_prompting = None
    RAG_AVAILABLE = False
    logger.info("utils.rag_prompting not available; RAG features disabled. (%s)", str(e))


# ----------------------------
# Low-level Groq / LLM wrapper
# ----------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# model used in original repo analysis; keep as default but allow override via env
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

def get_groq_api_key() -> Optional[str]:
    """Read Groq API key from environment or (for Streamlit) from st.secrets if available."""
    # Streamlit stores secrets in st.secrets; avoid importing streamlit here to keep module testable.
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    # try to be tolerant: import streamlit only when present
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def send_prompt_to_groq(prompt: str, max_tokens: int = 800, temperature: float = 0.0, model: Optional[str] = None, extra_payload: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Send a chat completion request to Groq (OpenAI-compatible endpoint).
    Returns a dict with at least:
      - 'raw': full response json (requests.Response.json())
      - 'text': best-effort extracted text content
      - 'ok': bool
      - 'error': optional error message

    This function intentionally keeps the payload shape compatible with the earlier project expectations.
    """
    api_key = get_groq_api_key()
    if not api_key:
        err = "GROQ_API_KEY not found in environment or Streamlit secrets."
        logger.error(err)
        return {"ok": False, "error": err, "raw": None, "text": None}

    model_to_use = model or DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": "You are an assistant that outputs JSON when asked. Be strict when asked for JSON."},
            {"role": "user", "content": prompt}
        ],
        # keep the response format field if the backend recognizes it; safe to include
        "max_tokens": max_tokens,
        "temperature": temperature,
        # allow the Groq-specific response format hint used in the original project,
        # but do not require it; keep as hint only.
        "response_format": {"type": "json_object"},
    }
    if extra_payload:
        payload.update(extra_payload)

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
        # Try to extract text from common shapes
        text = None
        # groq/openai-style: choices -> [ { message: {content: "..."} } ]
        try:
            choices = raw.get("choices", [])
            if choices:
                # choose first available message content
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta") or first
                    if isinstance(msg, dict):
                        text = msg.get("content") or msg.get("text")
                    else:
                        text = str(msg)
                else:
                    text = str(first)
        except Exception:
            text = None

        # fallback: look for 'content' or 'output'
        if not text:
            text = raw.get("content") or raw.get("output") or json.dumps(raw)

        return {"ok": True, "raw": raw, "text": text}
    except Exception as e:
        logger.exception("Groq request failed")
        return {"ok": False, "error": str(e), "raw": None, "text": None}


# ----------------------------
# JSON parsing helpers
# ----------------------------
def safe_json_parse(raw_text: Optional[str]) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Best-effort JSON parsing:
    - If raw_text already a dict/list => return as-is.
    - Try direct json.loads.
    - If it fails, attempt simple fixes (strip known wrappers, find first { ... } block).
    Returns (ok, parsed_obj_or_None, error_message_or_None)
    """
    if raw_text is None:
        return False, None, "No text to parse."

    # If already deserialized
    if not isinstance(raw_text, str):
        return True, raw_text, None

    txt = raw_text.strip()
    try:
        parsed = json.loads(txt)
        return True, parsed, None
    except Exception:
        # common heuristic: extract first {...} or [...] block
        import re
        m = re.search(r"(\{(?:.*\n*)*\})", txt, re.DOTALL)
        if not m:
            m = re.search(r"(\[.*\])", txt, re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                parsed = json.loads(candidate)
                return True, parsed, None
            except Exception as e:
                return False, None, f"heuristic parse failed: {e}"
        return False, None, "json.loads failed and no heuristic match"


# ----------------------------
# High-level content generator
# ----------------------------
def _default_llm_wrapper_for_rag(prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter to make send_prompt_to_groq compatible with rag_prompting.generate_with_rag's llm_call_fn interface.
    Expected signature: llm_call_fn(prompt: str, options: Dict) -> Dict (must include 'text')
    """
    max_tokens = options.get("max_tokens", 800)
    temperature = options.get("temperature", 0.0)
    model = options.get("model", DEFAULT_MODEL)
    extra = options.get("extra_payload")
    return send_prompt_to_groq(prompt, max_tokens=max_tokens, temperature=temperature, model=model, extra_payload=extra)


def generate_content(
    mode: str,
    user_inputs: Dict[str, Any],
    *,
    use_rag: bool = False,
    schema: Optional[Dict] = None,
    pydantic_model: Optional[type] = None,
    vector_store: Optional[Any] = None,
    k: int = 3,
    extra_instructions: Optional[str] = None,
    prompt_system_prefix: Optional[str] = None,
    # keep old-style parameters for backward compatibility with app.py callers
    max_tokens: int = 800,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level interface used by the Streamlit app to request model outputs.
    - mode: logical mode (e.g., 'hypotheses', 'prd_sections', 'risks', 'enrich_hypothesis', etc.)
    - user_inputs: dict containing business_goal, metric, current_value, target_value, description, etc.

    Returns a dict:
      {
        'ok': bool,
        'mode': mode,
        'raw_text': str or None,
        'parsed': dict or None,
        'errors': list[str],
        'metadata': {...}
      }

    Behavior:
    - If use_rag=True and rag_prompting is available, uses generate_with_rag to retrieve context, build adaptive prompt,
      call the LLM, and validate against schema/pydantic_model if provided.
    - Otherwise falls back to a direct prompt -> send_prompt_to_groq flow (keeps prior behavior).
    """
    start_time = time.time()
    metadata: Dict[str, Any] = {"use_rag": use_rag, "mode": mode}
    errors = []
    raw_text = None
    parsed = None

    # Compose extra_instructions tailored to the mode to preserve original behavior compatibility.
    mode_instructions_map = {
        "hypotheses": "Generate 2-3 short, testable hypotheses as JSON. Respond with a JSON array of objects: [{id, hypothesis, rationale}]",
        "enrich_hypothesis": "Enrich the provided hypothesis with experiment design and instrumentation notes in JSON.",
        "prd_sections": "Produce structured PRD sections as JSON: {title, description, sections: {Problem, Goals, Metrics, Instrumentation, Experiment Plan, Risks}}",
        "risks": "List top 5 risks and mitigation strategies in JSON array: [{risk, likelihood, impact, mitigation}]",
        "default": "Produce a helpful, structured output relevant to the inputs."
    }
    mode_hint = mode_instructions_map.get(mode, mode_instructions_map["default"])
    # combine mode hint with any extra_instructions passed in
    final_extra_instructions = (mode_hint + "\n" + extra_instructions) if extra_instructions else mode_hint

    # If RAG path enabled and available, use it
    if use_rag and RAG_AVAILABLE:
        try:
            logger.info("Calling RAG pipeline for mode=%s", mode)
            rag_result = rag_prompting.generate_with_rag(
                query_inputs=user_inputs,
                mode=mode if mode in ("growth", "retention", "monetization", "platform") else "default",
                llm_call_fn=_default_llm_wrapper_for_rag,
                vector_store=vector_store,
                k=k,
                extra_instructions=final_extra_instructions,
                schema=schema,
                pydantic_model=pydantic_model,
                prompt_system_prefix=prompt_system_prefix,
                timeout_seconds=60
            )
            # rag_result contains prompt, context, raw_llm_output, validated, latency
            raw_text = rag_result.get("raw_llm_output")
            validated = rag_result.get("validated", {})
            metadata.update({
                "rag_prompt": rag_result.get("prompt"),
                "rag_context": rag_result.get("context"),
                "rag_latency": rag_result.get("latency"),
                "llm_response_meta": rag_result.get("llm_response_meta"),
            })
            if validated.get("ok"):
                parsed = validated.get("data")
            else:
                errors.append("Validation failed: " + str(validated.get("errors")))
            return {
                "ok": len(errors) == 0,
                "mode": mode,
                "raw_text": raw_text,
                "parsed": parsed,
                "errors": errors,
                "metadata": metadata,
                "elapsed": time.time() - start_time
            }
        except Exception as e:
            # Fall through to non-RAG fallback but record error
            logger.exception("RAG pipeline failed; falling back to direct call.")
            errors.append(f"RAG pipeline error: {e}")

    # Non-RAG / fallback path: construct a simple prompt and call Groq directly (preserves prior behavior)
    try:
        # Build a plain prompt that includes the mode hint and structured user_inputs.
        prompt_parts = [
            f"Mode: {mode}",
            mode_hint,
            "User inputs:",
            json.dumps(user_inputs, indent=2, ensure_ascii=False),
            "",
            "Return the response in JSON where possible."
        ]
        prompt_str = "\n\n".join(prompt_parts)
        llm_resp = send_prompt_to_groq(prompt_str, max_tokens=max_tokens, temperature=temperature, model=model)
        if not llm_resp.get("ok"):
            errors.append(f"LLM call failed: {llm_resp.get('error')}")
            return {"ok": False, "mode": mode, "raw_text": None, "parsed": None, "errors": errors, "metadata": metadata, "elapsed": time.time() - start_time}

        raw_text = llm_resp.get("text")
        metadata["llm_raw"] = llm_resp.get("raw")

        # Try to parse with schema/pydantic if provided
        parsed_ok = False
        if pydantic_model and rag_prompting and getattr(rag_prompting, "PYDANTIC_AVAILABLE", False):
            ok, out = rag_prompting.validate_json_with_pydantic(raw_text, pydantic_model)
            if ok:
                parsed = out
                parsed_ok = True
            else:
                errors.append("Pydantic validation errors: " + "; ".join(out if isinstance(out, list) else [str(out)]))
        elif schema and rag_prompting and getattr(rag_prompting, "JSONSCHEMA_AVAILABLE", False):
            ok, out = rag_prompting.validate_json_with_jsonschema(raw_text, schema)
            if ok:
                parsed = out
                parsed_ok = True
            else:
                errors.append("JSON Schema validation errors: " + "; ".join(out if isinstance(out, list) else [str(out)]))

        if not parsed_ok:
            # Best-effort JSON parse
            ok, parsed_obj, parse_err = safe_json_parse(raw_text)
            if ok:
                parsed = parsed_obj
            else:
                # keep raw_text for UI inspection and report parse error
                errors.append(f"Could not parse LLM output as JSON: {parse_err}")

        return {
            "ok": len(errors) == 0,
            "mode": mode,
            "raw_text": raw_text,
            "parsed": parsed,
            "errors": errors,
            "metadata": metadata,
            "elapsed": time.time() - start_time
        }

    except Exception as e:
        logger.exception("generate_content failed")
        errors.append(str(e))
        return {"ok": False, "mode": mode, "raw_text": None, "parsed": None, "errors": errors, "metadata": metadata, "elapsed": time.time() - start_time}


# ----------------------------
# Convenience small helpers (kept for compatibility)
# ----------------------------
def generate_hypotheses(user_inputs: Dict[str, Any], use_rag: bool = False, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for hypothesis generation; preserves expected earlier API semantics."""
    return generate_content("hypotheses", user_inputs, use_rag=use_rag, **kwargs)

def generate_prd_sections(user_inputs: Dict[str, Any], use_rag: bool = False, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for PRD sections generation."""
    return generate_content("prd_sections", user_inputs, use_rag=use_rag, **kwargs)

def enrich_hypothesis(user_inputs: Dict[str, Any], use_rag: bool = False, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for enriching a hypothesis."""
    return generate_content("enrich_hypothesis", user_inputs, use_rag=use_rag, **kwargs)

def generate_risks(user_inputs: Dict[str, Any], use_rag: bool = False, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for risk generation."""
    return generate_content("risks", user_inputs, use_rag=use_rag, **kwargs)


# ----------------------------
# If run as script, quick sanity check (no side effects)
# ----------------------------
if __name__ == "__main__":
    print("api_handler.py sanity check")
    sample_inputs = {
        "business_goal": "Increase onboarding conversion",
        "metric": "signup_rate",
        "current_value": 0.08,
        "target_value": 0.12,
        "dau": 10000
    }
    # dry-run generate_content in non-RAG mode (does not call Groq if no key present)
    resp = generate_content("hypotheses", sample_inputs, use_rag=False)
    print(json.dumps(resp, indent=2, default=str))

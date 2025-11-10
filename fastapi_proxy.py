# fastapi_proxy.py
"""
Minimal FastAPI proxy / orchestration service for the PRD generator.

Purpose (single-file, drop-in):
- Proxy LLM/generation requests to the server side (keeps GROQ_API_KEY on the server).
- Reuse existing `api_handler.generate_content` when available to keep behavior identical.
- Provide endpoints:
    POST /generate     -> calls generate_content(...) (server-side)
    POST /save_prd     -> saves PRD via persistence.save_prd(...) (if available)
    POST /seed_index   -> runs utils.seed_vector_store.seed_vector_store(...) (optional)
    GET  /health       -> basic health check
- CORS enabled to allow your Streamlit front-end to call this service (configure origins as needed).
- Runs with Uvicorn if executed directly: `python fastapi_proxy.py`
- Config:
    - GROQ_API_KEY should be in environment or streamlit secrets (api_handler already expects it).
    - Optionally set ALLOWED_ORIGINS environment var (comma-separated) for CORS.
Notes:
- This is intentionally conservative: it will gracefully degrade if `api_handler`, `utils.persistence`, or `utils.seed_vector_store` are missing.
- No async background worker is included here — indexing runs synchronously if requested with index_now=true.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_proxy")

# Try to import existing project modules (graceful fallback)
try:
    import api_handler
    GENERATE_AVAILABLE = True
except Exception as e:
    api_handler = None
    GENERATE_AVAILABLE = False
    logger.warning("api_handler not importable; /generate will attempt fallback. (%s)", e)

try:
    from utils import persistence
    PERSISTENCE_AVAILABLE = True
except Exception as e:
    persistence = None
    PERSISTENCE_AVAILABLE = False
    logger.warning("utils.persistence not importable; /save_prd disabled. (%s)", e)

try:
    from utils import seed_vector_store
    SEED_AVAILABLE = True
except Exception as e:
    seed_vector_store = None
    SEED_AVAILABLE = False
    logger.warning("utils.seed_vector_store not importable; /seed_index disabled. (%s)", e)

# FastAPI app
app = FastAPI(title="PRD Generator Proxy", version="0.1")

# Configure CORS
allowed = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed.strip() == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pydantic request/response models -----
class GenerateRequest(BaseModel):
    mode: str
    user_inputs: Dict[str, Any]
    use_rag: Optional[bool] = False
    schema: Optional[Dict[str, Any]] = None
    pydantic_model: Optional[str] = None  # not used directly here
    k: Optional[int] = 3
    extra_instructions: Optional[str] = None
    prompt_system_prefix: Optional[str] = None

class GenerateResponse(BaseModel):
    ok: bool
    mode: str
    raw_text: Optional[str]
    parsed: Optional[Any]
    errors: Optional[Any]
    metadata: Optional[Dict[str, Any]] = None
    elapsed: Optional[float] = None

class SavePrdRequest(BaseModel):
    prd: Dict[str, Any]
    title: Optional[str] = None
    tags: Optional[list] = None
    actor: Optional[str] = "api_user"
    index_now: Optional[bool] = False

class SavePrdResponse(BaseModel):
    ok: bool
    prd_id: Optional[str] = None
    error: Optional[str] = None

class SeedIndexRequest(BaseModel):
    use_chroma: Optional[bool] = False
    embedder_model: Optional[str] = "all-MiniLM-L6-v2"
    limit: Optional[int] = None
    reindex: Optional[bool] = False
    dry_run: Optional[bool] = False

class SeedIndexResponse(BaseModel):
    ok: bool
    message: Optional[str] = None

# ----- Helpers -----
def _ensure_generate_available():
    if not GENERATE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Server-side generation not available (api_handler missing).")

def _ensure_persistence_available():
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Persistence not available (utils.persistence missing).")

def _ensure_seed_available():
    if not SEED_AVAILABLE:
        raise HTTPException(status_code=500, detail="Seed/index utility not available (utils.seed_vector_store missing).")

# ----- Endpoints -----
@app.get("/health")
def health():
    return {"ok": True, "service": "fastapi_proxy", "generate_available": GENERATE_AVAILABLE, "persistence_available": PERSISTENCE_AVAILABLE, "seed_available": SEED_AVAILABLE}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Server-side generation endpoint. Reuses api_handler.generate_content(...) where available.
    """
    logger.info("Generate request: mode=%s use_rag=%s", req.mode, req.use_rag)
    if GENERATE_AVAILABLE:
        try:
            # prefer passing structured args to api_handler.generate_content
            # api_handler.generate_content signature: (mode, user_inputs, use_rag=False, schema=None, pydantic_model=None, vector_store=None, k=3, ...)
            resp = api_handler.generate_content(
                mode=req.mode,
                user_inputs=req.user_inputs,
                use_rag=req.use_rag,
                schema=req.schema,
                pydantic_model=None,
                k=req.k,
                extra_instructions=req.extra_instructions,
                prompt_system_prefix=req.prompt_system_prefix
            )
            # If it already matches the expected shape, return; otherwise adapt a little
            if isinstance(resp, dict) and ("ok" in resp or "parsed" in resp):
                return GenerateResponse(
                    ok=resp.get("ok", True),
                    mode=req.mode,
                    raw_text=resp.get("raw_text") or resp.get("text") or resp.get("raw"),
                    parsed=resp.get("parsed"),
                    errors=resp.get("errors"),
                    metadata=resp.get("metadata"),
                    elapsed=resp.get("elapsed")
                )
            else:
                # fallback: wrap response as raw_text
                return GenerateResponse(ok=True, mode=req.mode, raw_text=str(resp), parsed=None, errors=None, metadata={})
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    else:
        # last-resort fallback: nothing available
        logger.error("Generation requested but api_handler not available.")
        raise HTTPException(status_code=500, detail="Server-side generation unavailable.")

@app.post("/save_prd", response_model=SavePrdResponse)
def save_prd(req: SavePrdRequest):
    """
    Save PRD to persistence. Optionally trigger immediate indexing (synchronous).
    """
    logger.info("Save PRD request: title=%s index_now=%s", req.title, req.index_now)
    _ensure_persistence_available()
    try:
        prd_id = persistence.save_prd(req.prd, title=req.title, tags=req.tags, actor=req.actor)
    except Exception as e:
        logger.exception("Failed to save PRD")
        return SavePrdResponse(ok=False, prd_id=None, error=str(e))

    # Optionally call seed/index sync (conservative)
    if req.index_now:
        if SEED_AVAILABLE:
            try:
                # call the seeder to index (limit= None, reindex False by default). This is synchronous.
                rc = seed_vector_store.seed_vector_store(use_chroma=False, embedder_model="all-MiniLM-L6-v2", limit=None, reindex=False, dry_run=False)
                logger.info("Indexing (index_now) completed with rc=%s", rc)
            except Exception as e:
                logger.exception("Indexing failed after save.")
                # don't fail the save; return warning in error field if indexing failed
                return SavePrdResponse(ok=True, prd_id=prd_id, error=f"Saved but indexing failed: {e}")

    return SavePrdResponse(ok=True, prd_id=prd_id, error=None)

@app.post("/seed_index", response_model=SeedIndexResponse)
def seed_index(req: SeedIndexRequest):
    """
    Trigger seeding/indexing of the vector store from saved PRDs.
    This can be called manually by a maintainer or CI job.
    """
    _ensure_seed_available()
    try:
        rc = seed_vector_store.seed_vector_store(use_chroma=req.use_chroma, embedder_model=req.embedder_model, limit=req.limit, reindex=req.reindex, dry_run=req.dry_run)
        if rc == 0:
            return SeedIndexResponse(ok=True, message="Indexing completed successfully.")
        else:
            return SeedIndexResponse(ok=False, message=f"Indexing returned code {rc}.")
    except Exception as e:
        logger.exception("Seed index failed")
        raise HTTPException(status_code=500, detail=f"Seed indexing failed: {e}")

# ----- Run with uvicorn when executed directly -----
if __name__ == "__main__":
    host = os.environ.get("FASTAPI_HOST", "127.0.0.1")
    port = int(os.environ.get("FASTAPI_PORT", "8000"))
    logger.info("Starting FastAPI proxy on %s:%s (CORS origins: %s)", host, port, origins)
    uvicorn.run("fastapi_proxy:app", host=host, port=port, reload=False)

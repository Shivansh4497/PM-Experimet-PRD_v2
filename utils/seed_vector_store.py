# utils/seed_vector_store.py
"""
Seed the RAG vector store from persisted PRDs.

Place this file at: utils/seed_vector_store.py

What it does (safe, idempotent-ish):
- Reads saved PRDs from utils.persistence (SQLite).
- Builds a vector store using utils.rag_prompting.build_default_vector_store().
  - If chromadb is available and use_chroma=True it will attempt to use Chroma; otherwise it uses an in-memory store.
- Chunks PRD text and indexes chunks as DocChunk entries with metadata.
- Writes a small manifest to data/vector_manifest.json with timestamp, counts and basic stats.
- Is robust to missing optional dependencies: if rag_prompting or persistence modules are missing, it prints helpful errors and exits.
- Default behavior: if a manifest already exists and reindex is False, it will skip re-indexing (conservative).
- Intended to be run manually (one-off) or as part of a CI / deployment step. It does NOT auto-run on import.

Usage:
    python -m utils.seed_vector_store         # runs with defaults (in-memory embedder fallback)
    python -m utils.seed_vector_store --use-chroma
    python -m utils.seed_vector_store --reindex
    python -m utils.seed_vector_store --limit 50

Notes:
- If you use Chroma and want persistence across runs, set use_chroma=True (and chromadb must be installed).
- Embeddings model: it will use sentence-transformers if installed; otherwise a trivial fallback embedder is used.
"""

from __future__ import annotations
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

# Local imports with graceful failure messages
try:
    from utils import rag_prompting
except Exception as e:
    rag_prompting = None
    rag_err = e

try:
    from utils import persistence
except Exception as e:
    persistence = None
    persistence_err = e

# Output manifest path
MANIFEST_PATH = Path(os.environ.get("VECTOR_MANIFEST", "data/vector_manifest.json"))
VECTOR_PERSIST_DIR = Path(os.environ.get("VECTOR_PERSIST_DIR", "data/vector_store"))

# ensure data dir exists
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
VECTOR_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

def _safe_exit(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)

def _load_saved_prds(limit: Optional[int] = None) -> List[dict]:
    if persistence is None:
        _safe_exit("utils.persistence is not importable. Please add utils/persistence.py to your repo.", 2)
    try:
        # list_prds returns a list of dicts with keys id,title,prd,created_at...
        rows = persistence.list_prds(limit=limit or 10000)
        return rows
    except Exception as e:
        _safe_exit(f"Failed to read saved PRDs from persistence: {e}", 3)

def _build_vector_store(use_chroma: bool = False, embedder_model: str = "all-MiniLM-L6-v2"):
    if rag_prompting is None:
        _safe_exit("utils.rag_prompting is not importable. Please add utils/rag_prompting.py to your repo.", 2)
    try:
        # build_default_vector_store will pick chroma if requested and available, otherwise fallback
        vs = rag_prompting.build_default_vector_store(persist_dir=str(VECTOR_PERSIST_DIR), use_chroma=use_chroma, embedder_model=embedder_model)
        return vs
    except Exception as e:
        _safe_exit(f"Failed to create vector store instance: {e}", 4)

def _prepare_docchunks_from_prd_rows(rows: List[dict]):
    """
    Accepts rows from persistence.list_prds() and returns a flat list of DocChunk objects.
    """
    if rag_prompting is None:
        _safe_exit("utils.rag_prompting not available to create DocChunk objects.", 2)

    doc_chunks = []
    total_chunks = 0
    for row in rows:
        prd_obj = row.get("prd") if isinstance(row, dict) else None
        if not prd_obj:
            # fallback: try reading prd_json-like fields
            prd_obj = row
        source_id = row.get("id") or row.get("prd", {}).get("id") or str(int(time.time()*1000))
        title = row.get("title") or prd_obj.get("intro_data", {}).get("business_goal") if isinstance(prd_obj, dict) else "PRD"
        # create a textual blob to chunk: prefer full_text if available, else derive
        text_blob = row.get("full_text") or rag_prompting._derive_full_text_from_prd(prd_obj) if hasattr(rag_prompting, "_derive_full_text_from_prd") else json.dumps(prd_obj, ensure_ascii=False)
        # chunk into smaller pieces
        chunks = rag_prompting.docs_to_docchunks(source_id=source_id, title=title, text=text_blob)
        doc_chunks.extend(chunks)
        total_chunks += len(chunks)
    return doc_chunks, total_chunks

def seed_vector_store(use_chroma: bool = False, embedder_model: str = "all-MiniLM-L6-v2", limit: Optional[int] = None, reindex: bool = False, dry_run: bool = False):
    """
    Main entry:
    - use_chroma: try to use chroma if available (requires chromadb installed)
    - embedder_model: name for sentence-transformers if available
    - limit: max number of PRDs to index (useful for testing)
    - reindex: force reindexing even if manifest exists
    - dry_run: do everything but do not actually call vector_store.add_documents (for preview)
    """
    print("Seed Vector Store — start")
    if rag_prompting is None:
        print("ERROR: utils.rag_prompting not available:", rag_err)
        return 2
    if persistence is None:
        print("ERROR: utils.persistence not available:", persistence_err)
        return 2

    # If manifest exists and not reindexing, skip to avoid accidental duplicates
    if MANIFEST_PATH.exists() and not reindex:
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            print(f"Manifest found at {MANIFEST_PATH} (indexed_at: {manifest.get('indexed_at')}, total_docs: {manifest.get('total_prds')}). Use --reindex to force re-index.")
            return 0
        except Exception:
            # if manifest corrupted, continue with reindex behavior
            print("Existing manifest unreadable; proceeding with indexing.")

    rows = _load_saved_prds(limit=limit)
    total_prds = len(rows)
    print(f"Loaded {total_prds} PRDs from persistence (limit={limit}).")

    if total_prds == 0:
        print("No PRDs to index. Exiting.")
        return 0

    vector_store = _build_vector_store(use_chroma=use_chroma, embedder_model=embedder_model)
    print(f"Vector store instance created. use_chroma={use_chroma}")

    doc_chunks, total_chunks = _prepare_docchunks_from_prd_rows(rows)
    print(f"Prepared {total_chunks} chunks from {total_prds} PRDs.")

    # if dry_run, just show a preview
    if dry_run:
        print("Dry run mode enabled. Preview of first 5 chunks:")
        for i, c in enumerate(doc_chunks[:5]):
            print(f" - id={c.id}, meta={c.meta}, preview={c.text[:200]!r}")
        print("Dry run complete. No documents were indexed.")
        return 0

    # Attempt to add documents (handle exceptions gracefully)
    try:
        # For chroma persistent stores, adding same ids twice may error; we intentionally allow re-run if reindex True.
        if reindex and hasattr(vector_store, "collection") and getattr(vector_store, "collection", None) is not None:
            # best-effort: if chroma adapter exposed, try to delete existing docs with same ids
            try:
                # attempt to remove by ids (chromadb collections expose delete(ids=[...]))
                ids = [c.id for c in doc_chunks]
                if hasattr(vector_store.collection, "delete"):
                    # delete existing ids (may be optional)
                    vector_store.collection.delete(ids=ids)
                    print("Existing docs removed from chroma collection prior to reindex.")
            except Exception:
                # ignore; not critical
                pass

        # Add documents in batches to avoid memory blowup
        BATCH_SIZE = 256
        added = 0
        for i in range(0, len(doc_chunks), BATCH_SIZE):
            batch = doc_chunks[i:i+BATCH_SIZE]
            try:
                vector_store.add_documents(batch)
                added += len(batch)
                print(f"Indexed batch {i//BATCH_SIZE + 1}: {len(batch)} chunks (total indexed: {added}).")
            except Exception as e:
                print(f"Failed to index batch starting at {i}: {e}")
        print(f"Indexing complete. Total chunks attempted: {len(doc_chunks)}. Successfully added (approx): {added}.")
    except Exception as e:
        _safe_exit(f"Indexing failed: {e}", 5)

    # write manifest
    manifest = {
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_prds": total_prds,
        "total_chunks": total_chunks,
        "use_chroma": use_chroma,
        "embedder_model": embedder_model,
        "vector_persist_dir": str(VECTOR_PERSIST_DIR),
    }
    try:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Manifest written to {MANIFEST_PATH}.")
    except Exception as e:
        print(f"Failed to write manifest: {e}")

    print("Seed Vector Store — finished successfully.")
    return 0

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Seed RAG Vector Store from persisted PRDs")
    p.add_argument("--use-chroma", action="store_true", help="Use chroma vector store if chromadb is installed (default: False)")
    p.add_argument("--embedder-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformers model name for embeddings (if available)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of PRDs to index (for testing)")
    p.add_argument("--reindex", action="store_true", help="Force reindexing even if manifest exists")
    p.add_argument("--dry-run", action="store_true", help="Prepare chunks and show a preview but do not index")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args()
    rc = seed_vector_store(use_chroma=args.use_chroma, embedder_model=args.embedder_model, limit=args.limit, reindex=args.reindex, dry_run=args.dry_run)
    sys.exit(rc)

# utils/persistence.py
"""
Lightweight persistence layer for PRD storage (SQLite).

Drop this file at: utils/persistence.py

Purpose:
- Provide a simple, zero-dependency (stdlib-only) SQLite-backed store to persist generated PRDs,
  user edits, and an audit trail. Ideal for Streamlit Community Cloud or local dev.
- Additive only: nothing runs at import time except a tiny, safe "ensure DB dir" step.
- Exposes a clear programmatic API so you (or the app) can enable persistence with a single import call.

Design choices:
- Uses sqlite3 (part of Python stdlib) so no new pip deps required.
- Stores PRD content as JSON text in a column (allows flexible schema).
- Provides simple search using LIKE on title and full_text columns for easy discovery.
- Keeps an audit log for edits with user/actor, timestamp, and diff/notes field.
- Uses UUID v4 strings as primary IDs to avoid collisions.
- DB file path is configurable via env var PERSISTENCE_DB (or defaults to ./data/prd_store.db).
"""

from __future__ import annotations
import os
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -- Configuration --
DEFAULT_DB_PATH = os.environ.get("PERSISTENCE_DB", "data/prd_store.db")
DB_PATH = Path(DEFAULT_DB_PATH)

# Ensure parent directory exists (safe at import time)
try:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    # best-effort; if this fails (readonly FS), operations will raise clearer errors later
    pass

# -- Utility helpers --
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def make_uuid() -> str:
    return str(uuid.uuid4())

# -- DB schema creation (idempotent) --
_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prds (
    id TEXT PRIMARY KEY,
    title TEXT,
    tags TEXT,           -- comma-separated tags (optional)
    full_text TEXT,      -- denormalized text for search (optional)
    prd_json TEXT NOT NULL, -- the full PRD JSON / dict serialized as text
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prds_title ON prds(title);
CREATE INDEX IF NOT EXISTS idx_prds_created_at ON prds(created_at);

CREATE TABLE IF NOT EXISTS prd_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prd_id TEXT NOT NULL,
    actor TEXT DEFAULT 'system',
    action TEXT NOT NULL, -- e.g., 'create', 'update', 'delete'
    note TEXT,
    snapshot_json TEXT, -- optional snapshot of prd_json at time of action
    created_at TEXT NOT NULL,
    FOREIGN KEY (prd_id) REFERENCES prds(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audit_prd_id ON prd_audit(prd_id);
"""

# -- Connection helper (context manager) --
def _get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    # Return rows as dict-like
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(_SCHEMA_SQL)
    conn.commit()

# -- Public API --
def init_db(path: Optional[str] = None) -> None:
    """
    Ensure the DB file and schema exist.
    - path: optional override of DB path
    """
    p = Path(path) if path else DB_PATH
    try:
        conn = _get_conn(p)
        _ensure_schema(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

def save_prd(prd: Dict[str, Any], title: Optional[str] = None, tags: Optional[List[str]] = None, actor: str = "system", db_path: Optional[str] = None) -> str:
    """
    Insert a new PRD into the DB.
    - prd: dict-like PRD object (will be JSON serialized)
    - title: optional human title (if not provided, tries prd.get('title') or uses generated id)
    - tags: optional list of tags
    - actor: who performed the create action (for audit)
    Returns the prd id (UUID string).
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    prd_id = make_uuid()
    created_at = now_iso()
    title_val = title or prd.get("title") if isinstance(prd, dict) else None
    if not title_val:
        title_val = f"PRD {prd_id[:8]}"
    tags_str = ",".join(tags) if tags else ""
    prd_json = json.dumps(prd, ensure_ascii=False)
    full_text = _derive_full_text_from_prd(prd)

    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prds (id, title, tags, full_text, prd_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (prd_id, title_val, tags_str, full_text, prd_json, created_at, created_at)
        )
        conn.commit()
        # audit
        _insert_audit(conn, prd_id=prd_id, actor=actor, action="create", note="created PRD", snapshot=prd_json)
        return prd_id
    finally:
        conn.close()

def get_prd(prd_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch a PRD by id. Returns the deserialized PRD dict with metadata fields, or None if missing.
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM prds WHERE id = ?", (prd_id,))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_prd_dict(row)
    finally:
        conn.close()

def list_prds(limit: int = 50, offset: int = 0, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List PRDs with basic pagination. Returns a list of PRD dicts (with metadata).
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM prds ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = cur.fetchall()
        return [_row_to_prd_dict(r) for r in rows]
    finally:
        conn.close()

def update_prd(prd_id: str, prd_updates: Dict[str, Any], actor: str = "system", note: Optional[str] = None, db_path: Optional[str] = None) -> bool:
    """
    Update an existing PRD by merging fields from prd_updates into existing PRD dict.
    - prd_updates: partial dict to merge into stored PRD
    Returns True if updated, False if PRD not found.
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        cur.execute("SELECT prd_json FROM prds WHERE id = ?", (prd_id,))
        row = cur.fetchone()
        if not row:
            return False
        current = json.loads(row["prd_json"])
        # shallow merge: update top-level keys
        if isinstance(current, dict) and isinstance(prd_updates, dict):
            current.update(prd_updates)
        else:
            # if not dicts, replace entirely
            current = prd_updates
        new_json = json.dumps(current, ensure_ascii=False)
        updated_at = now_iso()
        full_text = _derive_full_text_from_prd(current)
        cur.execute("UPDATE prds SET prd_json = ?, updated_at = ?, full_text = ? WHERE id = ?", (new_json, updated_at, full_text, prd_id))
        conn.commit()
        _insert_audit(conn, prd_id=prd_id, actor=actor, action="update", note=note or "updated PRD", snapshot=new_json)
        return True
    finally:
        conn.close()

def delete_prd(prd_id: str, actor: str = "system", note: Optional[str] = None, db_path: Optional[str] = None) -> bool:
    """
    Delete a PRD by id. Returns True if deleted, False if not found.
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        # optional: snapshot before delete
        cur.execute("SELECT prd_json FROM prds WHERE id = ?", (prd_id,))
        row = cur.fetchone()
        if not row:
            return False
        snapshot = row["prd_json"]
        cur.execute("DELETE FROM prds WHERE id = ?", (prd_id,))
        conn.commit()
        _insert_audit(conn, prd_id=prd_id, actor=actor, action="delete", note=note or "deleted PRD", snapshot=snapshot)
        return True
    finally:
        conn.close()

def search_prds(q: str, limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Very simple search over title and full_text using SQL LIKE.
    - q: search query (will be used with %q%)
    Note: basic and not token-aware. Good for quick discovery; for advanced search use vector DB + RAG.
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    like_q = f"%{q}%"
    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM prds WHERE title LIKE ? OR full_text LIKE ? ORDER BY created_at DESC LIMIT ?",
            (like_q, like_q, limit)
        )
        rows = cur.fetchall()
        return [_row_to_prd_dict(r) for r in rows]
    finally:
        conn.close()

def get_audit_for_prd(prd_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return audit log rows for a given PRD id (newest first).
    """
    p = Path(db_path) if db_path else DB_PATH
    init_db(str(p))
    conn = _get_conn(p)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM prd_audit WHERE prd_id = ? ORDER BY created_at DESC", (prd_id,))
        rows = cur.fetchall()
        return [_row_to_audit_dict(r) for r in rows]
    finally:
        conn.close()

# -- Internal helpers --
def _insert_audit(conn: sqlite3.Connection, prd_id: str, actor: str, action: str, note: Optional[str], snapshot: Optional[str]):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO prd_audit (prd_id, actor, action, note, snapshot_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (prd_id, actor, action, note or "", snapshot or None, now_iso())
    )
    conn.commit()

def _derive_full_text_from_prd(prd: Any) -> str:
    """
    Create a denormalized searchable text blob from the PRD JSON/dict.
    Attempts to extract useful fields if present (title, hypothesis, sections, description).
    """
    try:
        if isinstance(prd, str):
            return prd[:20000]  # cap size
        if isinstance(prd, dict):
            parts = []
            # common fields
            for k in ("title", "hypothesis", "description", "summary", "business_goal"):
                if k in prd and prd[k]:
                    parts.append(str(prd[k]))
            # sections
            sections = prd.get("sections") or prd.get("prd_sections") or prd.get("prd")
            if isinstance(sections, dict):
                for k, v in sections.items():
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, dict):
                        parts.append(" ".join([str(x) for x in v.values() if isinstance(x, str)]))
            # entire JSON fallback
            parts.append(json.dumps(prd, ensure_ascii=False))
            text = "\n\n".join(parts)
            return text[:20000]
        # fallback: stringify
        return str(prd)[:20000]
    except Exception:
        try:
            return json.dumps(prd, ensure_ascii=False)[:20000]
        except Exception:
            return ""

def _row_to_prd_dict(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        prd_json = json.loads(row["prd_json"])
    except Exception:
        prd_json = row["prd_json"]
    return {
        "id": row["id"],
        "title": row["title"],
        "tags": row["tags"].split(",") if row["tags"] else [],
        "full_text": row["full_text"],
        "prd": prd_json,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }

def _row_to_audit_dict(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        snapshot = json.loads(row["snapshot_json"]) if row["snapshot_json"] else None
    except Exception:
        snapshot = row["snapshot_json"]
    return {
        "id": row["id"],
        "prd_id": row["prd_id"],
        "actor": row["actor"],
        "action": row["action"],
        "note": row["note"],
        "snapshot": snapshot,
        "created_at": row["created_at"]
    }

# -- Sanity / example usage when run directly --
if __name__ == "__main__":
    # Quick demo: create DB, insert, list, update, search, delete
    print("Persistence sanity check: DB path =", DB_PATH)
    init_db()
    sample_prd = {
        "title": "Increase onboarding conversion",
        "business_goal": "Improve onboarding completion",
        "hypothesis": "If we reduce steps from 5 to 3, onboarding completion will increase by 20%.",
        "sections": {
            "Problem": "High drop-off on step 3.",
            "Metric": "onboarding_completion_rate",
            "Experiment": "A/B test reducing steps"
        }
    }
    pid = save_prd(sample_prd, tags=["onboarding", "growth"], actor="demo")
    print("Saved PRD id:", pid)
    all_prds = list_prds()
    print("Total PRDs:", len(all_prds))
    fetched = get_prd(pid)
    print("Fetched title:", fetched["title"])
    update_prd(pid, {"hypothesis": "Updated hypothesis text."}, actor="demo", note="tweaked hypothesis")
    results = search_prds("onboarding")
    print("Search results:", [r["id"] for r in results])
    audits = get_audit_for_prd(pid)
    print("Audit trail entries:", len(audits))
    # cleanup demo
    # delete_prd(pid, actor="demo", note="cleanup demo")
    print("Done.")

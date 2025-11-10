# utils/rag_prompting.py
"""
RAG + Adaptive Prompting + Schema Control utilities for the PRD generator project.

Place this file at: utils/rag_prompting.py

What this provides (purely additive, safe to add):
- a lightweight, pluggable retrieval layer (in-memory / optional Chroma/Faiss adapters)
- adaptive prompt builder (modes: growth/retention/monetization/platform or custom)
- JSON schema validation helpers using pydantic (and jsonschema fallback)
- a small orchestration helper `generate_with_rag` that composes retrieval + prompt + LLM call
  but does NOT call any LLM itself — you pass your existing LLM call function (keeps integration non-invasive)

Notes:
- The module tries to import chromadb/transformers/faiss if available and falls back to an in-memory vector store.
- No side effects at import time. Nothing will run until you call the functions.
- This file intentionally accepts a pluggable `llm_call_fn` so your current api_handler/groq wrapper can be used unchanged.
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Optional dependencies: we will gracefully degrade if not installed.
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

try:
    # lightweight embeddings via sentence-transformers if available
    from sentence_transformers import SentenceTransformer
    SBER_AVAILABLE = True
except Exception:
    SBER_AVAILABLE = False

try:
    # pydantic for schema validation
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False

# fallback JSON schema validator
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except Exception:
    JSONSCHEMA_AVAILABLE = False

# ---------- Data structures ----------
@dataclass
class DocChunk:
    id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class RetrievalResult:
    chunks: List[DocChunk]
    scores: List[float]  # higher is better

# ---------- Simple in-memory vector store (fallback) ----------
class InMemoryVectorStore:
    def __init__(self, embedder: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self._docs: List[DocChunk] = []
        self._embedder = embedder

    def add_documents(self, docs: List[DocChunk]):
        texts = [d.text for d in docs]
        if self._embedder:
            embs = self._embedder(texts)
            for d, e in zip(docs, embs):
                d.embedding = e
        self._docs.extend(docs)

    def similarity_search(self, query: str, k: int = 4) -> RetrievalResult:
        if not self._docs:
            return RetrievalResult(chunks=[], scores=[])
        # if embedder available, compute embed and do cosine similarity
        if self._embedder and self._docs[0].embedding is not None:
            q_emb = self._embedder([query])[0]
            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))
            def norm(a):
                return sum(x*x for x in a) ** 0.5
            scores = []
            for d in self._docs:
                score = 0.0
                try:
                    score = dot(q_emb, d.embedding) / (norm(q_emb) * norm(d.embedding))
                except Exception:
                    score = 0.0
                scores.append(score)
            ranked = sorted(zip(self._docs, scores), key=lambda x: x[1], reverse=True)[:k]
            docs, scores = zip(*ranked) if ranked else ([], [])
            return RetrievalResult(chunks=list(docs), scores=list(scores))
        # fallback: simple substring match scoring
        def score_fn(text: str, q: str):
            t = text.lower()
            ql = q.lower()
            score = 0
            if ql in t:
                score += 10
            score += text.lower().count(ql)
            # minor bonus for overlapping keywords
            q_tokens = set(ql.split())
            text_tokens = set(t.split())
            score += len(q_tokens & text_tokens) * 0.5
            return float(score)
        scored = [(d, score_fn(d.text, query)) for d in self._docs]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
        docs, scores = zip(*ranked) if ranked and ranked[0][1] > 0 else ([], [])
        return RetrievalResult(chunks=list(docs), scores=list(scores))


# ---------- Optional Chroma adapter ----------
class ChromaVectorStoreAdapter:
    def __init__(self, persist_directory: Optional[str] = None, embedder: Optional[Callable[[List[str]], List[List[float]]]] = None):
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("chromadb is not available. Install chromadb to use ChromaVectorStoreAdapter.")
        # create chroma client; avoid side-effects if not wanted
        settings = ChromaSettings()
        self.client = chromadb.Client(settings=settings)
        self.persist_directory = persist_directory
        self.embedder = embedder
        # use a collection name stable to the project
        self.collection = self.client.get_or_create_collection(name="prds", metadata={"source": "ai_prd_tool"})

    def add_documents(self, docs: List[DocChunk]):
        ids = [d.id for d in docs]
        metadatas = [d.meta for d in docs]
        texts = [d.text for d in docs]
        # if chroma embedder is available, we can pass embeddings; otherwise chroma will compute its own
        if self.embedder:
            embs = self.embedder(texts)
            self.collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embs)
        else:
            self.collection.add(ids=ids, metadatas=metadatas, documents=texts)

    def similarity_search(self, query: str, k: int = 4) -> RetrievalResult:
        # chroma returns results with distances (smaller better). We'll invert to a score.
        results = self.collection.query(query_texts=[query], n_results=k)
        # results: {'ids': [[...]], 'distances': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
        rows = results.get("ids", [[]])[0]
        docs = []
        scores = []
        for i, doc_id in enumerate(rows):
            try:
                doc_text = results.get("documents", [[]])[0][i]
                meta = results.get("metadatas", [[]])[0][i] or {}
                distance = results.get("distances", [[]])[0][i] or 0.0
            except Exception:
                continue
            score = 1.0 / (1.0 + float(distance))
            docs.append(DocChunk(id=str(doc_id), text=str(doc_text), meta=meta))
            scores.append(score)
        return RetrievalResult(chunks=docs, scores=scores)


# ---------- Embedding adapter ----------
def default_embedder_factory(model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns a function that accepts List[str] and returns List[List[float]] embeddings.
    Falls back to a trivial char-level embedding if transformer unavailable.
    """
    if SBER_AVAILABLE:
        model = SentenceTransformer(model_name)
        def embed(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, convert_to_numpy=False).tolist()
        return embed
    else:
        # simple fallback: sparse char-level / token counts vector (not ideal, but deterministic)
        def embed(texts: List[str]) -> List[List[float]]:
            out = []
            for t in texts:
                vec = [float(len(t)), float(sum(1 for c in t if c.isupper())), float(sum(1 for c in t if c.isdigit()))]
                out.append(vec)
            return out
        return embed


# ---------- Adaptive prompt builder ----------
DEFAULT_MODE_INSTRUCTIONS = {
    "growth": "You are a Growth PM assistant. Focus on acquisition, activation, and conversion. Prefer short, testable hypotheses and tractable instrumentation.",
    "retention": "You are a Retention PM assistant. Focus on retention loops, habit formation, and long-term user engagement. Consider cohort effects.",
    "monetization": "You are a Monetization PM assistant. Focus on revenue, ARPU, pricing, trade-offs, and implications for user experience.",
    "platform": "You are a Platform PM assistant. Focus on internal metrics, developer UX, API ergonomics, and platform-level adoption.",
    # fallback default
    "default": "You are a Product Manager assistant. Prioritize clarity, testability, and measurable outcomes."
}

def build_adaptive_prompt(
    mode: str,
    user_inputs: Dict[str, Any],
    context_snippets: Optional[List[Tuple[str, float]]] = None,
    extra_instructions: Optional[str] = None,
    system_prefix: Optional[str] = None
) -> str:
    """
    Compose a robust system + user prompt using:
    - mode: one of growth/retention/monetization/platform/default (influences style & constraints)
    - user_inputs: structured dict available to the LLM (e.g. business_goal, metric, current_value, target)
    - context_snippets: list of (text, score) retrieved from RAG to ground the model
    - extra_instructions: ad-hoc instructions appended
    - system_prefix: overrides default system-level instructions if provided

    Returns a single string prompt ready to send to the LLM.
    """
    mode = (mode or "default").lower()
    mode_inst = DEFAULT_MODE_INSTRUCTIONS.get(mode, DEFAULT_MODE_INSTRUCTIONS["default"])
    system_prefix = system_prefix or "You generate JSON responses only when asked. Be concise and follow any schema instructions strictly."
    parts = []
    parts.append(f"{system_prefix}\n\nMode instructions: {mode_inst}\n")
    parts.append("User-provided inputs:")
    # include structured representation of user inputs
    try:
        parts.append(json.dumps(user_inputs, indent=2, ensure_ascii=False))
    except Exception:
        parts.append(str(user_inputs))

    if context_snippets:
        parts.append("\nContext (most relevant first):")
        for i, (text, score) in enumerate(context_snippets):
            snippet = text.strip().replace("\n", " ")
            parts.append(f"[{i+1}] (score={score:.3f}) {snippet}")

    if extra_instructions:
        parts.append("\nAdditional instructions:")
        parts.append(extra_instructions)

    # small safety: ask for JSON only if requested in extra_instructions, otherwise free text
    prompt = "\n\n".join(parts)
    return prompt


# ---------- Schema validation helpers ----------
class JSONSchemaValidationError(Exception):
    pass

def validate_json_with_pydantic(raw: Union[str, Dict], model: Optional[type] = None) -> Tuple[bool, Union[Dict, List[str]]]:
    """
    If pydantic is available and a model class is provided, try to validate the raw data.
    - raw: JSON string or dict
    - model: a pydantic.BaseModel subclass
    Returns (ok:bool, data_or_errors)
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception as e:
            return False, [f"invalid json: {str(e)}"]
    else:
        parsed = raw

    if not PYDANTIC_AVAILABLE or model is None:
        # if pydantic not available, fallback to returning parsed dict
        return True, parsed

    try:
        validated = model.parse_obj(parsed)
        return True, validated.dict()
    except ValidationError as e:
        # return a list of human-friendly errors
        errs = []
        for err in e.errors():
            loc = ".".join([str(x) for x in err.get("loc", [])])
            msg = err.get("msg", "")
            errs.append(f"{loc}: {msg}")
        return False, errs

def validate_json_with_jsonschema(raw: Union[str, Dict], schema: Dict) -> Tuple[bool, Union[Dict, List[str]]]:
    """
    Validate a JSON object (or string) against a jsonschema dict.
    Requires jsonschema package. If not available, returns (False, ['jsonschema not installed'])
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception as e:
            return False, [f"invalid json: {str(e)}"]
    else:
        parsed = raw

    if not JSONSCHEMA_AVAILABLE:
        return False, ["jsonschema not installed"]

    try:
        jsonschema.validate(instance=parsed, schema=schema)
        return True, parsed
    except jsonschema.ValidationError as e:
        return False, [str(e)]

# ---------- Orchestration helper ----------
def generate_with_rag(
    query_inputs: Dict[str, Any],
    mode: str,
    llm_call_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    vector_store: Optional[Union[InMemoryVectorStore, ChromaVectorStoreAdapter]] = None,
    k: int = 3,
    extra_instructions: Optional[str] = None,
    schema: Optional[Dict] = None,
    pydantic_model: Optional[type] = None,
    prompt_system_prefix: Optional[str] = None,
    timeout_seconds: int = 30
) -> Dict[str, Any]:
    """
    High-level helper: retrieve relevant context, build adaptive prompt, call LLM via llm_call_fn,
    and validate the response against a schema/pydantic model if provided.

    - query_inputs: user inputs (business_goal, metric, etc.)
    - mode: 'growth' | 'retention' | 'monetization' | 'platform' | 'default'
    - llm_call_fn: function(prompt_str, options_dict) -> dict with keys 'text' or 'content' depending on your wrapper.
        Example wrapper signature:
            def my_llm_wrapper(prompt: str, options: Dict) -> Dict:
                return {"text": "<llm raw output>", "usage": {...}}
    - vector_store: optional vector store instance. If not provided, retrieval step will be skipped.
    - k: number of context snippets to retrieve
    - schema: optional JSON Schema dict to validate LLM output
    - pydantic_model: optional pydantic model class to validate LLM output
    - returns dict:
        {
          "prompt": "<final prompt string>",
          "context": [...],
          "raw_llm_output": "<raw>",
          "validated": {"ok": True/False, "data": {...} or "errors": [...]},
          "latency": float
        }
    """
    start = time.time()
    context_snippets: List[Tuple[str, float]] = []
    if vector_store:
        try:
            retrieval = vector_store.similarity_search(
                query=query_inputs.get("business_goal", "") or json.dumps(query_inputs),
                k=k
            )
            # pack text and score for prompt insertion
            context_snippets = [(c.text, s) for c, s in zip(retrieval.chunks, retrieval.scores)]
        except Exception as e:
            # retrieval failure should not break everything; keep empty context
            context_snippets = []

    # Compose prompt
    prompt = build_adaptive_prompt(
        mode=mode,
        user_inputs=query_inputs,
        context_snippets=context_snippets,
        extra_instructions=extra_instructions,
        system_prefix=prompt_system_prefix
    )

    # Call LLM wrapper (pluggable)
    llm_opts = {"max_tokens": 800, "temperature": 0.0}
    try:
        llm_response = llm_call_fn(prompt, llm_opts)
    except Exception as e:
        return {
            "prompt": prompt,
            "context": context_snippets,
            "raw_llm_output": None,
            "validated": {"ok": False, "data": [], "error": f"llm_call_fn raised: {str(e)}"},
            "latency": time.time() - start
        }

    # extract text from wrapper response (compatibility tolerant)
    raw_text = None
    if isinstance(llm_response, dict):
        raw_text = llm_response.get("text") or llm_response.get("content") or llm_response.get("output") or llm_response.get("choices", [{}])[0].get("message", {}).get("content")
    else:
        raw_text = str(llm_response)

    validated_result = {"ok": True, "data": None, "errors": None}
    # Try pydantic validation first (if model available), then jsonschema
    if pydantic_model and PYDANTIC_AVAILABLE:
        ok, out = validate_json_with_pydantic(raw_text, pydantic_model)
        if ok:
            validated_result["ok"] = True
            validated_result["data"] = out
        else:
            validated_result["ok"] = False
            validated_result["errors"] = out
    elif schema and JSONSCHEMA_AVAILABLE:
        ok, out = validate_json_with_jsonschema(raw_text, schema)
        if ok:
            validated_result["ok"] = True
            validated_result["data"] = out
        else:
            validated_result["ok"] = False
            validated_result["errors"] = out
    else:
        # best-effort parse JSON if possible
        try:
            parsed = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
            validated_result["ok"] = True
            validated_result["data"] = parsed
        except Exception:
            validated_result["ok"] = False
            validated_result["errors"] = ["Could not parse LLM output as JSON; consider providing a schema or pydantic model for strict validation."]

    return {
        "prompt": prompt,
        "context": context_snippets,
        "raw_llm_output": raw_text,
        "validated": validated_result,
        "latency": time.time() - start,
        "llm_response_meta": llm_response if isinstance(llm_response, dict) else None
    }

# ---------- Helper to build an in-memory vector_store quickly ----------
def build_default_vector_store(persist_dir: Optional[str] = None, use_chroma: bool = False, embedder_model: str = "all-MiniLM-L6-v2"):
    """
    Utility to quickly create a vector store instance:
    - If use_chroma=True and chromadb installed, returns ChromaVectorStoreAdapter
    - Otherwise returns InMemoryVectorStore with a default embedder (sentence-transformers if installed)
    """
    embedder = default_embedder_factory(embedder_model)
    if use_chroma and CHROMADB_AVAILABLE:
        try:
            return ChromaVectorStoreAdapter(persist_directory=persist_dir, embedder=embedder)
        except Exception:
            # fallback to in-memory
            return InMemoryVectorStore(embedder=embedder)
    else:
        return InMemoryVectorStore(embedder=embedder)

# ---------- Simple utilities for chunking docs ----------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Naive text chunker that splits long documents into overlapping chunks.
    """
    text = text.strip()
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def docs_to_docchunks(source_id: str, title: str, text: str) -> List[DocChunk]:
    """
    Convert a long text into a list of DocChunk instances with generated ids.
    """
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    out = []
    for idx, c in enumerate(chunks):
        did = f"{source_id}__{idx}"
        out.append(DocChunk(id=did, text=c, meta={"title": title, "source_id": source_id, "chunk_index": idx}))
    return out

# ---------- Example pydantic model template (for PRD output) ----------
# This is an *example* Pydantic model showing how you can validate the LLM JSON output.
if PYDANTIC_AVAILABLE:
    class PRDHypothesisModel(BaseModel):
        hypothesis: str
        primary_metric: str
        current_value: Union[float, int, None]
        target_value: Union[float, int, None]
        rationale: Optional[str] = None
        experiment_design: Optional[Dict[str, Any]] = None

    class PRDSectionsModel(BaseModel):
        title: str
        description: Optional[str]
        sections: Dict[str, Any]  # flexible; each key -> section content
else:
    PRDHypothesisModel = None
    PRDSectionsModel = None

# ---------- End of module ----------

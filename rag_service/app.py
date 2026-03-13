import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


PROMPT_RE = re.compile(r"\[[^\]]+\]> $")
RESULT_RE = re.compile(
    r"^\s*- \[score=(?P<score>[-+]?\d*\.?\d+), sid=(?P<sid>-?\d+), label=(?P<label>\d+), tokens=\[(?P<t0>-?\d+),(?P<t1>-?\d+)\]\] (?P<text>.*)$"
)

logger = logging.getLogger("rag_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class IngestRequest(BaseModel):
    index: str = Field(default="default", min_length=1, max_length=128)
    doc_id: str = Field(min_length=1, max_length=256)
    chunks: List[str] = Field(default_factory=list, min_length=1, max_length=1000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    index: str = "default"
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["knn", "radius", "relative", "adaptive"] = "knn"
    k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.7, ge=0.0, le=10.0)
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    minN: int = Field(default=3, ge=1, le=100)
    lookahead: int = Field(default=10, ge=1, le=200)
    gapDelta: float = Field(default=0.1, ge=0.0, le=5.0)
    max_results: int = Field(default=25, ge=1, le=200)
    dedup_by_chunk_id: bool = True


class AnswerRequest(BaseModel):
    index: str = "default"
    query: str
    mode: Literal["knn", "radius", "relative", "adaptive"] = "knn"
    k: int = 6
    temperature: float = 0.2
    max_tokens: int = 600
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: Optional[str] = None
    system_prompt: str = (
        "You are a retrieval-grounded assistant. Use only the provided context. "
        "If context is insufficient, say you do not know. Include chunk citations by chunk_id."
    )


@dataclass
class RetrievedChunk:
    score: float
    sentence_id: int
    label: int
    token_start: int
    token_end: int
    text: str
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init()

    def _connect(self):
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init(self):
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    index_name TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    sentence_id INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(index_name, sentence_id)
                )
                """
            )
            columns = {
                row[1]
                for row in con.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "created_at" not in columns:
                con.execute("ALTER TABLE chunks ADD COLUMN created_at TEXT")
                con.execute("UPDATE chunks SET created_at = datetime('now') WHERE created_at IS NULL")
            if "updated_at" not in columns:
                con.execute("ALTER TABLE chunks ADD COLUMN updated_at TEXT")
                con.execute("UPDATE chunks SET updated_at = datetime('now') WHERE updated_at IS NULL")

            con.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_unique_chunk
                ON chunks(index_name, doc_id, chunk_id)
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_doc
                ON chunks(index_name, doc_id)
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS namespace_configs (
                    index_name TEXT PRIMARY KEY,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER,
                    metric TEXT NOT NULL,
                    hnsw_m INTEGER,
                    hnsw_ef_construction INTEGER,
                    hnsw_ef_search INTEGER,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    index_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    retrieved_count INTEGER NOT NULL,
                    citations_count INTEGER NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    response_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_audit_request
                ON llm_audit(request_id)
                """
            )

    def has_sentence_id(self, index_name: str, sentence_id: int) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM chunks WHERE index_name = ? AND sentence_id = ?",
                (index_name, sentence_id),
            ).fetchone()
            return row is not None

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        json.dumps(metadata)
        return metadata

    def put_chunk(self, index_name: str, doc_id: str, chunk_id: str, sentence_id: int, chunk_text: str, metadata: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        metadata = self._normalize_metadata(metadata)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO chunks(index_name, doc_id, chunk_id, sentence_id, chunk_text, metadata_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_name, sentence_id)
                DO UPDATE SET
                    doc_id=excluded.doc_id,
                    chunk_id=excluded.chunk_id,
                    chunk_text=excluded.chunk_text,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    index_name,
                    doc_id,
                    chunk_id,
                    sentence_id,
                    chunk_text,
                    json.dumps(metadata, sort_keys=True),
                    now,
                    now,
                ),
            )

    def get_by_sentence_id(self, index_name: str, sentence_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT doc_id, chunk_id, chunk_text, metadata_json
                FROM chunks
                WHERE index_name = ? AND sentence_id = ?
                """,
                (index_name, sentence_id),
            ).fetchone()
            if not row:
                return None
            return {
                "doc_id": row[0],
                "chunk_id": row[1],
                "chunk_text": row[2],
                "metadata": json.loads(row[3] or "{}"),
            }

    def upsert_namespace_config(
        self,
        index_name: str,
        embedding_model: str,
        embedding_dimension: Optional[int],
        metric: str,
        hnsw_m: Optional[int],
        hnsw_ef_construction: Optional[int],
        hnsw_ef_search: Optional[int],
        metadata: Dict[str, Any],
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO namespace_configs(
                    index_name,
                    embedding_model,
                    embedding_dimension,
                    metric,
                    hnsw_m,
                    hnsw_ef_construction,
                    hnsw_ef_search,
                    metadata_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_name)
                DO UPDATE SET
                    embedding_model=excluded.embedding_model,
                    embedding_dimension=excluded.embedding_dimension,
                    metric=excluded.metric,
                    hnsw_m=excluded.hnsw_m,
                    hnsw_ef_construction=excluded.hnsw_ef_construction,
                    hnsw_ef_search=excluded.hnsw_ef_search,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    index_name,
                    embedding_model,
                    embedding_dimension,
                    metric,
                    hnsw_m,
                    hnsw_ef_construction,
                    hnsw_ef_search,
                    json.dumps(metadata, sort_keys=True),
                    now,
                ),
            )

    def get_namespace_config(self, index_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT
                    embedding_model,
                    embedding_dimension,
                    metric,
                    hnsw_m,
                    hnsw_ef_construction,
                    hnsw_ef_search,
                    metadata_json,
                    updated_at
                FROM namespace_configs
                WHERE index_name = ?
                """,
                (index_name,),
            ).fetchone()
        if not row:
            return None
        return {
            "embedding_model": row[0],
            "embedding_dimension": row[1],
            "metric": row[2],
            "hnsw_m": row[3],
            "hnsw_ef_construction": row[4],
            "hnsw_ef_search": row[5],
            "metadata": json.loads(row[6] or "{}"),
            "updated_at": row[7],
        }

    def record_llm_audit(
        self,
        request_id: str,
        index_name: str,
        provider: str,
        model: str,
        retrieved_count: int,
        citations_count: int,
        prompt_hash: str,
        response_hash: str,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO llm_audit(
                    request_id,
                    index_name,
                    provider,
                    model,
                    retrieved_count,
                    citations_count,
                    prompt_hash,
                    response_hash,
                    created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    index_name,
                    provider,
                    model,
                    retrieved_count,
                    citations_count,
                    prompt_hash,
                    response_hash,
                    now,
                ),
            )


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def validate_identifier(name: str, label: str) -> str:
    if not IDENTIFIER_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{label} must match {IDENTIFIER_RE.pattern} "
                f"(allowed: letters, numbers, dot, underscore, colon, dash)"
            ),
        )
    return name


def derive_stable_sentence_id(index_name: str, doc_id: str, chunk_position: int, chunk_text: str) -> int:
    payload = f"{index_name}\n{doc_id}\n{chunk_position}\n{chunk_text}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**31 - 1)


def build_ingest_rows(index_name: str, doc_id: str, chunks: List[str], metadata_store: MetadataStore) -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    for chunk_position, raw_chunk in enumerate(chunks, start=1):
        chunk_text = raw_chunk.strip()
        if not chunk_text:
            continue

        chunk_id = f"{doc_id}#chunk-{chunk_position}"
        sentence_id = derive_stable_sentence_id(index_name, doc_id, chunk_position, chunk_text)
        probe = 0
        while metadata_store.has_sentence_id(index_name, sentence_id):
            probe += 1
            sentence_id = derive_stable_sentence_id(index_name, doc_id, chunk_position + probe, chunk_text)

        rows.append((chunk_id, sentence_id, chunk_text))
    return rows


class EmbeddingsCliClient:
    def __init__(self, binary_path: str, model_path: str):
        self.binary_path = binary_path
        self.model_path = model_path
        self.process: Optional[subprocess.Popen[str]] = None
        self.lock = threading.Lock()

    def _ensure_started(self):
        if self.process is not None:
            return
        if not os.path.exists(self.binary_path):
            raise RuntimeError(f"Embeddings binary not found: {self.binary_path}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Embeddings model not found: {self.model_path}")

        self.process = subprocess.Popen(
            [self.binary_path, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._read_until_prompt()

    def _read_until_prompt(self) -> str:
        assert self.process is not None and self.process.stdout is not None
        out = []
        line_buf = ""
        while True:
            ch = self.process.stdout.read(1)
            if ch == "":
                break
            out.append(ch)
            line_buf += ch
            if PROMPT_RE.search(line_buf):
                break
            if ch == "\n":
                line_buf = ""
        return "".join(out)

    def command(self, command_text: str) -> str:
        with self.lock:
            self._ensure_started()
            assert self.process is not None and self.process.stdin is not None
            self.process.stdin.write(command_text + "\n")
            self.process.stdin.flush()
            return self._read_until_prompt()

    def use_index(self, index_name: str):
        self.command(f"use {index_name}")

    def append_with_sentence_id(self, index_name: str, sentence_id: int, text: str):
        self.use_index(index_name)
        safe = text.replace("\n", " ")
        self.command(f"appendid {sentence_id} {safe}")

    def search(self, req: SearchRequest) -> List[RetrievedChunk]:
        self.use_index(req.index)

        if req.mode == "knn":
            raw = self.command(f"knn {req.k} {req.query}")
        elif req.mode == "radius":
            raw = self.command(f"radius {req.min_score} {req.query}")
        elif req.mode == "relative":
            raw = self.command(f"relative {req.alpha} {req.query}")
        elif req.mode == "adaptive":
            raw = self.command(f"adaptive {req.alpha} {req.minN} {req.lookahead} {req.gapDelta} {req.query}")
        else:
            raise RuntimeError(f"Unsupported mode: {req.mode}")

        return self._parse_results(raw)

    @staticmethod
    def _parse_results(raw: str) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        for line in raw.splitlines():
            match = RESULT_RE.match(line.strip())
            if not match:
                continue
            chunks.append(
                RetrievedChunk(
                    score=float(match.group("score")),
                    sentence_id=int(match.group("sid")),
                    label=int(match.group("label")),
                    token_start=int(match.group("t0")),
                    token_end=int(match.group("t1")),
                    text=match.group("text"),
                )
            )
        return chunks


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: Optional[str], model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            body = r.json()
            return body["choices"][0]["message"]["content"]


class AnthropicCompatibleLLM:
    def __init__(self, base_url: str, api_key: Optional[str], model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic provider")

        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.base_url}/v1/messages", json=payload, headers=headers)
            r.raise_for_status()
            body = r.json()
            content = body.get("content", [])
            text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
            return "\n".join(part for part in text_parts if part).strip()




@dataclass
class ProviderHealthState:
    consecutive_failures: int = 0
    circuit_open_until_epoch: float = 0.0
    last_error: Optional[str] = None
    last_attempt_at: Optional[str] = None
    last_success_at: Optional[str] = None

class NamespaceConfigRequest(BaseModel):
    embedding_model: str = Field(min_length=1, max_length=256)
    embedding_dimension: Optional[int] = Field(default=None, ge=1)
    metric: Literal["l2", "ip", "cosine"] = "cosine"
    hnsw_m: Optional[int] = Field(default=None, ge=1)
    hnsw_ef_construction: Optional[int] = Field(default=None, ge=1)
    hnsw_ef_search: Optional[int] = Field(default=None, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimpleRateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self._lock = threading.Lock()
        self._counters: Dict[str, Tuple[int, int]] = {}

    def allow(self, key: str, now_epoch_seconds: int) -> bool:
        window_start = now_epoch_seconds - (now_epoch_seconds % 60)
        with self._lock:
            stored = self._counters.get(key)
            if stored is None or stored[0] != window_start:
                self._counters[key] = (window_start, 1)
                return True

            count = stored[1]
            if count >= self.max_requests_per_minute:
                return False

            self._counters[key] = (window_start, count + 1)
            return True


def normalize_search_results(
    chunks: List[RetrievedChunk], max_results: int, dedup_by_chunk_id: bool
) -> List[RetrievedChunk]:
    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (
            -chunk.score,
            chunk.doc_id or "",
            chunk.chunk_id or "",
            chunk.sentence_id,
        ),
    )

    if not dedup_by_chunk_id:
        return sorted_chunks[:max_results]

    deduped_chunks: List[RetrievedChunk] = []
    seen_chunk_keys = set()
    for chunk in sorted_chunks:
        chunk_key = chunk.chunk_id or f"sid:{chunk.sentence_id}"
        if chunk_key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(chunk_key)
        deduped_chunks.append(chunk)
        if len(deduped_chunks) >= max_results:
            break
    return deduped_chunks


def execute_search_with_timeout(search_request: SearchRequest, timeout_seconds: float = 20.0) -> List[RetrievedChunk]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(emb_client.search, search_request)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as timeout_exception:
            future.cancel()
            raise HTTPException(
                status_code=504,
                detail=f"Search request timed out after {timeout_seconds:.1f}s",
            ) from timeout_exception
        except Exception as search_exception:
            raise HTTPException(
                status_code=502,
                detail=f"Search backend failed: {search_exception}",
            ) from search_exception


def validate_mode_specific_constraints(search_request: SearchRequest):
    if search_request.mode == "knn" and search_request.k > search_request.max_results:
        raise HTTPException(
            status_code=422,
            detail="For knn mode, k must be <= max_results",
        )
    if search_request.mode == "radius" and search_request.min_score <= 0.0:
        raise HTTPException(
            status_code=422,
            detail="For radius mode, min_score must be > 0",
        )
    if search_request.mode == "adaptive" and search_request.minN > search_request.lookahead:
        raise HTTPException(
            status_code=422,
            detail="For adaptive mode, minN must be <= lookahead",
        )


def build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        cid = chunk.chunk_id or f"sid:{chunk.sentence_id}"
        did = chunk.doc_id or "unknown_doc"
        blocks.append(
            f"[CTX {idx}] chunk_id={cid} doc_id={did} score={chunk.score:.4f}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_citation_aware_user_prompt(query_text: str, context_text: str) -> str:
    return (
        "You must answer ONLY from the context below.\n"
        "Rules:\n"
        "1) Cite supporting evidence inline using exact chunk IDs in the form [chunk_id=<value>].\n"
        "2) If context is insufficient, answer exactly: I do not know based on the provided context.\n"
        "3) Do not fabricate citations, sources, or facts not present in context.\n\n"
        f"Question:\n{query_text}\n\n"
        f"Context:\n{context_text}\n"
    )


def hash_text_for_audit(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_structured_event(event_name: str, request_id: str, **payload: Any):
    event_payload = {
        "event": event_name,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    event_payload.update(payload)
    logger.info(json.dumps(event_payload, sort_keys=True))


def enrich_with_metadata(index_name: str, chunks: List[RetrievedChunk], store: MetadataStore) -> List[RetrievedChunk]:
    for c in chunks:
        row = store.get_by_sentence_id(index_name, c.sentence_id)
        if row:
            c.doc_id = row["doc_id"]
            c.chunk_id = row["chunk_id"]
            c.metadata = row["metadata"]
            if row["chunk_text"]:
                c.text = row["chunk_text"]
    return chunks


app = FastAPI(title="re-Isearch RAG Service", version="0.1.0")

EMBEDDINGS_BIN = os.getenv("EMBEDDINGS_BIN", "./embeddings/build/sbert_search")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "")
METADATA_DB = os.getenv("RAG_METADATA_DB", "rag_service/rag_metadata.db")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
RAG_SERVICE_API_KEY = os.getenv("RAG_SERVICE_API_KEY")
RAG_RATE_LIMIT_PER_MIN = int(os.getenv("RAG_RATE_LIMIT_PER_MIN", "120"))
RAG_LLM_FAILOVER_CHAIN = os.getenv("RAG_LLM_FAILOVER_CHAIN", "openai,anthropic")
RAG_LLM_MAX_RETRIES = int(os.getenv("RAG_LLM_MAX_RETRIES", "2"))
RAG_LLM_BACKOFF_BASE_MS = int(os.getenv("RAG_LLM_BACKOFF_BASE_MS", "250"))
RAG_LLM_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("RAG_LLM_CIRCUIT_BREAKER_THRESHOLD", "3"))
RAG_LLM_CIRCUIT_BREAKER_COOLDOWN_S = int(os.getenv("RAG_LLM_CIRCUIT_BREAKER_COOLDOWN_S", "60"))

metadata_store = MetadataStore(METADATA_DB)
emb_client = EmbeddingsCliClient(EMBEDDINGS_BIN, EMBEDDINGS_MODEL)
rate_limiter = SimpleRateLimiter(max_requests_per_minute=RAG_RATE_LIMIT_PER_MIN)
provider_health_states: Dict[str, ProviderHealthState] = {
    "openai": ProviderHealthState(),
    "anthropic": ProviderHealthState(),
}


def get_llm_client(provider: str, model_override: Optional[str]):
    if provider == "openai":
        model_name = model_override or LLM_MODEL
        return OpenAICompatibleLLM(LLM_BASE_URL, LLM_API_KEY, model_name), model_name
    if provider == "anthropic":
        model_name = model_override or ANTHROPIC_MODEL
        return AnthropicCompatibleLLM(ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, model_name), model_name
    raise HTTPException(status_code=422, detail=f"Unsupported llm_provider: {provider}")


def parse_failover_chain(primary_provider: str) -> List[str]:
    configured = [p.strip() for p in RAG_LLM_FAILOVER_CHAIN.split(",") if p.strip()]
    chain: List[str] = []
    if primary_provider:
        chain.append(primary_provider)
    for provider in configured:
        if provider not in chain:
            chain.append(provider)
    return chain


def provider_is_circuit_open(provider: str, now_epoch_seconds: float) -> bool:
    state = provider_health_states.get(provider)
    if not state:
        return False
    return state.circuit_open_until_epoch > now_epoch_seconds


def update_provider_health(provider: str, success: bool, error_text: Optional[str] = None):
    state = provider_health_states.setdefault(provider, ProviderHealthState())
    now_dt = datetime.now(timezone.utc).isoformat()
    state.last_attempt_at = now_dt

    if success:
        state.consecutive_failures = 0
        state.last_error = None
        state.last_success_at = now_dt
        state.circuit_open_until_epoch = 0.0
        return

    state.consecutive_failures += 1
    state.last_error = error_text
    if state.consecutive_failures >= RAG_LLM_CIRCUIT_BREAKER_THRESHOLD:
        state.circuit_open_until_epoch = (
            datetime.now(timezone.utc).timestamp() + RAG_LLM_CIRCUIT_BREAKER_COOLDOWN_S
        )


def call_provider_with_retries(
    provider: str,
    model_override: Optional[str],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str, int]:
    last_error = None
    provider_client, active_model = get_llm_client(provider, model_override)

    for attempt_index in range(RAG_LLM_MAX_RETRIES + 1):
        try:
            answer_text = provider_client.generate(system_prompt, user_prompt, temperature, max_tokens)
            update_provider_health(provider, success=True)
            return answer_text, active_model, attempt_index + 1
        except Exception as ex:
            last_error = str(ex)
            update_provider_health(provider, success=False, error_text=last_error)
            if attempt_index < RAG_LLM_MAX_RETRIES:
                sleep_seconds = ((2 ** attempt_index) * RAG_LLM_BACKOFF_BASE_MS) / 1000.0
                time.sleep(sleep_seconds)

    raise RuntimeError(last_error or f"Provider {provider} failed after retries")


def answer_with_failover(req: AnswerRequest, user_prompt: str) -> Tuple[str, str, str, int, List[Dict[str, Any]]]:
    providers_to_try = parse_failover_chain(req.llm_provider)
    now_epoch = datetime.now(timezone.utc).timestamp()
    attempt_log: List[Dict[str, Any]] = []

    for provider in providers_to_try:
        if provider_is_circuit_open(provider, now_epoch):
            state = provider_health_states.get(provider)
            attempt_log.append(
                {
                    "provider": provider,
                    "status": "skipped_circuit_open",
                    "circuit_open_until_epoch": state.circuit_open_until_epoch if state else None,
                }
            )
            continue

        try:
            answer_text, active_model, attempts = call_provider_with_retries(
                provider,
                req.llm_model,
                req.system_prompt,
                user_prompt,
                req.temperature,
                req.max_tokens,
            )
            attempt_log.append(
                {
                    "provider": provider,
                    "status": "success",
                    "attempts": attempts,
                }
            )
            return answer_text, provider, active_model, attempts, attempt_log
        except Exception as ex:
            attempt_log.append(
                {
                    "provider": provider,
                    "status": "failed",
                    "error": str(ex),
                }
            )

    raise RuntimeError(f"All configured providers failed. attempts={attempt_log}")


@app.middleware("http")
async def enforce_auth_and_rate_limit(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    if request.url.path == "/health":
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    if RAG_SERVICE_API_KEY:
        supplied = request.headers.get("x-api-key", "")
        if supplied != RAG_SERVICE_API_KEY:
            response = JSONResponse(status_code=401, content={"detail": "Invalid or missing x-api-key"})
            response.headers["x-request-id"] = request_id
            return response

    client_key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    if not rate_limiter.allow(client_key, now_epoch):
        response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        response.headers["x-request-id"] = request_id
        return response

    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
def health():
    return {
        "ok": True,
        "embeddings_bin": EMBEDDINGS_BIN,
        "embeddings_model_configured": bool(EMBEDDINGS_MODEL),
        "auth_enabled": bool(RAG_SERVICE_API_KEY),
        "rate_limit_per_min": RAG_RATE_LIMIT_PER_MIN,
        "llm_failover_chain": parse_failover_chain(""),
        "llm_max_retries": RAG_LLM_MAX_RETRIES,
        "llm_circuit_breaker_threshold": RAG_LLM_CIRCUIT_BREAKER_THRESHOLD,
        "llm_circuit_breaker_cooldown_s": RAG_LLM_CIRCUIT_BREAKER_COOLDOWN_S,
    }


@app.get("/health/providers")
def health_providers():
    now_epoch = datetime.now(timezone.utc).timestamp()
    statuses: Dict[str, Any] = {}
    for provider_name, state in provider_health_states.items():
        statuses[provider_name] = {
            "consecutive_failures": state.consecutive_failures,
            "circuit_open": state.circuit_open_until_epoch > now_epoch,
            "circuit_open_until_epoch": state.circuit_open_until_epoch,
            "last_error": state.last_error,
            "last_attempt_at": state.last_attempt_at,
            "last_success_at": state.last_success_at,
        }
    return {"ok": True, "providers": statuses}


@app.post("/namespaces/{index_name}/config")
def upsert_namespace_config(index_name: str, req: NamespaceConfigRequest):
    safe_index = validate_identifier(index_name, "index")
    metadata_store.upsert_namespace_config(
        safe_index,
        req.embedding_model,
        req.embedding_dimension,
        req.metric,
        req.hnsw_m,
        req.hnsw_ef_construction,
        req.hnsw_ef_search,
        req.metadata,
    )
    return {"ok": True, "index": safe_index}


@app.get("/namespaces/{index_name}/config")
def get_namespace_config(index_name: str):
    safe_index = validate_identifier(index_name, "index")
    row = metadata_store.get_namespace_config(safe_index)
    if not row:
        raise HTTPException(status_code=404, detail=f"No namespace config found for index {safe_index}")
    return {"ok": True, "index": safe_index, "config": row}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not EMBEDDINGS_MODEL:
        raise HTTPException(status_code=500, detail="EMBEDDINGS_MODEL is required")

    index_name = validate_identifier(req.index, "index")
    doc_id = validate_identifier(req.doc_id, "doc_id")

    ingest_rows = build_ingest_rows(index_name, doc_id, req.chunks, metadata_store)
    if not ingest_rows:
        raise HTTPException(status_code=422, detail="chunks must contain at least one non-empty text chunk")

    inserted = 0
    failures: List[Dict[str, Any]] = []

    for chunk_id, sentence_id, chunk_text in ingest_rows:
        try:
            emb_client.append_with_sentence_id(index_name, sentence_id, chunk_text)
            metadata_store.put_chunk(index_name, doc_id, chunk_id, sentence_id, chunk_text, req.metadata)
            inserted += 1
        except Exception as ex:
            failures.append(
                {
                    "chunk_id": chunk_id,
                    "sentence_id": sentence_id,
                    "error": str(ex),
                }
            )

    if inserted == 0:
        raise HTTPException(status_code=502, detail={"message": "ingest failed for all chunks", "failures": failures})

    return {
        "ok": True,
        "inserted": inserted,
        "failed": len(failures),
        "index": index_name,
        "doc_id": doc_id,
        "failures": failures,
    }


@app.post("/search")
def search(req: SearchRequest):
    if not EMBEDDINGS_MODEL:
        raise HTTPException(status_code=500, detail="EMBEDDINGS_MODEL is required")

    index_name = validate_identifier(req.index, "index")
    validate_mode_specific_constraints(req)

    req.index = index_name
    start_time = time.perf_counter()
    chunks = execute_search_with_timeout(req, timeout_seconds=20.0)
    chunks = enrich_with_metadata(index_name, chunks, metadata_store)
    chunks = normalize_search_results(chunks, max_results=req.max_results, dedup_by_chunk_id=req.dedup_by_chunk_id)
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "ok": True,
        "index": index_name,
        "mode": req.mode,
        "latency_ms": latency_ms,
        "query_chars": len(req.query),
        "count": len(chunks),
        "result_count": len(chunks),
        "results": [
            {
                "score": c.score,
                "sentence_id": c.sentence_id,
                "label": c.label,
                "token_start": c.token_start,
                "token_end": c.token_end,
                "text": c.text,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "metadata": c.metadata or {},
            }
            for c in chunks
        ],
    }


@app.post("/answer")
def answer(req: AnswerRequest, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    index_name = validate_identifier(req.index, "index")
    search_req = SearchRequest(index=index_name, query=req.query, mode=req.mode, k=req.k)
    chunks = emb_client.search(search_req)
    chunks = enrich_with_metadata(index_name, chunks, metadata_store)

    if not chunks:
        return {
            "ok": True,
            "answer": "I do not know based on the available context.",
            "citations": [],
            "retrieved": 0,
            "request_id": request_id,
        }

    context = build_context(chunks)
    user_prompt = build_citation_aware_user_prompt(req.query, context)

    log_structured_event(
        "answer_request_started",
        request_id,
        index=index_name,
        requested_provider=req.llm_provider,
        retrieved_candidates=len(chunks),
    )

    try:
        answer_text, selected_provider, active_model, provider_attempts, failover_attempt_log = answer_with_failover(req, user_prompt)
    except Exception as ex:
        log_structured_event(
            "answer_request_failed",
            request_id,
            index=index_name,
            error=str(ex),
            failover_attempt_log=failover_attempt_log if "failover_attempt_log" in locals() else [],
        )
        raise HTTPException(status_code=502, detail=f"LLM call failed: {ex}")

    prompt_hash = hash_text_for_audit(user_prompt)
    response_hash = hash_text_for_audit(answer_text)
    metadata_store.record_llm_audit(
        request_id=request_id,
        index_name=index_name,
        provider=selected_provider,
        model=active_model,
        retrieved_count=len(chunks),
        citations_count=len([c.chunk_id or f"sid:{c.sentence_id}" for c in chunks]),
        prompt_hash=prompt_hash,
        response_hash=response_hash,
    )

    log_structured_event(
        "answer_request_completed",
        request_id,
        index=index_name,
        provider=selected_provider,
        model=active_model,
        provider_attempts=provider_attempts,
        retrieved=len(chunks),
    )

    return {
        "ok": True,
        "answer": answer_text,
        "llm_provider": selected_provider,
        "llm_model": active_model,
        "provider_attempts": provider_attempts,
        "failover_attempt_log": failover_attempt_log,
        "citations": [c.chunk_id or f"sid:{c.sentence_id}" for c in chunks],
        "retrieved": len(chunks),
        "request_id": request_id,
    }

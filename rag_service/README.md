# RAG Service (Retriever + LLM Orchestrator)

This module adds a first-class HTTP service layer for RAG on top of `embeddings/src/main.cpp` interactive CLI.

## Features

- `GET /health/providers`: provider health/circuit-breaker status for LLM adapters.
- `POST /ingest`: ingest chunks into a named embeddings index and local metadata store.
- `POST /search`: retrieve semantic hits from embeddings (`knn`, `radius`, `relative`, `adaptive`).
- `POST /answer`: retrieve context + call OpenAI-compatible LLM API.
- `POST /namespaces/{index}/config` and `GET /namespaces/{index}/config`: persist/retrieve namespace retrieval metadata (model, dimension, metric, HNSW knobs).
- SQLite metadata mapping (`rag_service/rag_metadata.db`) for doc/chunk/source traceability.
- Optional API-key auth (`RAG_SERVICE_API_KEY`) and in-process per-minute rate limiting (`RAG_RATE_LIMIT_PER_MIN`).
- Request tracing via `x-request-id` propagation (generated if missing).
- LLM audit hook persistence (`llm_audit` table) storing hashes/metadata only (no raw prompt/response secrets).

## Environment Variables

- `EMBEDDINGS_BIN` (default: `./embeddings/build/sbert_search`)
- `EMBEDDINGS_MODEL` (**required**): path to GGML model file.
- `RAG_METADATA_DB` (default: `rag_service/rag_metadata.db`)
- `LLM_BASE_URL` (default: `https://api.openai.com`)
- `LLM_API_KEY` (optional, but required by most providers)
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `ANTHROPIC_BASE_URL` (default: `https://api.anthropic.com`)
- `ANTHROPIC_API_KEY` (required for `llm_provider=anthropic`)
- `ANTHROPIC_MODEL` (default: `claude-3-5-sonnet-latest`)
- `RAG_SERVICE_API_KEY` (optional; if set, clients must send `x-api-key`)
- `RAG_RATE_LIMIT_PER_MIN` (default: `120`)
- `RAG_LLM_FAILOVER_CHAIN` (default: `openai,anthropic`)
- `RAG_LLM_MAX_RETRIES` (default: `2`)
- `RAG_LLM_BACKOFF_BASE_MS` (default: `250`)
- `RAG_LLM_CIRCUIT_BREAKER_THRESHOLD` (default: `3`)
- `RAG_LLM_CIRCUIT_BREAKER_COOLDOWN_S` (default: `60`)

### Search hardening options

`POST /search` now supports Day-3 guardrail fields:

- `max_results` (default `25`, max `200`) to cap output size
- `dedup_by_chunk_id` (default `true`) to avoid duplicate chunk payloads in response

Mode-specific constraints:

- `knn`: `k <= max_results`
- `radius`: `min_score > 0`
- `adaptive`: `minN <= lookahead`

The response now includes telemetry fields:

- `mode`
- `latency_ms`
- `query_chars`
- `result_count`

## Run

```bash
pip install -r rag_service/requirements.txt
EMBEDDINGS_MODEL=/path/to/sbert.ggml uvicorn rag_service.app:app --reload --port 8080
```

## Example

```bash
curl -X POST localhost:8080/ingest -H 'content-type: application/json' -d '{
  "index": "kb",
  "doc_id": "doc-1",
  "chunks": ["RAG combines retrieval and generation."],
  "metadata": {"source": "intro.md"}
}'

curl -X POST localhost:8080/search -H 'content-type: application/json' -d '{
  "index": "kb",
  "query": "What is RAG?",
  "mode": "knn",
  "k": 5
}'

curl -X POST localhost:8080/namespaces/kb/config -H 'content-type: application/json' -d '{
  "embedding_model": "sbert.ggml",
  "embedding_dimension": 384,
  "metric": "cosine",
  "hnsw_m": 16,
  "hnsw_ef_construction": 200,
  "hnsw_ef_search": 64,
  "metadata": {"owner": "rag-team"}
}'

curl -X POST localhost:8080/answer -H 'content-type: application/json' -d '{
  "index": "kb",
  "query": "What is RAG?",
  "llm_provider": "anthropic"
}'
```


### Day-4 answer reliability

`POST /answer` now uses configured provider failover and retries.

- Primary provider comes from request `llm_provider`
- Additional providers come from `RAG_LLM_FAILOVER_CHAIN`
- Retries use exponential backoff (`RAG_LLM_MAX_RETRIES`, `RAG_LLM_BACKOFF_BASE_MS`)
- Circuit breaker opens after repeated failures per provider

The `/answer` response now includes:

- `provider_attempts`
- `failover_attempt_log`

### Day-5 grounding + tracing

- `/answer` now uses a stricter citation-aware prompt template requiring inline chunk citations in form `[chunk_id=<value>]`.
- Middleware propagates `x-request-id` to responses and generates one if absent.
- Structured JSON logs are emitted for answer start/fail/complete events.
- Audit hooks persist prompt/response hashes and retrieval metadata into SQLite (`llm_audit`) without storing raw content.


### Day-6 evaluation harness

Run offline sample scoring:

```bash
python rag_service/eval/evaluate_rag.py \
  --offline-results rag_service/eval/sample_eval_results.json \
  --out rag_service/eval/latest_report.json
```

Run online against a running service:

```bash
python rag_service/eval/evaluate_rag.py \
  --cases rag_service/eval/sample_eval_cases.json \
  --base-url http://localhost:8080
```

Reported metrics include: `MRR`, `Recall@k`, `nDCG@k`, `keyword_hit`, `citation_precision`, `citation_recall`.

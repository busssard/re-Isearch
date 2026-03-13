# RAG Setup Guide for re-Isearch Embeddings

## What is still missing (read this first)

Before using this in a production RAG stack, these gaps remain:

1. **Vendor HNSW build issues in this environment (still open)**
   - The embeddings tree still has compile blockers in vendored HNSW/SIMD paths on this container/toolchain.
   - Validate on your target build image and pin compiler/flags.

2. **Service hardening is now partially implemented, but still incomplete**
   - `rag_service/` now includes optional API-key auth (`RAG_SERVICE_API_KEY`) and per-minute rate limiting (`RAG_RATE_LIMIT_PER_MIN`).
   - Still missing for production: distributed/global rate limiting, stronger tenancy isolation, and production WAF policy.

3. **LLM provider coverage expanded, but failover policy is still pending**
   - `/answer` now supports `llm_provider` values `openai` and `anthropic`.
   - Still missing: deterministic failover chains, circuit breaking, and provider health checks.

4. **Metadata schema/versioning now implemented for namespace config, but governance is pending**
   - `rag_service` now stores namespace-level retrieval config (`embedding_model`, `dimension`, `metric`, HNSW knobs) via `/namespaces/{index}/config`.
   - Still missing: migration policy/version contracts and CI checks that block incompatible schema drift.

5. **Evaluation harness is still not end-to-end RAG**
   - Existing tests validate embedding/index behavior, but not full retrieval-grounded answer quality.
   - Add golden QA sets, retrieval hit-rate checks, and hallucination/factuality scoring.

6. **Operational packaging is not turnkey**
   - You still need deployment concerns: snapshots/backups, index lifecycle jobs, monitoring, auth/rate-limit telemetry, and incident runbooks.

---

## 1) Architecture you should build

Use a 4-layer RAG architecture:

1. **Embedding provider layer**
   - Encodes chunks/queries.
   - Use SBERT/GGML from this repo initially.

2. **Vector index layer**
   - Uses this embeddings engine for add/search/remove.
   - Keep index files and offset files per namespace.

3. **Query orchestration layer**
   - Retrieves candidates (`knn`, `relative`, `adaptive`).
   - Applies filtering/reranking/context-window packing.

4. **LLM API adapter layer**
   - Calls your selected LLM API with retrieved context and user question.

This layering matches the repository integration guidance and keeps concerns separate.

---

## 2) Build the embeddings engine

From repo root:

```bash
cd embeddings
mkdir -p build && cd build
cmake ..
make -j
```

CMake expects `libbert` and `libggml` discoverable via `../lib`/`lib` search paths.

---

## 3) Verify local retrieval flow

Use the included script pattern:

```bash
cd ../test
./run_test.sh
```

This demonstrates:
- ingest (`append`)
- retrieval (`knn`, `radius`, `relative`, `adaptive`)
- reconstruction (`reconstruct_sid`)

---

## 4) Choose retrieval mode for v1

Recommended rollout:

1. Start with **kNN** (`k=5..12`) for stable behavior.
2. Add **relative** threshold mode for adaptive precision.
3. Add **adaptive** mode once you have benchmark data.

Do not start with over-optimized heuristics. Get baseline quality first.

---

## 5) Ingestion pipeline design

For each source document:

1. Normalize text and metadata.
2. Chunk into sentence/paragraph windows.
3. Generate stable IDs:
   - `doc_id`
   - `chunk_id`
   - optional `source_uri`
4. Write to embeddings index namespace:
   - `append` / `appendid`
5. Persist external metadata store keyed by chunk IDs.

Minimum metadata to persist per index namespace:
- embedding model name
- embedding dimension
- metric semantics (L2/IP/Cosine)
- HNSW params: `M`, `ef_construction`, `ef_search`

---

## 6) Build a retriever service (thin wrapper)

Implement a small service (C++ or Python wrapper) with endpoints:

### `POST /ingest`
Input:
```json
{
  "index": "knowledge_base",
  "doc_id": "doc-123",
  "chunks": ["...", "..."],
  "metadata": {"source": "..."}
}
```

### `POST /search`
Input:
```json
{
  "index": "knowledge_base",
  "query": "How does X work?",
  "mode": "knn",
  "k": 8
}
```
Output should include:
- score
- chunk text
- chunk/doc identifiers
- source metadata

### `POST /answer`
Input:
```json
{
  "index": "knowledge_base",
  "query": "How does X work?",
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini"
}
```
Flow:
1. retrieve chunks
2. build constrained prompt
3. call external LLM API
4. return answer + citations

---

## 7) Connect any LLM API

Your adapter should standardize:

- `generate(prompt, model, temperature, max_tokens)`
- provider-specific auth and retries
- timeout and rate-limit handling

Prompt contract (recommended):
- “Use only the provided context.”
- “If context is insufficient, explicitly say unknown.”
- “Return source IDs used.”

---

## 8) Context assembly best practices

When packing context for LLM call:

1. Deduplicate near-identical chunks.
2. Keep highest scoring chunks first.
3. Cap token budget (e.g., 1.5k–4k context tokens depending on model).
4. Preserve source IDs inline for citation traceability.

---

## 9) Hybrid retrieval (recommended phase 2)

Add lexical + semantic blend:

1. semantic retrieval candidate set
2. lexical retrieval candidate set
3. union and rerank
4. context pack top N

This improves edge cases (exact terms, identifiers, short acronyms).

---

## 10) Evaluation before production

Create an evaluation suite:

1. **Retrieval metrics**
   - Recall@k, MRR, nDCG
2. **Answer metrics**
   - groundedness / citation correctness
   - factuality checks against known answers
3. **Operational metrics**
   - P50/P95 latency
   - token usage cost
   - error rates / timeout rates

Gate releases on measurable thresholds.

---

## 11) Operations checklist

- [ ] snapshot/backup index files
- [ ] index versioning and migration policy
- [ ] rolling rebuild strategy
- [ ] access control and tenant isolation
- [ ] observability dashboards + alerting
- [ ] prompt/version audit logs

---

## 12) Minimal implementation plan (7-day sprint)

**Day 1**: Run `scripts/day1_readiness_check.sh` to run dependency preflight, configure/build embeddings (fallback mode if libs are missing), execute smoke tests when inference libs exist, and verify `rag_service` syntax.

Day 1 is considered complete when the script reports no failures. If external inference libs are not yet available, warnings are acceptable only when tracked in `docs/DAY1_STATUS.md`.

**Day 2**: Close Day‑1 dependency warnings in your target environment (`libbert`/`libggml` placement or env vars), then harden `/ingest` + metadata persistence (stable sentence IDs, identifier validation, metadata schema/indexes, partial-failure reporting).

**Day 3 (start now)**: Harden `/search` and retrieval orchestration.
- ✅ Added request bounds and mode-specific guardrails (tracked in `docs/DAY3_STATUS.md`).
- ✅ Added per-request timeout wrapper and explicit HTTP error mapping for search backend failures.
- ✅ Added retrieval response normalization (stable ordering, cap results, optional dedup by chunk_id).
- ✅ Added search telemetry fields (`mode`, `latency_ms`, `query_chars`, `result_count`) in responses.
- Remaining: move timeout/cancellation to process-level controls for production-grade kill semantics.

**Day 4**: Harden `/answer` for multi-provider reliability.
- ✅ Implemented provider failover sequence policy (primary from request + configurable chain via `RAG_LLM_FAILOVER_CHAIN`).
- ✅ Added retry/backoff/circuit-breaker conditions per provider (`RAG_LLM_MAX_RETRIES`, `RAG_LLM_BACKOFF_BASE_MS`, breaker thresholds/cooldown).
- ✅ Added provider health endpoint/check (`GET /health/providers`) and retained model override policy (`llm_model`).
- Remaining: move circuit-breaker state to shared storage for multi-process deployments.

**Day 5**: Strengthen grounding and traceability.
- ✅ Added citation-aware prompt template with strict chunk-id citation form (`[chunk_id=<value>]`) in `/answer`.
- ✅ Added request ID propagation (`x-request-id`) and structured JSON logs for answer lifecycle events.
- ✅ Added prompt/response audit record hooks using hashes only (no raw secret content) via `llm_audit` table.
- Remaining: expand request-id/log/audit coverage from `/answer` to full `/ingest` and `/search` lifecycle paths.

**Day 6**: Build end-to-end evaluation harness.
- ✅ Added Day-6 evaluation harness script (`rag_service/eval/evaluate_rag.py`) with online and offline scoring modes.
- ✅ Implemented metrics: `Recall@k`, `MRR`, `nDCG@k`, plus citation/answer proxy metrics (`keyword_hit`, `citation_precision`, `citation_recall`).
- ✅ Added sample eval datasets and report generation artifacts for baseline comparisons.
- Remaining: add domain-specific golden-answer labels and explicit latency/cost/error promotion thresholds.

**Day 7**: Production hardening and operational closure.
- Snapshot/backup strategy, restore drills, and index lifecycle jobs.
- Access control/tenant isolation policy and rate-limit telemetry dashboards.
- Deployment checklist + runbooks + rollback plan.

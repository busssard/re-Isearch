# Day 2 Status — Ingest + Metadata Hardening

This document tracks Day 2 implementation from `setup_RAG.md`.

## Scope

Day 2 objective:

- harden `/ingest`
- harden metadata persistence
- make ingestion behavior deterministic and debuggable

## Implemented in this pass

1. **Stable sentence IDs for ingestion**
   - Replaced Python runtime `hash(...)` based IDs with deterministic BLAKE2b-derived IDs.
   - Added collision probing against existing `sentence_id` values per index.

2. **Input validation hardening**
   - Added request constraints for `index`, `doc_id`, and `chunks` shape.
   - Added strict identifier validation (`[A-Za-z0-9._:-]`) for `index` and `doc_id`.
   - Empty/whitespace-only chunk payloads are rejected when no valid chunks remain.

3. **Metadata durability improvements**
   - Enabled SQLite `WAL` mode and `NORMAL` synchronous mode for better write behavior.
   - Added `created_at` + `updated_at` columns with lightweight migration for existing DB files.
   - Added indexes:
     - unique `(index_name, doc_id, chunk_id)`
     - secondary `(index_name, doc_id)`

4. **Ingest observability and failure reporting**
   - `/ingest` now returns `inserted`, `failed`, and per-chunk failure details.
   - Full failure returns HTTP 502 with structured error payload.

5. **Cross-cutting hardening completed early (pulled from later days)**
   - Added optional service API-key enforcement (`RAG_SERVICE_API_KEY`).
   - Added in-process per-minute request limiting (`RAG_RATE_LIMIT_PER_MIN`).
   - Added namespace metadata config endpoints to persist retrieval/index contract fields:
     - `POST /namespaces/{index}/config`
     - `GET /namespaces/{index}/config`
   - Added `anthropic` provider support in `/answer` alongside existing `openai` adapter.

## Known caveat for this environment

Runtime API tests requiring full service startup are currently constrained by missing Python package deps in this container (`httpx` not installed globally outside venv).

## Day 2 completion checklist

- [x] deterministic sentence IDs
- [x] identifier validation
- [x] metadata schema/index hardening
- [x] partial-failure reporting in ingest response
- [x] service-level auth/rate-limit baseline
- [x] namespace metadata contract persistence endpoints
- [x] second LLM provider path (`anthropic`)
- [ ] full integration test run of ingest->search->answer in a dependency-complete runtime

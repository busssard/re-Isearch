# Day 3 Status — Search Hardening and Retrieval Guardrails

This document tracks Day 3 implementation from `setup_RAG.md`.

## Scope

Day 3 objective:

- harden `/search`
- add mode-specific request guardrails
- add timeout/error mapping for backend failures
- normalize/cap retrieval output and expose telemetry

## Implemented in this pass

1. **Request bound hardening**
   - Added request-level bounds in `SearchRequest`:
     - `query` length cap
     - bounded `k`, `min_score`, `alpha`, `minN`, `lookahead`, `gapDelta`
     - added `max_results` and `dedup_by_chunk_id`

2. **Mode-specific guardrails**
   - Added explicit constraints:
     - `knn`: `k <= max_results`
     - `radius`: `min_score > 0`
     - `adaptive`: `minN <= lookahead`

3. **Timeout + backend error mapping**
   - Added `execute_search_with_timeout(...)` wrapper with explicit:
     - HTTP `504` on timeout
     - HTTP `502` on backend CLI/search errors

4. **Retrieval normalization**
   - Added stable sort normalization by score/doc/chunk/sentence.
   - Added optional chunk-level dedup (`dedup_by_chunk_id`).
   - Added hard cap via `max_results`.

5. **Search telemetry fields**
   - `/search` response now includes:
     - `mode`
     - `latency_ms`
     - `query_chars`
     - `result_count`

## Known caveat

Search timeout currently uses an in-process thread wrapper around CLI calls. For full production safety, move timeout control into a subprocess boundary with stronger cancellation guarantees.

## Day 3 completion checklist

- [x] request bounds
- [x] mode-specific validation
- [x] timeout/error mapping
- [x] result normalization + cap + dedup
- [x] telemetry fields in response
- [ ] production-grade cancellation semantics for long-running CLI calls

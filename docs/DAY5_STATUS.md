# Day 5 Status — Grounding, Tracing, and Audit Hooks

This document tracks Day 5 implementation from `setup_RAG.md`.

## Scope

Day 5 objective:

- strengthen citation grounding in `/answer`
- add request ID propagation and structured service logs
- add audit hooks for prompt/response lifecycle without storing secrets

## Implemented in this pass

1. **Citation-aware prompt template**
   - Added a dedicated prompt builder for `/answer` that enforces strict citation behavior:
     - inline citations must use `[chunk_id=<value>]`
     - explicit unknown response when context is insufficient
     - no fabrication policy baked into prompt instructions

2. **Request ID propagation**
   - Middleware now reads `x-request-id` (or generates one) and attaches it to:
     - `request.state.request_id`
     - response header `x-request-id`
   - `/answer` response payload now includes `request_id`.

3. **Structured JSON logs**
   - Added structured logging helper for key answer lifecycle events:
     - `answer_request_started`
     - `answer_request_failed`
     - `answer_request_completed`

4. **Audit hook persistence (no raw secrets)**
   - Added `llm_audit` table in metadata SQLite with hashed prompt/response fields.
   - Stored fields include request/index/provider/model/retrieval counts plus SHA-256 hashes.
   - Raw prompt/response text is not stored.

## Known caveat

Audit hooks currently focus on `/answer` path only. In a fuller rollout, add similar audit records for `/ingest` and `/search` control-plane actions if required by compliance policy.

## Day 5 completion checklist

- [x] citation-aware prompt template
- [x] request ID propagation (`x-request-id`)
- [x] structured logging hooks
- [x] prompt/response hash-based audit persistence
- [ ] extended cross-endpoint audit policy coverage

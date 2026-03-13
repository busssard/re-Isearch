# Day 4 Status — Multi-provider Answer Reliability

This document tracks Day 4 implementation from `setup_RAG.md`.

## Scope

Day 4 objective:

- harden `/answer` reliability
- add deterministic failover policy
- add retry/backoff and circuit-break behavior
- expose provider health status and model selection behavior

## Implemented in this pass

1. **Failover sequence policy**
   - `/answer` now builds a provider chain using:
     - primary provider from request (`llm_provider`)
     - configured chain from `RAG_LLM_FAILOVER_CHAIN`
   - Providers are attempted in deterministic order without duplicates.

2. **Retry/backoff controls**
   - Added configurable retry strategy per provider call:
     - `RAG_LLM_MAX_RETRIES`
     - `RAG_LLM_BACKOFF_BASE_MS`
   - Backoff uses exponential progression between attempts.

3. **Circuit-breaker behavior**
   - Added in-memory provider health states with:
     - consecutive failure counters
     - open-circuit cooldown windows
     - last error / last success / last attempt timestamps
   - Controlled by:
     - `RAG_LLM_CIRCUIT_BREAKER_THRESHOLD`
     - `RAG_LLM_CIRCUIT_BREAKER_COOLDOWN_S`

4. **Provider health endpoint**
   - Added `GET /health/providers` returning provider-level breaker and health status.

5. **Answer response observability**
   - `/answer` now returns:
     - selected provider/model
     - `provider_attempts`
     - `failover_attempt_log`

## Known caveat

Circuit-breaker state is currently process-local (in-memory). For multi-instance deployments, move state to shared storage (Redis or equivalent) for consistent behavior.

## Day 4 completion checklist

- [x] failover policy implemented
- [x] retry/backoff controls implemented
- [x] circuit-breaker behavior implemented
- [x] provider health endpoint added
- [x] answer-level attempt telemetry added
- [ ] shared state for multi-process consistency

# Day 1 Status — RAG Foundation Checks

This file tracks completion state for Day 1 from `setup_RAG.md`.

## Scope

Day 1 target: verify the repository can run an initial local readiness flow for embeddings + RAG service scaffolding without introducing regressions.

## Did we lose `libbert`?

Short answer: no repository-owned `libbert` artifact appears to have been removed in recent commits.

- The embeddings build expects external `libbert`/`libggml` binaries.
- Those binaries are not tracked in git under `lib/` or `embeddings/lib/` in this repository history.
- Day 1 tooling now treats missing external inference libraries as a **documented dependency gap**, not as a source-tree regression.

## What was implemented

- Added automated readiness script: `scripts/day1_readiness_check.sh`.
- Updated the script so it can:
  1. detect whether `libbert` + `libggml` are available
  2. run full inference-target configure/build/smoke when present
  3. fall back to `-DEMBEDDINGS_REQUIRE_INFERENCE_LIBS=OFF` when absent (still validates build plumbing and service syntax)
- Updated `embeddings/CMakeLists.txt` to support `EMBEDDINGS_REQUIRE_INFERENCE_LIBS` and to avoid unconditional macOS-only linker flags on non-Apple systems.
- Updated Day 1 plan in `setup_RAG.md` with explicit completion criteria.

## Latest execution result

Command:

```bash
./scripts/day1_readiness_check.sh
```

Observed status:

- PASS: 2
- WARN: 3
- FAIL: 0

Warnings:

- Inference libs (`libbert`/`libggml`) not found in expected local library directories.
- Embeddings configure intentionally reports warning mode because fallback configure (`EMBEDDINGS_REQUIRE_INFERENCE_LIBS=OFF`) is used.
- Embeddings build + smoke checks skipped because inference libs are unavailable in this environment.

## Current Day 1 completion state

- [x] Day 1 validation script exists and is executable.
- [x] Day 1 script now completes without hard failure when only external inference libs are missing.
- [x] CMake fallback mode allows day-level work to remain self-contained and non-breaking.
- [ ] Full inference-mode build + smoke test (`embeddings/tests/run_test.sh`) still pending on an environment that provides `libbert` + `libggml`.

## Next actions to fully close Day 1 in production-like environments

1. Provide `libbert` and `libggml` in `lib/`, `embeddings/lib/`, or via `BERT_LIB_DIR` / `GGML_LIB_DIR`.
2. Re-run `./scripts/day1_readiness_check.sh` and confirm warning count drops to zero.
3. Record the successful environment/toolchain details for reproducibility.

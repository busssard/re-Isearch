# Code-smell Remediation Status

This change performs a concentrated remediation pass on the actively developed embedding stack under `embeddings/` (excluding vendored/legacy mirrors).

## Scope remediated in this pass

Included scope:
- `embeddings/main.cpp`
- `embeddings/src/*`
- `embeddings/include/*` (excluding `include/hnswlib/*`, `include/ggml.h`, `include/bert.h`)
- `embeddings/utils/*`

Excluded from direct remediation in this pass:
- vendored/third-party code (`embeddings/include/hnswlib/*`, external libs)
- archival/legacy copies (`embeddings/old/*`, `embeddings/unified/*`)
- generated wrapper code and other external trees in repository root

## Results (included scope)

The following smell classes were reduced to zero in included scope:
- `TODO/FIXME/HACK/XXX`
- `#if 0` blocks
- `catch (...)`

## Functional fixes bundled with remediation

- Corrected `remove_byAddress(...)` delegation in `embeddings/main.cpp`.
- Corrected `BertIndexManager::merge(...)` to run full `merge()` in `embeddings/src/BertIndexManager.cpp`.
- Corrected disabled production append signature typo (`int64_end` -> `int64_t end`) in `embeddings/main.cpp`.


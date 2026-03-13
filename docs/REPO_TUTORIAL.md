# re-Isearch Repository Tutorial (Modernized Orientation)

This tutorial is a practical map of the repository for contributors who need to understand **what to change, where, and why**.

## 1) Mental model

re-Isearch is organized as a core C/C++ search engine with supporting modules, plugins, bindings, and documentation:

- `src/`: core search/indexing engine and CLI tooling.
- `plugins/`: compiled document-type plugins (`.sob`) for format-specific processing.
- `swig/`: language binding interface and generated wrapper artifacts.
- `embeddings/`: modern dense-vector/embedding prototype built around SBERT + GGML + HNSW.
- `docs/`: design papers, handbooks, and practical references.

## 2) Suggested contributor path (orderly onboarding)

If you are new to the codebase, follow this sequence:

1. Read `README.md` (project goals and broad architecture).
2. Read this file (`docs/REPO_TUTORIAL.md`) to understand where to work.
3. Read `docs/EMBEDDINGS_INTEGRATION_GUIDE.md` for dense vector support strategy.
4. Read `embeddings/README.md` and `embeddings/main.cpp` for the current embedding implementation.
5. Move into `src/` only after understanding the extension boundaries between lexical and vector retrieval.

## 3) Core extension points

### Lexical / classic engine

- Add parser/doctype behavior in plugin and doctype integration layers.
- Add ranking or query behavior in `src/` components used by `Isearch` / `Iindex` tools.

### Dense embeddings

- Current reference implementation is in `embeddings/main.cpp`.
- It supports sentence embedding creation, shard-aware ANN indexing, and search modes.
- Use it as a transition path toward deeper integration into the core engine.

## 4) Refactor priorities (recommended)

The repository has historical naming conventions and monolithic files. Use this order:

1. **Name clarity first**
   - Replace ambiguous variable names in touched modules.
   - Prefer explicit names (`next_available_label`) over abbreviated names.

2. **Module extraction second**
   - Separate storage, embedding, and retrieval logic into dedicated headers/sources.
   - Keep wrappers thin and business logic testable.

3. **Behavior-preserving cleanup third**
   - Avoid changing ranking semantics during naming refactors.
   - Add comments to explain invariants (offset encoding, shard merge behavior).

## 5) Practical coding standard for modernization

- Prefer self-describing identifiers and focused methods.
- Keep backward compatibility in serialized formats unless explicitly versioned.
- Add documentation alongside code changes, not as a later pass.
- Favor small, reviewable refactors over giant rewrites.

## 6) What is possible now

- Classic indexing/search workflows through existing engine and tools.
- Plugin-based content format support.
- SWIG-driven language binding paths.
- Experimental/advanced embedding search via `embeddings/`.


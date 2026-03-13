# Embeddings Integration Guide for re-Isearch

This guide explains how to add or evolve dense-vector retrieval in a way that fits re-Isearch's architecture.

## 1) Current implementation baseline

The embedding search reference implementation is in `embeddings/main.cpp` and currently includes:

- SBERT/GGML model loading and text encoding.
- HNSW-based ANN index management.
- Sharding logic with merge support.
- Multiple search modes (kNN, radius, relative, adaptive).
- On-disk offset mapping for sentence text reconstruction.

## 2) Integration strategy

Use a layered model:

1. **Embedding provider layer**
   - Converts text to vectors.
   - Can be backed by SBERT/GGML, llama.cpp embedding models, or external providers.

2. **Vector index layer**
   - Adds/searches/removes vectors.
   - Persists ANN graph + offset metadata.

3. **Query orchestration layer**
   - Runs lexical-only, vector-only, or hybrid query flows.
   - Fuses/reranks results using deterministic policies.

4. **Storage/schema layer**
   - Stores model identity and vector dimension metadata.
   - Version serialized offset/index formats when changed.

## 3) Minimum metadata required

For safe embeddings operations, record at least:

- Embedding model ID/name.
- Embedding dimension.
- Distance metric (L2/IP/Cosine semantics).
- Index build-time parameters (`M`, `ef_construction`, `ef_search`).

Without this metadata, restored indexes can silently become incompatible.

## 4) Recommended refactor slices

To reduce risk, modernize in slices:

- Slice A: isolate byte-order and offset-file helpers.
- Slice B: isolate embedding model wrapper API.
- Slice C: isolate shard lifecycle (create/load/flush/merge).
- Slice D: isolate search policy methods and threshold logic.

Each slice should preserve existing behavior and include doc updates.

## 5) Hybrid search possibilities

Once integration is complete, re-Isearch can support:

- Semantic-only retrieval (embedding nearest-neighbor).
- Lexical-only retrieval (existing engine strengths).
- Hybrid retrieval (candidate generation from one mode, rerank with the other).
- Field-aware embeddings (dense vectors only on selected fields).

## 6) Operational guidance

- Keep shard counts modest per node; scale horizontally when needed.
- Flush on controlled thresholds to balance durability and throughput.
- Validate vector-dimension compatibility before loading index shards.
- Treat deleted vector labels and offset tombstones as first-class state.


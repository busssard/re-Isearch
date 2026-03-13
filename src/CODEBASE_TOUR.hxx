/* Copyright (c) 2020-21 Project re-Isearch and its contributors: See CONTRIBUTORS.
It is made available and licensed under the Apache 2.0 license: see LICENSE */

#ifndef CODEBASE_TOUR_HXX
#define CODEBASE_TOUR_HXX

/*
 * re-Isearch codebase tour (developer quick reference)
 * ====================================================
 *
 * Purpose
 * -------
 * This header is intentionally comment-only and serves as in-code orientation
 * for maintainers working in this historical codebase.
 *
 * Repository-level map
 * --------------------
 * - src/       : core indexing/search engine and command implementations
 * - doctype/   : document type handlers/parsers used during indexing
 * - plugins/   : compiled plugin modules (.sob) for format-specific behavior
 * - swig/      : cross-language binding interfaces/wrappers
 * - embeddings/   : dense-vector ANN prototype (SBERT/GGML/HNSW)
 * - docs/      : handbooks, design notes and modernization guidance
 *
 * Core runtime path
 * -----------------
 * 1) Iindex/Iutil/Isearch style CLIs parse user input and DB selection.
 * 2) VIDB/IDB/INDEX objects open index files and metadata stores.
 * 3) Parsing + doctype logic converts source records to normalized fields.
 * 4) Dictionaries/posting structures are persisted and optionally merged.
 * 5) Query components (SQUERY/RSET/IRSET) execute retrieval and ranking.
 * 6) Results resolve back to MDT/RECORD metadata for rendering.
 *
 * Important source clusters inside src/
 * ------------------------------------
 * - `*main.cxx|.hxx`    : command entrypoint implementations
 * - `index.*`           : central index orchestration APIs
 * - `record.*` / `mdt*` : record and metadata table representation
 * - `db*`               : low-level key/value database wrappers
 * - `squery*` `rset*`   : query parsing, evaluation and result containers
 * - `fuzzy*` `geo*`     : specialized retrieval modes
 * - `stoplist*`         : language stopword handling
 *
 * Modernization guidance
 * ----------------------
 * - Prefer behavior-preserving refactors (naming/comments/extraction).
 * - Keep serialized layout compatibility unless versioning is introduced.
 * - Split monolithic units by responsibility boundaries:
 *     parsing, execution, serialization, formatting.
 * - Add documentation in touched files as part of each change.
 */

#endif

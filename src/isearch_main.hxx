/*
Copyright (c) 2020-21 Project re-Isearch and its contributors: See CONTRIBUTORS.
It is made available and licensed under the Apache 2.0 license: see LICENSE
*/

/*
 * Isearch CLI entrypoint contract
 * ===============================
 *
 * This header intentionally stays minimal because it is included by C and C++
 * launcher wrappers. `_Isearch_main()` contains the full command-line search
 * flow implemented in `isearch_main.cxx`:
 *
 *   1) parse CLI flags/query expression
 *   2) open one or more database contexts via VIDB/IDB
 *   3) compile/evaluate query using query + result-set components
 *   4) stream formatted output (records, fields, diagnostics, timing)
 *
 * Keeping this ABI-style declaration isolated avoids pulling large C++ headers
 * into wrappers and preserves long-standing launcher compatibility.
 */
extern "C" {
 int _Isearch_main (int argc, char **argv);
}

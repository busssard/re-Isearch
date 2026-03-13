/*
Copyright (c) 2020-21 Project re-Isearch and its contributors: See CONTRIBUTORS.
It is made available and licensed under the Apache 2.0 license: see LICENSE
*/

/*
 * Iutil CLI entrypoint contract
 * =============================
 *
 * `_Iutil_main()` is the utility/maintenance counterpart to the user-facing
 * search binary. Its implementation in `iutil_main.cxx` drives operations such
 * as diagnostics, metadata inspection, integrity checks and maintenance tasks
 * over one or more indexes.
 *
 * This declaration is kept C-linkage friendly so tiny wrappers can call into
 * the C++ implementation without exposing the rest of the C++ symbol surface.
 */
extern "C" {
 int _Iutil_main (int argc, char **argv);
}

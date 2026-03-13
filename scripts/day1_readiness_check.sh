#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMBEDDINGS_BUILD_DIR="${REPO_ROOT}/embeddings/build"
EMBEDDINGS_TEST_SCRIPT="${REPO_ROOT}/embeddings/tests/run_test.sh"
HAS_INFERENCE_LIBS=1

pass_count=0
warn_count=0
fail_count=0

run_check() {
  local label="$1"
  shift
  if "$@"; then
    echo "[PASS] ${label}"
    pass_count=$((pass_count + 1))
  else
    local rc=$?
    if [[ ${rc} -eq 2 ]]; then
      echo "[WARN] ${label}"
      warn_count=$((warn_count + 1))
    else
      echo "[FAIL] ${label}"
      fail_count=$((fail_count + 1))
    fi
  fi
}

check_embeddings_configure() {
  local bert_path ggml_path
  bert_path="$(find "${REPO_ROOT}/lib" "${REPO_ROOT}/embeddings/lib" -maxdepth 1 -type f \( -name 'libbert.so' -o -name 'libbert.a' -o -name 'libbert.dylib' \) 2>/dev/null | head -n1)"
  ggml_path="$(find "${REPO_ROOT}/lib" "${REPO_ROOT}/embeddings/lib" -maxdepth 1 -type f \( -name 'libggml.so' -o -name 'libggml.a' -o -name 'libggml.dylib' \) 2>/dev/null | head -n1)"

  if [[ -z "${bert_path}" || -z "${ggml_path}" ]]; then
    echo "Optional inference libs not found (libbert/libggml). Configuring in fallback mode for Day 1 sanity checks."
    HAS_INFERENCE_LIBS=0
    cmake -S "${REPO_ROOT}/embeddings" -B "${EMBEDDINGS_BUILD_DIR}" -DEMBEDDINGS_REQUIRE_INFERENCE_LIBS=OFF
    return 2
  fi

  cmake -S "${REPO_ROOT}/embeddings" -B "${EMBEDDINGS_BUILD_DIR}" -DEMBEDDINGS_REQUIRE_INFERENCE_LIBS=ON
}

check_embeddings_build() {
  cmake --build "${EMBEDDINGS_BUILD_DIR}" -j
}

check_embeddings_smoke() {
  if [[ ${HAS_INFERENCE_LIBS} -eq 0 ]]; then
    echo "Skipping smoke test because libbert/libggml are not available in this environment."
    return 2
  fi
  if [[ ! -x "${EMBEDDINGS_TEST_SCRIPT}" ]]; then
    echo "embeddings test script not executable: ${EMBEDDINGS_TEST_SCRIPT}"
    return 1
  fi
  (
    cd "${REPO_ROOT}/embeddings/tests" &&
    "${EMBEDDINGS_TEST_SCRIPT}"
  )
}

check_rag_service_syntax() {
  PYTHONDONTWRITEBYTECODE=1 python -m py_compile "${REPO_ROOT}/rag_service/app.py"
}

check_setup_doc_mentions_script() {
  if rg -n "day1_readiness_check\.sh" "${REPO_ROOT}/setup_RAG.md" >/dev/null; then
    return 0
  fi
  return 1
}

echo "=== Day 1 readiness check ==="

run_check "Embeddings CMake configure" check_embeddings_configure

if [[ ${fail_count} -eq 0 && ${HAS_INFERENCE_LIBS} -eq 1 ]]; then
  run_check "Embeddings build" check_embeddings_build
  run_check "Embeddings smoke test script" check_embeddings_smoke
elif [[ ${fail_count} -eq 0 ]]; then
  echo "Skipping embeddings build/tests because inference libs are absent in this environment."
  warn_count=$((warn_count + 2))
else
  echo "Skipping embeddings build/tests due to configure failure."
  warn_count=$((warn_count + 2))
fi

run_check "RAG service Python syntax" check_rag_service_syntax
run_check "setup_RAG.md references Day 1 checker" check_setup_doc_mentions_script

echo "=== Summary ==="
echo "PASS: ${pass_count}"
echo "WARN: ${warn_count}"
echo "FAIL: ${fail_count}"

if [[ ${fail_count} -gt 0 ]]; then
  exit 1
fi

if [[ ${warn_count} -gt 0 ]]; then
  exit 2
fi

exit 0

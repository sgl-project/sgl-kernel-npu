#!/usr/bin/env bash
set -euo pipefail

export ACL_OP_SELECT_IMPL_MODE=high_precision

WORKSPACE="$(cd "${GITHUB_WORKSPACE:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd -P)"
TEST_ROOT="$WORKSPACE/tests/python"
TEST_GROUP="${1:-all}"
MEMFABRIC_METADATA_TEST="tests/python/operators/mem_cache/test_build_kv_sparse_metadata.py"

case "$TEST_GROUP" in
    speculative) TEST_DIRS=("$TEST_ROOT/operators/speculative") ;;
    matmul) TEST_DIRS=("$TEST_ROOT/operators/matmul") ;;
    mamba) TEST_DIRS=("$TEST_ROOT/operators/mamba") ;;
    moe) TEST_DIRS=("$TEST_ROOT/operators/moe") ;;
    embedding) TEST_DIRS=("$TEST_ROOT/operators/embedding") ;;
    data_move) TEST_DIRS=("$TEST_ROOT/operators/data_move") ;;
    activation) TEST_DIRS=("$TEST_ROOT/operators/activation") ;;
    diffusion)          TEST_DIRS=("$TEST_ROOT/operators/diffusion") ;;
    attention)          TEST_DIRS=("$TEST_ROOT/operators/attention") ;;
    batch_invariant)    TEST_DIRS=("$TEST_ROOT/operators/batch_invariant") ;;
    elementwise)        TEST_DIRS=("$TEST_ROOT/operators/elementwise") ;;
    examples)           TEST_DIRS=("$TEST_ROOT/operators/examples") ;;
    hyper_conn)         TEST_DIRS=("$TEST_ROOT/operators/hyper_conn") ;;
    indexer)            TEST_DIRS=("$TEST_ROOT/operators/indexer") ;;
    lora)               TEST_DIRS=("$TEST_ROOT/operators/lora") ;;
    mem_cache)          TEST_DIRS=("$TEST_ROOT/operators/mem_cache") ;;
    multimodal)         TEST_DIRS=("$TEST_ROOT/operators/multimodal") ;;
    norm)               TEST_DIRS=("$TEST_ROOT/operators/norm") ;;
    rope)               TEST_DIRS=("$TEST_ROOT/operators/rope") ;;
    sample)             TEST_DIRS=("$TEST_ROOT/operators/sample") ;;
    sampler)            TEST_DIRS=("$TEST_ROOT/operators/sampler") ;;
    operator_framework) TEST_DIRS=("$TEST_ROOT/framework/operator_framework") ;;
    testing_framework)  TEST_DIRS=("$TEST_ROOT/framework/testing_framework") ;;
    all)                TEST_DIRS=("$TEST_ROOT/operators" "$TEST_ROOT/framework") ;;
    *)
        echo "Unknown test group: $TEST_GROUP" >&2
        exit 1
        ;;
esac

cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

TESTS=()
PASSED=()
FAILED=()

discover_tests() {
    local discovered test_file relative_path memfabric_test=""
    discovered=$(find "$@" -type f -name 'test_*.py' | LC_ALL=C sort) || return 1

    while IFS= read -r test_file; do
        [[ -n "$test_file" ]] || continue
        relative_path="${test_file#"$WORKSPACE"/}"
        if [[ "${GITHUB_EVENT_NAME:-}" == pull_request && "$relative_path" == "$MEMFABRIC_METADATA_TEST" ]]; then
            memfabric_test="$relative_path"
        else
            TESTS+=("$relative_path")
        fi
    done <<< "$discovered"

    # Installation changes the test container, so run this file after its peers.
    if [[ -n "$memfabric_test" ]]; then
        TESTS+=("$memfabric_test")
    fi
}

install_memfabric_kernels() {
    local soc_version
    if ! command -v mfcli >/dev/null; then
        if python3 -c 'from importlib.util import find_spec; raise SystemExit(find_spec("memfabric_hybrid") is None)'; then
            echo "MemFabric Hybrid is installed but mfcli is not on PATH" >&2
            return 1
        fi
        echo "MemFabric Hybrid is not installed; leaving the dependency skip to pytest"
        return 0
    fi

    case "${NPU_HARDWARE:-}" in
        a3) soc_version=A3 ;;
        950) soc_version=A5 ;;
        *) echo "Unsupported NPU_HARDWARE for MemFabric: ${NPU_HARDWARE:-unset}" >&2; return 1 ;;
    esac
    echo "Installing MemFabric kernels: mfcli kernel install --soc-version $soc_version"
    mfcli kernel install --soc-version "$soc_version"
}

run_test() {
    local test_file="$1"
    local exit_code=0

    echo "=========================================="
    echo "Running: $test_file"
    echo "=========================================="

    if [[ "${GITHUB_EVENT_NAME:-}" == pull_request && "$test_file" == "$MEMFABRIC_METADATA_TEST" ]]; then
        install_memfabric_kernels || exit_code=$?
        if [[ "$exit_code" -ne 0 ]]; then
            echo "MemFabric kernel installation failed (exit code: $exit_code)" >&2
        fi
    fi
    if [[ "$exit_code" -eq 0 ]]; then
        python3 -m pytest "$test_file" -q -x --enable-perf=false || exit_code=$?
    fi

    if [[ "$exit_code" -eq 0 ]]; then
        PASSED+=("$test_file")
        echo "PASSED: $test_file"
    else
        FAILED+=("$test_file")
        echo "FAILED: $test_file (exit code: $exit_code)"
    fi
    echo
}

discover_tests "${TEST_DIRS[@]}"

if [[ "${#TESTS[@]}" -eq 0 ]]; then
    echo "No test files found in group: $TEST_GROUP" >&2
    exit 1
fi

echo "Running test group: $TEST_GROUP (${#TESTS[@]} test files)"
for test_file in "${TESTS[@]}"; do
    run_test "$test_file"
done

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total:  ${#TESTS[@]}"
echo "Passed: ${#PASSED[@]}"
echo "Failed: ${#FAILED[@]}"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "Failed test files:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi

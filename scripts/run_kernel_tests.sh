#!/usr/bin/env bash
set -euo pipefail

export ACL_OP_SELECT_IMPL_MODE=high_precision

WORKSPACE="$(cd "${GITHUB_WORKSPACE:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd -P)"
TEST_ROOT="$WORKSPACE/tests/python"
TEST_GROUP="${1:-all}"

case "$TEST_GROUP" in
    speculative) TEST_DIRS=("$TEST_ROOT/operators/speculative") ;;
    matmul) TEST_DIRS=("$TEST_ROOT/operators/matmul") ;;
    embedding) TEST_DIRS=("$TEST_ROOT/operators/embedding") ;;
    data_move) TEST_DIRS=("$TEST_ROOT/operators/data_move") ;;
    activation) TEST_DIRS=("$TEST_ROOT/operators/activation") ;;
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
    local discovered test_file relative_path
    discovered=$(find "$@" -type f -name 'test_*.py' | LC_ALL=C sort) || return 1

    while IFS= read -r test_file; do
        [[ -n "$test_file" ]] || continue
        relative_path="${test_file#"$WORKSPACE"/}"
        TESTS+=("$relative_path")
    done <<< "$discovered"
}

filter_tests() {
    local test_file
    local -a filtered=()

    for test_file in "${TESTS[@]}"; do
        # Failures confirmed with developers on A5.
        if [[ "${NPU_HARDWARE:-}" == 950 ]]; then
            case "$test_file" in
                tests/python/operators/attention/test_compressor.py|\
                tests/python/operators/attention/test_flash_mla.py|\
                tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_dsl.py|\
                tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata_dsl.py|\
                tests/python/operators/indexer/test_indexer_prologue_k_dsl.py|\
                tests/python/operators/indexer/test_quant_lightning_indexer_dsl.py|\
                tests/python/operators/indexer/test_quant_lightning_indexer_metadata_dsl.py|\
                tests/python/operators/indexer/test_quant_sparse_lightning_indexer_dsl.py|\
                tests/python/operators/indexer/test_quant_sparse_lightning_indexer_metadata_dsl.py|\
                tests/python/operators/mem_cache/test_kv_transfer.py|\
                tests/python/operators/norm/test_add_rmsnorm_bias.py|\
                tests/python/operators/sample/test_markov_sample_step_triton.py|\
                tests/python/operators/attention/gdn/test_chunk_gated_delta_rule.py|\
                tests/python/operators/attention/test_kv_quant_sparse_attn_sharedkv.py|\
                tests/python/operators/attention/test_kv_quant_sparse_attn_sharedkv_metadata.py|\
                tests/python/operators/indexer/test_pool_key_indexer.py)
                    echo "[NPU_HARDWARE=950] Skipping: $test_file"
                    continue
                    ;;
            esac
        fi

        if [[ "${NPU_HARDWARE:-}" == a3 ]]; then
            case "$test_file" in
                tests/python/operators/attention/gdn/test_recompute_w_u_fwd.py|\
                tests/python/operators/hyper_conn/test_mhc_post.py|\
                tests/python/operators/hyper_conn/test_mhc_pre_sinkhorn.py|\
                tests/python/operators/attention/test_compressor_v2.py|\
                tests/python/operators/attention/test_kv_quant_sparse_attn_sharedkv.py|\
                tests/python/operators/attention/test_kv_quant_sparse_attn_sharedkv_metadata.py|\
                tests/python/operators/indexer/test_lightning_indexer_v2.py|\
                tests/python/operators/mem_cache/test_kv_transfer.py|\
                tests/python/operators/indexer/test_pool_key_indexer.py|\
                tests/python/operators/indexer/test_qsa_mqa.py)
                    echo "[NPU_HARDWARE=a3] Skipping: $test_file"
                    continue
                    ;;
            esac
        fi

        filtered+=("$test_file")
    done

    TESTS=("${filtered[@]}")
}

run_test() {
    local test_file="$1"
    local exit_code=0

    echo "=========================================="
    echo "Running: $test_file"
    echo "=========================================="

    python3 -m pytest "$test_file" -q -x --enable-perf=false || exit_code=$?

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

filter_tests

if [[ "${#TESTS[@]}" -eq 0 ]]; then
    echo "All test files in group '$TEST_GROUP' were skipped by environment rules."
    exit 0
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

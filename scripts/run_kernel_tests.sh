#!/bin/bash

TEST_DIR="${GITHUB_WORKSPACE}/tests/python/sgl_kernel_npu"
cd "$TEST_DIR" || { echo "Directory not found: $TEST_DIR"; exit 1; }

PASSED=()
FAILED=()

run_test() {
    local test_file="$1"
    echo "=========================================="
    echo "Running: $test_file"
    echo "=========================================="
    if python3 "$test_file"; then
        PASSED+=("$test_file")
        echo "PASSED: $test_file"
    else
        FAILED+=("$test_file")
        echo "FAILED: $test_file"
    fi
    echo ""
}

SMOKE_TESTS=(
    test_hello_world.py
)

NORM_TESTS=(
    test_add_rmsnorm_bias.py
    test_rmsnorm_split.py
    test_rmsnorm_without_weight.py
    test_l1_norm.py
    test_scale_shift.py
)

ATTENTION_TESTS=(
    test_decode_attention.py
    test_mla_preprocess.py
    test_split_qkv_rmsnorm_rope.py
    test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py
    test_split_qkv_tp_rmsnorm_rope.py
)

CACHE_TESTS=(
    test_alloc_extend_slot.py
    test_cache_assign.py
    test_cache_update.py
    test_inplace_assign_cache.py
    test_lightning_indexer.py
    test_transfer_kv_dim_exchange.py
)

SPECULATIVE_TESTS=(
    test_build_tree.py
    test_verify_tree.py
    test_apply_token_bitmask.py
)

MAMBA_TESTS=(
    test_conv1d_prefill.py
    test_conv1d_update.py
    test_mamba_conv.py
    test_mamba_state_update.py
)

FLA_TESTS=(
    test_gated_delta.py
    test_gated_delta_ascendc_tri_inv.py
    test_mega_chunk_gdn.py
    test_recurrent_gated_delta_rule.py
    test_fused_gdn_gating_without_sigmoid.py
    test_solve_tril.py
    test_triangular_inverse.py
)

FUSED_TESTS=(
    test_swiglu_quant.py
    test_batch_matmul_transpose.py
    test_catlass_matmul_basic.py
    test_qkvzba_split_reshape_cat.py
    test_lora_kernels.py
)

ALL_TESTS=(
    "${SMOKE_TESTS[@]}"
    "${NORM_TESTS[@]}"
    "${ATTENTION_TESTS[@]}"
    "${CACHE_TESTS[@]}"
    "${SPECULATIVE_TESTS[@]}"
    "${MAMBA_TESTS[@]}"
    "${FLA_TESTS[@]}"
    "${FUSED_TESTS[@]}"
)

SMALL_BATCH_TESTS=(
    test_hello_world.py
    test_decode_attention.py
    test_mla_preprocess.py
    test_add_rmsnorm_bias.py
    test_split_qkv_rmsnorm_rope.py
    test_alloc_extend_slot.py
    test_cache_assign.py
    test_cache_update.py
)

TEST_GROUP="${1:-small}"

case "$TEST_GROUP" in
    small)
        TESTS=("${SMALL_BATCH_TESTS[@]}")
        ;;
    all)
        TESTS=("${ALL_TESTS[@]}")
        ;;
    smoke)
        TESTS=("${SMOKE_TESTS[@]}")
        ;;
    norm)
        TESTS=("${NORM_TESTS[@]}")
        ;;
    attention)
        TESTS=("${ATTENTION_TESTS[@]}")
        ;;
    cache)
        TESTS=("${CACHE_TESTS[@]}")
        ;;
    speculative)
        TESTS=("${SPECULATIVE_TESTS[@]}")
        ;;
    mamba)
        TESTS=("${MAMBA_TESTS[@]}")
        ;;
    fla)
        TESTS=("${FLA_TESTS[@]}")
        ;;
    fused)
        TESTS=("${FUSED_TESTS[@]}")
        ;;
    *)
        echo "Unknown test group: $TEST_GROUP"
        echo "Available groups: small, all, smoke, norm, attention, cache, speculative, mamba, fla, fused"
        exit 1
        ;;
esac

echo "Running test group: $TEST_GROUP (${#TESTS[@]} tests)"
echo ""

for test in "${TESTS[@]}"; do
    run_test "$test"
done

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total:  ${#TESTS[@]}"
echo "Passed: ${#PASSED[@]}"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED[@]}"; do
        echo "  - $test"
    done
    exit 1
fi

echo ""
echo "All tests passed!"

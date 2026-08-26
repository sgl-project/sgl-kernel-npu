#!/bin/bash

# High precision mode for CANN operators (may help with quantization precision)
export ACL_OP_SELECT_IMPL_MODE=high_precision

TEST_DIR="${GITHUB_WORKSPACE}/tests/python/sgl_kernel_npu"
cd "$TEST_DIR" || { echo "Directory not found: $TEST_DIR"; exit 1; }

PASSED=()
FAILED=()

run_test() {
    local test_file="$1"
    local exit_code=0
    echo "=========================================="
    echo "Running: $test_file"
    echo "=========================================="
    if grep -q '__main__' "$test_file"; then
        # Script-style test with a __main__ entry: run it directly
        python3 "$test_file" || exit_code=$?
    elif grep -q 'def test_' "$test_file"; then
        # Pytest-style test without __main__: collect and run it with pytest.
        # Direct 'python3 file.py' would only import and falsely pass.
        python3 -m pytest "$test_file" -q -x || exit_code=$?
    else
        # Plain script (e.g. test_hello_world.py): run it directly
        python3 "$test_file" || exit_code=$?
    fi
    if [ "$exit_code" -eq 0 ]; then
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
    # test_add_rmsnorm_bias.py  # FAILING: add_rmsnorm_bias() got multiple values for 'norm_bias'
    test_rmsnorm_split.py
    test_rmsnorm_without_weight.py
    test_l1_norm.py
    test_scale_shift.py
)

ATTENTION_TESTS=(
    # test_decode_attention.py  # FAILING: tl.parallel removed in triton 3.5.0
    test_mla_preprocess.py
    test_split_qkv_rmsnorm_rope.py  # fixed by PR#701 (partial rope dim)
    test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py
    test_split_qkv_tp_rmsnorm_rope.py
    test_fused_rope_qk_mqa.py
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
    test_argmax_softmax_prob.py
)

MAMBA_TESTS=(
    # test_conv1d_prefill.py  # FAILING: PyTorch 2.10.0 strict schema check
    test_conv1d_update.py
    test_mamba_conv.py
    # test_mamba_state_update.py  # FAILING: move_intermediate_cache strided-dst exact mismatch
)

FLA_TESTS=(
    test_gated_delta_ascendc_tri_inv.py
    test_chunk_gdn_pto.py
    test_chunk_gdn_triton.py
    test_recurrent_gated_delta_rule.py
    test_fused_gdn_gating_without_sigmoid.py
    test_solve_tril.py
    test_triangular_inverse.py
)

FUSED_TESTS=(
    test_swiglu_quant.py  # fixed by PR#701 (fp32 ref aligned)
    test_batch_matmul_transpose.py
    # test_catlass_matmul_basic.py  # FAILING: flaky float16 precision (0.0078 > 0.0005)
    test_qkvzba_split_reshape_cat.py
    test_lora_kernels.py
    test_gmm_wfp8a16.py
    test_mm_wfp8a16.py
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

# CANN 9.1.0: skip flaky/broken fla tests that only fail on this image.
# - test_triangular_inverse.py: torch_npu builtin op output mismatch (DTS 20260819-#2)
# - test_chunk_gdn_triton.py:    intermittent aicore timeout, error 507014 (DTS 20260819-#3)
# Both still run on CANN 9.0.0 where they pass.
if [[ "${CANN_VERSION:-}" == 9.1* ]]; then
    SKIP_ON_910=(
        test_triangular_inverse.py
        test_chunk_gdn_triton.py
    )
    FILTERED_TESTS=()
    for t in "${TESTS[@]}"; do
        local_skip=0
        for s in "${SKIP_ON_910[@]}"; do
            if [[ "$t" == "$s" ]]; then
                local_skip=1
                break
            fi
        done
        if [ "$local_skip" -eq 0 ]; then
            FILTERED_TESTS+=("$t")
        fi
    done
    echo "[CANN_VERSION=$CANN_VERSION] skipping on 9.1.0: ${SKIP_ON_910[*]}"
    TESTS=("${FILTERED_TESTS[@]}")
fi

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

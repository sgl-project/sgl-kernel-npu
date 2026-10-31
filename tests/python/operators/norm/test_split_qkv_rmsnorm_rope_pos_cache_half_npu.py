"""Packed-cache QKV shapes from SGLang LLaDA2 and MiniMax M3 NPU calls.

LLaDA2.0-mini: Q/KV heads 16/4; MiniMax-M3: 64/4; both D=128, R=64.
models/llada2.py::forward and models/minimax_m3.py::forward_prepare_npu
pass projected [N,Q+2K] tensors plus integer positions and [max_seq,R] cache.
MiniMax supplies gemma_weight=1+weight, so this operator must not add 1 again.
TP uses max(1, global KV heads / TP). Loads are synthetic single-token,
batched/block and prefill cases, including the 1024-token grid transition.
Inspected SGLang revision: 2456c9111a121a02e41e625e3930e1eb143fe9bf.
Model dimensions:
https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/1b8bc9e1106e797d11bb0257b4f177e40db383b2/config.json
https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json (text_config)
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.norm import generate_pos_cache_data
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Q/K are quantized through BF16 even when their storage dtype is FP16/FP32.
# Use the framework's BF16 tolerance for these supplementary output dtypes;
# the output-rounding test below still checks Q/K and the untouched V exactly.
cache_accuracy_standard = {
    "torch.float16": AccuracyStandard(atol=5e-2, rtol=5e-2),
    "torch.float32": AccuracyStandard(atol=5e-2, rtol=5e-2),
}

production_cases = []
perf_shape = []
for model, qh, kh, tp_values in (
    ("llada2-mini", 16, 4, (1, 4, 8)),
    ("minimax-m3", 64, 4, (1, 4, 8, 16)),
):
    for tp in tp_values:
        for tokens in (1, 16, 128, 1024, 2048):
            shape = (tokens, qh // tp, max(1, kh // tp), 128, 64)
            production_cases.append(
                pytest.param(shape, id=f"{model}-tp{tp}-tokens{tokens}")
            )
            perf_shape.append(shape)


def generate_test_data(shape, dtype=torch.bfloat16, **kwargs):
    return generate_pos_cache_data(shape, dtype, **kwargs)


# Required: actual projected widths, KV replication, packed cache and both grid regimes.
@pytest.mark.parametrize("shape", production_cases)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_split_qkv_rmsnorm_rope_pos_cache_production(
    shape,
    cache_dtype,
):
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    executor = check_accuracy(
        split_qkv_rmsnorm_rope_pos_cache_half_npu,
        kwargs=generate_test_data(shape, cache_dtype=cache_dtype),
    )
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape",
    [(0, 2, 1, 64, 32), (1, 1, 1, 2, 2), (7, 3, 1, 64, 16), (33, 4, 2, 256, 128)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("cast_norm_to_bf16", [False, True])
@pytest.mark.parametrize("norm", [False, True])
def test_split_qkv_rmsnorm_rope_pos_cache_generalization(
    shape,
    dtype,
    cast_norm_to_bf16,
    norm,
):
    # Include non-BF16 outputs, no-norm calls, empty rows and both pre-RoPE casts.
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data(
        shape, dtype, norm=norm, bias=norm, cast_norm_to_bf16=cast_norm_to_bf16
    )
    executor = check_accuracy(
        split_qkv_rmsnorm_rope_pos_cache_half_npu,
        kwargs=case,
        standard=cache_accuracy_standard,
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("wide_grid_min_tokens", [1, 4096])
def test_split_qkv_rmsnorm_rope_pos_cache_forced_grid(
    wide_grid_min_tokens,
):
    # Force each launch layout at identical shape and compare to the independent backend.
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data((17, 16, 4, 128, 64))
    case["wide_grid_min_tokens"] = wide_grid_min_tokens
    executor = check_accuracy(split_qkv_rmsnorm_rope_pos_cache_half_npu, kwargs=case)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_split_qkv_rmsnorm_rope_pos_cache_clamping_and_layout(
    positions_dtype,
):
    # Negative/oversized positions clamp on-device; strided position/cache buffers are copied.
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data((5, 2, 1, 64, 32))
    positions = torch.empty(10, dtype=positions_dtype)[::2]
    positions.copy_(torch.tensor([-7, 0, 15, 4095, 4102], dtype=positions_dtype))
    cache = torch.empty(4096, 64, dtype=case["cos_sin_cache"].dtype)[:, ::2]
    cache.copy_(case["cos_sin_cache"])
    case["positions"], case["cos_sin_cache"] = positions, cache
    executor = check_accuracy(split_qkv_rmsnorm_rope_pos_cache_half_npu, kwargs=case)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("cast_norm_to_bf16", [False, True])
def test_split_qkv_rmsnorm_rope_pos_cache_output_rounding(
    cast_norm_to_bf16,
):
    # Q/K, including the non-rotary tail, round through BF16; V keeps its original FP32 bits.
    if get_skip_acc_check() is True:
        pytest.skip("Output checks disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data(
        (3, 2, 1, 64, 32),
        torch.float32,
        norm=False,
        cast_norm_to_bf16=cast_norm_to_bf16,
    )
    source = case["input_tensor"]
    source.fill_(1.00390625)  # Exact halfway BF16 value: RNE rounds to even 1.0.
    case["cos_sin_cache"][:, :16] = 1
    case["cos_sin_cache"][:, 16:] = 0
    snapshot = source.clone()
    impl, _ = get_test_impls()
    require_test_impl(split_qkv_rmsnorm_rope_pos_cache_half_npu, impl)
    q, k, v = split_qkv_rmsnorm_rope_pos_cache_half_npu(**case, impl=impl)
    torch.testing.assert_close(q, torch.ones_like(q), atol=0, rtol=0)
    torch.testing.assert_close(k, torch.ones_like(k), atol=0, rtol=0)
    torch.testing.assert_close(v, source[:, -64:], atol=0, rtol=0)
    for output in (q, k, v):
        assert output.dtype == source.dtype and output.device == source.device
        assert output.is_contiguous()
        assert (
            output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(source, snapshot, atol=0, rtol=0)


@pytest.mark.extra_checks
@pytest.mark.parametrize("cast_norm_to_bf16", [False, True])
def test_split_qkv_rmsnorm_rope_pos_cache_pre_rope_rounding(
    cast_norm_to_bf16,
):
    # A halfway value followed by a 0.75 rotation distinguishes the two cast modes exactly.
    if get_skip_acc_check() is True:
        pytest.skip("Output checks disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data(
        (3, 2, 1, 64, 32),
        torch.float32,
        norm=False,
        cast_norm_to_bf16=cast_norm_to_bf16,
    )
    case["input_tensor"].fill_(1.01171875)
    case["cos_sin_cache"][:, :16] = 0.75
    case["cos_sin_cache"][:, 16:] = 0
    impl, _ = get_test_impls()
    require_test_impl(split_qkv_rmsnorm_rope_pos_cache_half_npu, impl)
    q, k, _ = split_qkv_rmsnorm_rope_pos_cache_half_npu(**case, impl=impl)
    expected = 0.76171875 if cast_norm_to_bf16 else 0.7578125
    for output in (q, k):
        rotated = output.reshape(3, -1, 64)[..., :32]
        torch.testing.assert_close(
            rotated, torch.full_like(rotated, expected), atol=0, rtol=0
        )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input",
    [
        "position_count",
        "position_dtype",
        "position_device",
        "cache_empty",
        "cache_width",
        "head_dim",
        "epsilon",
        "weight_short",
        "bias_pair",
        "stride",
        "threshold",
        "cast_flag",
    ],
)
def test_split_qkv_rmsnorm_rope_pos_cache_invalid_input(
    invalid_input,
):
    # Reject invalid cache metadata and controls without synchronizing position values.
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data((3, 2, 1, 64, 32))
    if invalid_input == "position_count":
        case["positions"] = torch.zeros(2, dtype=torch.int64)
    elif invalid_input == "position_dtype":
        case["positions"] = case["positions"].float()
    elif invalid_input == "position_device":
        case["positions"] = torch.empty(3, dtype=torch.int64, device="meta")
    elif invalid_input == "cache_empty":
        case["cos_sin_cache"] = torch.empty(0, 32)
    elif invalid_input == "cache_width":
        case["cos_sin_cache"] = torch.empty(32, 64)
    elif invalid_input == "head_dim":
        case["head_dim"] = 48
    elif invalid_input == "epsilon":
        case["eps"] = float("inf")
    elif invalid_input == "weight_short":
        case["k_weight"] = torch.ones(63)
    elif invalid_input == "bias_pair":
        case["k_bias"] = torch.zeros(64)
    elif invalid_input == "stride":
        case["input_tensor"] = torch.empty(3, 512)[:, ::2]
    elif invalid_input == "threshold":
        case["wide_grid_min_tokens"] = 0
    elif invalid_input == "cast_flag":
        case["cast_norm_to_bf16"] = 1
    impl, _ = get_test_impls()
    require_test_impl(split_qkv_rmsnorm_rope_pos_cache_half_npu, impl)
    with pytest.raises(AssertionError):
        split_qkv_rmsnorm_rope_pos_cache_half_npu(**case, impl=impl)


@pytest.mark.extra_checks
def test_split_qkv_rmsnorm_rope_pos_cache_graph_replay():
    # Replay must read updated position buffers without host synchronization in the wrapper.
    if get_skip_acc_check() is True:
        pytest.skip("Graph correctness checks disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

    case = generate_test_data((8, 4, 1, 128, 64))
    executor = check_accuracy(split_qkv_rmsnorm_rope_pos_cache_half_npu, kwargs=case)
    for _ in range(3):
        executor()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        captured = executor()
    torch.npu.synchronize()
    case["positions"].add_(3)
    graph.replay()
    torch.npu.synchronize()
    reference = split_qkv_rmsnorm_rope_pos_cache_half_npu(
        **case, impl=executor.impl_ref
    )
    assert_accuracy(captured, reference)

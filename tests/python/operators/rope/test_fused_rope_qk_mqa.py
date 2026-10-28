"""Public-interface accuracy/performance tests for fused Q/K RoPE.

SGLang's RotaryEmbedding.forward_npu selects cache rows before this call
(sglang/srt/layers/rotary_embedding/base.py). Its BF16/FP32-cache fast path
motivates mixed-cache tests. Concrete production shapes have not been traced;
the four legacy shapes below come from the old test, not model attribution.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# (tokens, query_heads, key_heads, head_size, rotary_dim).
# Legacy regression workloads, not verified production traces.
perf_shape = [(32, 8, 1, 64, 32), (64, 32, 1, 128, 64)]


# Shared by the standard accuracy/performance suite and optional contract checks.
rope_cases = [
    pytest.param(
        (32, 8, 1, 64, 32), "random", "contiguous", False, id="legacy-mqa-partial"
    ),
    pytest.param(
        (32, 8, 1, 32, 32), "random", "contiguous", False, id="legacy-mqa-full"
    ),
    pytest.param(
        (17, 16, 1, 128, 64),
        "random",
        "contiguous",
        False,
        id="legacy-mqa-odd-tokens",
    ),
    pytest.param(
        (64, 32, 1, 128, 64), "random", "contiguous", False, id="legacy-mqa-batched"
    ),
    pytest.param(
        (1, 1, 1, 3, 2),
        "random",
        "contiguous",
        True,
        id="edge-minimum-pair-one-tail",
    ),
    pytest.param(
        (3, 8, 2, 96, 64),
        "identity",
        "strided_inputs",
        False,
        id="gqa-identity-strided-features",
    ),
    pytest.param(
        (5, 4, 4, 64, 64),
        "quarter_turn",
        "transposed_heads",
        False,
        id="mha-quarter-turn-transposed-heads",
    ),
    pytest.param(
        (7, 8, 2, 64, 32),
        "zero",
        "strided_all",
        True,
        id="gqa-zero-all-strides-fp32-cache",
    ),
    pytest.param(
        (9, 8, 1, 128, 128),
        "random",
        "strided_cache",
        True,
        id="mqa-full-strided-fp32-cache",
    ),
    pytest.param(
        (3, 4, 4, 96, 64),
        "random",
        "transposed_heads",
        True,
        id="mha-partial-transposed-fp32-cache",
    ),
]


def _make_input(tokens, heads, head_size, dtype, layout):
    if layout == "strided_inputs":
        return torch.randn(tokens, heads, head_size * 2, dtype=dtype)[..., 1::2]
    if layout == "transposed_heads":
        return torch.randn(heads, tokens, head_size, dtype=dtype).transpose(0, 1)
    if layout == "strided_all":
        return torch.randn(tokens * 2, heads * 2, head_size * 2, dtype=dtype)[
            1::2, 1::2, 1::2
        ]
    return torch.randn(tokens, heads, head_size, dtype=dtype)


def generate_test_data(shape, dtype, mode, layout, fp32_cache, seed=42):
    """Generate NPU inputs in conftest's default-device context.

    random: signed Q/K and different angles across tokens and rotary pairs.
    identity: cos=1, sin=0 supplies an exact unchanged-input oracle.
    quarter_turn: cos=0, sin=1 supplies an exact (-second, first) oracle.
    zero: zero Q/K and finite nontrivial angles must produce exact zeros.

    Extra cache rows differ from token rows; only rows [0:T] are consumed.
    Strided cases exercise token, head and/or feature strides without overlap.
    """
    torch.npu.manual_seed(seed)
    tokens, query_heads, key_heads, head_size, rotary_dim = shape
    query = _make_input(tokens, query_heads, head_size, dtype, layout)
    key = _make_input(tokens, key_heads, head_size, dtype, layout)
    # Bound values while retaining nontrivial low-precision rotary products.
    query.mul_(0.25)
    key.mul_(0.25)
    cache_dtype = torch.float32 if fp32_cache else dtype
    cache_rows = tokens + 3
    if layout in ("strided_cache", "strided_all"):
        cos_sin = torch.empty(cache_rows * 2, rotary_dim * 2, dtype=cache_dtype)[
            1::2, 1::2
        ]
    else:
        cos_sin = torch.empty(cache_rows, rotary_dim, dtype=cache_dtype)
    angles = torch.randn(cache_rows, rotary_dim // 2, dtype=torch.float32)
    cos_sin[:, : rotary_dim // 2] = angles.cos().to(cache_dtype)
    cos_sin[:, rotary_dim // 2 :] = angles.sin().to(cache_dtype)
    if mode == "identity":
        cos_sin[:tokens, : rotary_dim // 2] = 1
        cos_sin[:tokens, rotary_dim // 2 :] = 0
    elif mode == "quarter_turn":
        cos_sin[:tokens, : rotary_dim // 2] = 0
        cos_sin[:tokens, rotary_dim // 2 :] = 1
    elif mode == "zero":
        query.zero_()
        key.zero_()
    elif mode != "random":
        raise ValueError(f"Unknown data mode: {mode}")
    return query, key, cos_sin


def _check_output_contract(result, inputs, snapshots, rotary_dim, is_neox_style, mode):
    assert isinstance(result, tuple) and len(result) == 2
    input_storages = [tensor.untyped_storage().data_ptr() for tensor in inputs]
    output_storages = []
    for output, original in zip(result, inputs[:2]):
        assert output.shape == original.shape
        assert output.dtype == original.dtype
        assert output.device == original.device
        pointer = output.untyped_storage().data_ptr()
        assert pointer not in input_storages
        output_storages.append(pointer)
        torch.testing.assert_close(
            output[..., rotary_dim:], original[..., rotary_dim:], atol=0, rtol=0
        )
        if mode in ("identity", "zero"):
            torch.testing.assert_close(output, original, atol=0, rtol=0)
        elif mode == "quarter_turn":
            if is_neox_style:
                first = original[..., : rotary_dim // 2]
                second = original[..., rotary_dim // 2 : rotary_dim]
                rotated = torch.cat((-second, first), dim=-1)
            else:
                first = original[..., :rotary_dim:2]
                second = original[..., 1:rotary_dim:2]
                rotated = torch.stack((-second, first), dim=-1).flatten(-2)
            torch.testing.assert_close(
                output[..., :rotary_dim], rotated, atol=0, rtol=0
            )
    assert output_storages[0] != output_storages[1]
    for original, before in zip(inputs, snapshots):
        torch.testing.assert_close(original, before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox_style", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize("shape,mode,layout,fp32_cache", rope_cases)
def test_fused_rope_qk_mqa(
    shape,
    dtype,
    is_neox_style,
    mode,
    layout,
    fp32_cache,
):
    from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

    query, key, cos_sin = generate_test_data(shape, dtype, mode, layout, fp32_cache)
    inputs = (query, key, cos_sin)
    rotary_dim = shape[-1]
    executor = check_accuracy(
        fused_rope_qk_mqa, args=(*inputs, rotary_dim, is_neox_style)
    )
    enable_perf = (
        shape in perf_shape
        and dtype == torch.float16
        and mode == "random"
        and layout == "contiguous"
        and not fp32_cache
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox_style", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize("shape,mode,layout,fp32_cache", rope_cases)
def test_fused_rope_qk_mqa_output_contract(
    shape,
    dtype,
    is_neox_style,
    mode,
    layout,
    fp32_cache,
):
    """Optional allocation, input-preservation and exact special-case checks."""
    if get_skip_acc_check() is True:
        pytest.skip("Output-contract checks are disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

    inputs = generate_test_data(shape, dtype, mode, layout, fp32_cache)
    snapshots = tuple(tensor.clone() for tensor in inputs)
    rotary_dim = shape[-1]
    impl, _ = get_test_impls()
    require_test_impl(fused_rope_qk_mqa, impl)
    result = fused_rope_qk_mqa(*inputs, rotary_dim, is_neox_style, impl=impl)
    _check_output_contract(result, inputs, snapshots, rotary_dim, is_neox_style, mode)


@pytest.mark.parametrize("is_neox_style", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize(
    "dtype,cache_dtype",
    [
        pytest.param(torch.float16, torch.bfloat16, id="fp16-bf16-cache"),
        pytest.param(torch.bfloat16, torch.float16, id="bf16-fp16-cache"),
    ],
)
def test_fused_rope_qk_mqa_mixed_cache_dtype(
    dtype,
    cache_dtype,
    is_neox_style,
):
    from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

    query, key, cos_sin = generate_test_data(
        (3, 4, 2, 32, 32), dtype, "random", "contiguous", False
    )
    executor = check_accuracy(
        fused_rope_qk_mqa,
        args=(query, key, cos_sin.to(cache_dtype), 32, is_neox_style),
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input",
    [
        "query_rank",
        "key_rank",
        "cache_rank",
        "token_mismatch",
        "head_size_mismatch",
        "empty_tokens",
        "empty_query_heads",
        "empty_key_heads",
        "short_cache",
        "cache_width",
        "query_dtype",
        "key_dtype",
        "cache_dtype",
        "dtype_mismatch",
        "rotary_zero",
        "rotary_odd",
        "rotary_too_large",
        "rotary_float",
        "rotary_bool",
        "style_not_bool",
        "query_heads_not_power_of_two",
        "key_heads_not_power_of_two",
        "rotary_half_not_power_of_two",
        "tail_not_power_of_two",
    ],
)
def test_fused_rope_qk_mqa_invalid_input(invalid_input):
    from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

    query, key, cos_sin = generate_test_data(
        (3, 8, 2, 64, 32), torch.float16, "random", "contiguous", False
    )
    rotary_dim, is_neox_style = 32, True
    if invalid_input == "query_rank":
        query = query.flatten(1)
    elif invalid_input == "key_rank":
        key = key.flatten(1)
    elif invalid_input == "cache_rank":
        cos_sin = cos_sin.unsqueeze(0)
    elif invalid_input == "token_mismatch":
        key = key[:2]
    elif invalid_input == "head_size_mismatch":
        key = key[..., :32]
    elif invalid_input == "empty_tokens":
        query, key = query[:0], key[:0]
    elif invalid_input == "empty_query_heads":
        query = query[:, :0]
    elif invalid_input == "empty_key_heads":
        key = key[:, :0]
    elif invalid_input == "short_cache":
        cos_sin = cos_sin[:2]
    elif invalid_input == "cache_width":
        cos_sin = cos_sin[:, :30]
    elif invalid_input == "query_dtype":
        query, key = query.to(torch.int32), key.to(torch.int32)
    elif invalid_input == "key_dtype":
        key = key.to(torch.int32)
    elif invalid_input == "cache_dtype":
        cos_sin = cos_sin.to(torch.int32)
    elif invalid_input == "dtype_mismatch":
        key = key.to(torch.float32)
    elif invalid_input == "rotary_zero":
        rotary_dim = 0
    elif invalid_input == "rotary_odd":
        rotary_dim = 31
    elif invalid_input == "rotary_too_large":
        rotary_dim = 128
    elif invalid_input == "rotary_float":
        rotary_dim = 32.0
    elif invalid_input == "rotary_bool":
        rotary_dim = True
    elif invalid_input == "style_not_bool":
        is_neox_style = 1
    elif invalid_input == "query_heads_not_power_of_two":
        query = query[:, :3]
    elif invalid_input == "key_heads_not_power_of_two":
        key = query[:, :3]
    elif invalid_input == "rotary_half_not_power_of_two":
        rotary_dim, cos_sin = 12, cos_sin[:, :12]
    elif invalid_input == "tail_not_power_of_two":
        query, key = query[..., :35], key[..., :35]
    else:
        raise AssertionError(f"Unhandled invalid input: {invalid_input}")
    with pytest.raises(AssertionError):
        fused_rope_qk_mqa(query, key, cos_sin, rotary_dim, is_neox_style, impl="triton")

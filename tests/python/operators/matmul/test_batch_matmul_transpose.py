"""Accuracy and performance through the public matmul API."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# SGLang MLA/DSA post-attention: shape=(tokens, local_heads, kv_lora_rank,
# v_head_dim), local_heads=num_attention_heads/effective_attention_tp.
# On A2/A3, MLA uses AscendC when M, B*K or N reaches 65536. DSA
# decode/verify also uses AscendC at smaller shapes and higher attention TP.
# Keep only DeepSeek/GLM, using one or two recent models for identical shapes.
# (Model ID, total attention heads, K, N, representative attention TP sizes.)
MODEL_PROFILES = [
    ("deepseek-v3.1", 128, 512, 128, (1,)),
    ("deepseek-v3.2", 128, 512, 128, (1, 2, 4, 8)),
    ("glm-5.2", 64, 512, 256, (1, 2, 4, 8)),
    ("glm-5.3-flash", 64, 512, 256, (1, 2, 4, 8)),
]
# Representative loads, not captured traces or fixed per-model batch sizes.
# M counts operator input tokens, including flattened verification tokens.
MODEL_TOKEN_COUNTS = (1, 8, 36, 128, 1024)
MODEL_SHAPES = [
    pytest.param(
        (tokens, heads // tp, k, n),
        id=f"{model}-attn-tp{tp}-tokens{tokens}",
    )
    for model, heads, k, n, attention_tp_sizes in MODEL_PROFILES
    for tp in attention_tp_sizes
    for tokens in MODEL_TOKEN_COUNTS
]
# Exercise both real output widths around the M=16 tile boundary.
MODEL_SHAPES += [
    pytest.param(
        (tokens, heads // 8, 512, n),
        id=f"{model}-attn-tp8-tokens{tokens}",
    )
    for model, heads, n in [("deepseek-v3.2", 128, 128), ("glm-5.2", 64, 256)]
    for tokens in (15, 16, 17)
]
ND_SHAPES = [
    *MODEL_SHAPES,
    pytest.param((1, 1, 1, 1), id="minimum"),
    pytest.param((2, 3, 4, 5), id="small"),
    pytest.param((7, 3, 31, 33), id="unaligned-k-n-tail"),
    pytest.param((15, 2, 16, 16), id="below-m-tile"),
    pytest.param((17, 3, 33, 17), id="above-m-k-n-tiles"),
    pytest.param((1024, 1, 16, 16), id="maximum-m"),
]
PERF_SHAPES = [(1, 128, 512, 128), (8, 16, 512, 128), (36, 128, 512, 128)]


def generate_test_data(shape, dtype, mode):
    """Generate inputs for random and exact-zero products."""
    torch.npu.manual_seed(42)
    m, batch, k, n = shape
    a = torch.randn((m, batch, k), dtype=dtype) * 0.1
    b = torch.randn((batch, k, n), dtype=dtype) * 0.1
    if mode == "zero":
        a.zero_()
    return a, b


def run_case(shape, dtype, mode):
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

    a, b = generate_test_data(shape, dtype, mode)
    a_before, b_before = a.clone(), b.clone()
    impl, impl_ref = get_test_impls()
    if impl == "auto":
        # This shape suite includes the AscendC tiling boundaries.
        require_test_impl(batch_matmul_transpose, "ascendc")
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(batch_matmul_transpose, impl)
    else:
        require_distinct_impls(batch_matmul_transpose, impl, impl_ref)
        reference = batch_matmul_transpose(
            a.clone(), b.clone(), is_prefill=False, impl=impl_ref
        )
        output = batch_matmul_transpose(a, b, is_prefill=False, impl=impl)
        m, batch, _, n = shape
        assert output.shape == (m, batch, n)
        assert output.dtype == a.dtype
        assert output.device == a.device
        assert_accuracy(output, reference)
        assert torch.equal(a, a_before)
        assert torch.equal(b, b_before)
        if mode == "zero":
            assert torch.count_nonzero(output) == 0
    executor = make_executor(
        batch_matmul_transpose,
        args=(a, b),
        kwargs={"is_prefill": False},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(
        executor,
        enable_perf=(
            shape in PERF_SHAPES and dtype == torch.bfloat16 and mode == "random"
        ),
    )


@pytest.mark.parametrize("shape", ND_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["random", "zero"])
def test_batch_matmul_transpose_nd(
    shape,
    dtype,
    mode,
):
    run_case(shape, dtype, mode)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "dtype",
        "batch",
        "k",
        "noncontiguous-b",
        "empty",
        "m-limit",
    ],
)
def test_batch_matmul_transpose_rejects_invalid_input(
    invalid,
):
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

    impl, _ = get_test_impls()
    if impl == "auto":
        require_test_impl(batch_matmul_transpose, "ascendc")
    elif impl == "torch_npu":
        pytest.skip("These cases exercise AscendC and torch_native input limits")
    require_test_impl(batch_matmul_transpose, impl)
    a, b = generate_test_data((3, 2, 32, 32), torch.float16, "random")
    if invalid == "rank":
        a = a.flatten()
    elif invalid == "dtype":
        b = b.float()
    elif invalid == "batch":
        b = b[:1]
    elif invalid == "k":
        a = a[..., :16].contiguous()
    elif invalid == "noncontiguous-b":
        b = b.transpose(1, 2)
    elif invalid == "empty":
        a = a[:0]
    elif invalid == "m-limit":
        a = torch.empty((1025, 2, 32), dtype=a.dtype)
    with pytest.raises((ValueError, RuntimeError)):
        batch_matmul_transpose(a, b, is_prefill=False, impl=impl)

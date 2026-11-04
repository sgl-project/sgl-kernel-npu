"""Accuracy, output mutation and performance through the public matmul API."""

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

# DeepSeek MLA post-attention: A=[tokens, local_heads, kv_lora_rank],
# B=[local_heads, kv_lora_rank, v_head_dim]. Source:
# https://github.com/sgl-project/sglang/blob/99fd01047d67bb249641a8e50282a1a5218afc63/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
# DeepSeek-V3 dimensions: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
# Representative token loads, not captured traces. TP=1/8 gives 128/16 heads.
ND_SHAPES = [
    pytest.param((1, 128, 512, 128), id="deepseek-v3-decode-tp1"),
    pytest.param((8, 16, 512, 128), id="deepseek-v3-decode-tp8"),
    pytest.param((36, 128, 512, 128), id="legacy-mla-target"),
    pytest.param((1, 1, 1, 1), id="minimum"),
    pytest.param((2, 3, 4, 5), id="legacy-small"),
    pytest.param((7, 3, 31, 33), id="unaligned-k-n-tail"),
    pytest.param((15, 2, 16, 16), id="below-m-tile"),
    pytest.param((17, 3, 33, 17), id="above-m-k-n-tiles"),
    pytest.param((1024, 1, 16, 16), id="maximum-m"),
]
NZ_SHAPES = [
    pytest.param((1, 1, 16, 16), id="one-nz-block"),
    pytest.param((7, 3, 32, 48), id="nz-m-tail-multiple-n-blocks"),
    pytest.param((8, 16, 512, 128), id="mla-dimensions-nz"),
]
PERF_SHAPES = [(1, 128, 512, 128), (8, 16, 512, 128), (36, 128, 512, 128)]


def generate_test_data(shape, dtype, format_mode, mode):
    """Random products or exact-zero products; pack NZ without custom APIs."""
    torch.npu.manual_seed(42)
    m, batch, k, n = shape
    a = torch.randn((m, batch, k), dtype=dtype) * 0.1
    b = torch.randn((batch, k, n), dtype=dtype) * 0.1
    if mode == "zero":
        a.zero_()  # Nonzero B checks full output overwrite.
    if format_mode == "NZ":
        b = b.reshape(batch, k, n // 16, 16).permute(0, 2, 1, 3).contiguous()
    c = torch.full((m, batch, n), float("nan"), dtype=dtype)
    return a, b, c


def run_case(shape, dtype, format_mode, quant_mode, mode):
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

    a, b, c = generate_test_data(shape, dtype, format_mode, mode)
    a_before, b_before = a.clone(), b.clone()
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(batch_matmul_transpose, impl)
    else:
        require_distinct_impls(batch_matmul_transpose, impl, impl_ref)
        reference = torch.empty_like(c)
        assert (
            batch_matmul_transpose(
                a.clone(), b.clone(), reference, format_mode, quant_mode, impl=impl_ref
            )
            is None
        )
        pointer = c.data_ptr()
        assert (
            batch_matmul_transpose(a, b, c, format_mode, quant_mode, impl=impl) is None
        )
        assert c.data_ptr() == pointer
        # The return value is None: compare the mutated output explicitly.
        assert_accuracy(c, reference)
        assert torch.equal(a, a_before)
        assert torch.equal(b, b_before)
        if mode == "zero":
            assert torch.count_nonzero(c) == 0
    executor = make_executor(
        batch_matmul_transpose,
        args=(a, b, c, format_mode, quant_mode),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    # C is fully overwritten, so no reset/copies enter profiling.
    check_performance(
        executor,
        enable_perf=(
            shape in PERF_SHAPES
            and dtype == torch.bfloat16
            and format_mode is None
            and quant_mode is None
            and mode == "random"
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
    run_case(shape, dtype, None, None, mode)


@pytest.mark.parametrize("shape", NZ_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["random", "zero"])
def test_batch_matmul_transpose_nz(
    shape,
    dtype,
    mode,
):
    run_case(shape, dtype, "NZ", None, mode)


@pytest.mark.parametrize("format_mode", ["ND", "NZ"])
@pytest.mark.parametrize(
    "quant_mode", ["per_channel_symm", "per_channel_asymm", "per_token_symm"]
)
def test_batch_matmul_transpose_modes(
    format_mode,
    quant_mode,
):
    run_case((3, 2, 32, 48), torch.float16, format_mode, quant_mode, "random")


@pytest.mark.parametrize(
    "invalid",
    [
        "format",
        "quant",
        "rank",
        "dtype",
        "batch",
        "k",
        "output-shape",
        "noncontiguous-a",
        "noncontiguous-b",
        "noncontiguous-c",
        "empty",
        "m-limit",
        "nz-inner",
        "nz-k",
        "output-overlaps-a",
        "output-overlaps-b",
    ],
)
def test_batch_matmul_transpose_rejects_invalid_input(
    invalid,
):
    from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

    impl, _ = get_test_impls()
    require_test_impl(batch_matmul_transpose, impl)
    a, b, c = generate_test_data((3, 2, 32, 32), torch.float16, "ND", "random")
    format_mode = quant_mode = None
    if invalid == "format":
        format_mode = "invalid"
    elif invalid == "quant":
        quant_mode = "invalid"
    elif invalid == "rank":
        a = a.flatten()
    elif invalid == "dtype":
        c = c.float()
    elif invalid == "batch":
        b = b[:1]
    elif invalid == "k":
        a = a[..., :16].contiguous()
    elif invalid == "output-shape":
        c = c[..., :16].contiguous()
    elif invalid.startswith("noncontiguous"):
        name = invalid.rsplit("-", 1)[1]
        if name == "a":
            a = torch.empty((3, 2, 64), dtype=a.dtype)[..., ::2]
        elif name == "b":
            b = b.transpose(1, 2)
        else:
            c = torch.empty((3, 2, 64), dtype=c.dtype)[..., ::2]
    elif invalid == "empty":
        a, c = a[:0], c[:0]
    elif invalid == "m-limit":
        a = torch.empty((1025, 2, 32), dtype=a.dtype)
        c = torch.empty_like(a)
    elif invalid == "nz-inner":
        format_mode = "NZ"
        b = torch.empty((2, 4, 32, 8), dtype=b.dtype)
    elif invalid == "nz-k":
        format_mode = "NZ"
        a = a[..., :31].contiguous()
        b = torch.empty((2, 2, 31, 16), dtype=b.dtype)
    elif invalid == "output-overlaps-a":
        c = a
    else:
        b = torch.empty((2, 3, 32), dtype=b.dtype)
        a = a[..., :3].contiguous()
        c = b.view(3, 2, 32)
    with pytest.raises((ValueError, RuntimeError)):
        batch_matmul_transpose(a, b, c, format_mode, quant_mode, impl=impl)

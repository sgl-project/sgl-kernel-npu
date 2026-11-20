"""Quantized GEMM accuracy cases for W8A8, A16W8, MXFP8 and W4A4."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_BF16_EXACT = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}


def generate_int8_data(m, k, n):
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randint(
        -3, 4, (m, k), dtype=torch.int8, device="cpu", generator=generator
    )
    rhs = torch.randint(
        -3, 4, (k, n), dtype=torch.int8, device="cpu", generator=generator
    )
    channel_scale = torch.full((n,), 0.25, dtype=torch.float32)
    token_scale = torch.full((m,), 0.125, dtype=torch.float32)
    return (lhs.npu(), rhs.npu(), channel_scale), dict(
        pertoken_scale=token_scale, output_dtype=torch.bfloat16
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 128, 64), id="w8a8-linear-decode-tile")]
)
def test_quant_matmul_int8_with_pertoken_scale(shape):
    from sgl_kernel_npu.operators.matmul import quant_matmul

    args, kwargs = generate_int8_data(*shape)
    executor = check_accuracy(quant_matmul, args, kwargs=kwargs, standard=_BF16_EXACT)
    check_performance(executor)


def generate_a16w8_data(m, k, n):
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randn((m, k), generator=generator, device="cpu", dtype=torch.float16)
    weights = torch.randint(
        -8, 8, (k, n), generator=generator, device="cpu", dtype=torch.int8
    )
    scales = (torch.arange(n, device="cpu") % 4 + 1).to(torch.float16) / 16
    offsets = (torch.arange(n, device="cpu") % 3 - 1).to(torch.float16)
    return (lhs.npu(), weights.npu(), scales.unsqueeze(0).npu()), dict(
        antiquant_offset=offsets.unsqueeze(0).npu()
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 320, 256), id="a16w8-per-channel-decode")]
)
def test_weight_quant_batchmatmul_int8(shape):
    from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul

    args, kwargs = generate_a16w8_data(*shape)
    executor = check_accuracy(
        weight_quant_batchmatmul,
        args,
        kwargs=kwargs,
        standard={"torch.float16": AccuracyStandard(atol=0.002, rtol=0.002)},
    )
    check_performance(executor)


def generate_mxfp8_data(m, batch, k, n):
    # DeepSeek V4 groups the WoA projection by attention-head groups.
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randint(-3, 4, (m, batch, k), generator=generator, device="cpu")
    rhs = torch.randint(-3, 4, (batch, k, n), generator=generator, device="cpu")
    lhs_bits = torch.randint(
        125, 128, (m, batch, k // 32), generator=generator, device="cpu"
    )
    rhs_bits = torch.randint(
        125, 128, (batch, k // 32, n), generator=generator, device="cpu"
    )
    x1_scale = lhs_bits.to(torch.uint8).reshape(m, batch, k // 64, 2)
    x2_scale = (
        rhs_bits.to(torch.uint8)
        .reshape(batch, k // 64, 2, n)
        .transpose(-1, -2)
        .contiguous()
    )
    return (
        lhs.to(torch.float8_e4m3fn).npu(),
        rhs.to(torch.float8_e4m3fn).npu(),
        torch.bfloat16,
    ), dict(
        x1_scale=x1_scale.npu().view(torch.float8_e8m0fnu),
        x2_scale=x2_scale.npu().view(torch.float8_e8m0fnu),
        group_sizes=(0, 0, 32),
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 2, 512, 128), id="deepseek-v4-mxfp8-wo-a")]
)
def test_transpose_quant_batchmatmul_mxfp8(shape):
    from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul

    require_test_impl(transpose_quant_batchmatmul, get_test_impls()[0])
    args, kwargs = generate_mxfp8_data(*shape)
    executor = check_accuracy(
        transpose_quant_batchmatmul, args, kwargs=kwargs, standard=_BF16_EXACT
    )
    check_performance(executor)


@pytest.fixture
def allow_internal_weight_format():
    import torch_npu

    option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT").decode()
    try:
        torch_npu.npu.config.allow_internal_format = True
        yield
    finally:
        torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": option})


def generate_w4a4_data(m, k, n):
    import torch_npu

    # W4A4 decode uses a packed ND activation and a transposed NZ weight.
    # The bytes encode exact FP4 values with both dequantization scales one.
    x1 = torch.tensor([0x57, 0xDF], dtype=torch.uint8).repeat(m, k // 4)
    rhs = torch.tensor([0xD7, 0xF5], dtype=torch.uint8).repeat(n, k // 4)
    rhs[1::2] ^= 0x88
    x2 = torch_npu.npu_format_cast(rhs.view(torch.int8), 29, customize_dtype=torch.int8)
    return (
        x1,
        x2,
        torch.ones((m, k // 512), dtype=torch.float32),
        torch.ones((k // 512, n), dtype=torch.float32),
        torch.full((m, k // 64, 2), 127, dtype=torch.uint8),
        torch.full((n, k // 64, 2), 127, dtype=torch.uint8),
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 4096, 128), id="qwen3-w4a4-linear-decode-tile")]
)
def test_dual_level_quant_matmul_w4a4(shape, request):
    from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul

    require_test_impl(dual_level_quant_matmul, get_test_impls()[0])
    request.getfixturevalue("allow_internal_weight_format")
    executor = check_accuracy(
        dual_level_quant_matmul,
        generate_w4a4_data(*shape),
        kwargs={"output_dtype": torch.bfloat16},
        standard=_BF16_EXACT,
    )
    check_performance(executor)

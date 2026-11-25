"""Check quantized GEMMs at SGLang model-layer shapes."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_BF16_EXACT = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}


@pytest.fixture
def allow_internal_weight_format():
    import torch_npu

    option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT").decode()
    try:
        torch_npu.npu.config.allow_internal_format = True
        yield
    finally:
        torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": option})


def generate_int8_data(m, k, n):
    import torch_npu

    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randint(
        -3, 4, (m, k), dtype=torch.int8, device="cpu", generator=generator
    )
    # SGLang transposes the checkpoint weight and casts it to FRACTAL_NZ.
    rhs = torch.randint(
        -3, 4, (n, k), dtype=torch.int8, device="cpu", generator=generator
    ).T.contiguous()
    weights = torch_npu.npu_format_cast(rhs.npu(), 29)
    channel_scale = (torch.arange(n, device="cpu") % 4 + 1).float() / 16
    token_scale = 2.0 ** -(torch.arange(m, device="cpu") + 2)
    return (lhs.npu(), weights, channel_scale.npu()), dict(
        pertoken_scale=token_scale.npu(), bias=None, output_dtype=torch.bfloat16
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 1536, 1536), id="qwen2-5-1p5b-w8a8-o-proj-tp1-decode")]
)
def test_quant_matmul_int8_with_pertoken_scale(shape, allow_internal_weight_format):
    from sgl_kernel_npu.operators.matmul import quant_matmul

    args, kwargs = generate_int8_data(*shape)
    executor = check_accuracy(quant_matmul, args, kwargs=kwargs, standard=_BF16_EXACT)
    check_performance(executor)


def generate_gptq_int8_data(m, k, n, group_size):
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = (
        torch.randint(-8, 9, (m, k), generator=generator, device="cpu").to(
            torch.float16
        )
        / 8
    )
    weights = torch.randint(
        -8, 8, (k, n), generator=generator, device="cpu", dtype=torch.int8
    )
    groups = k // group_size
    channels = torch.arange(n, device="cpu").view(1, -1)
    k_groups = torch.arange(groups, device="cpu").view(-1, 1)
    scales = ((channels + k_groups) % 4 + 1).to(torch.float16) / 16
    # The symmetric GPTQ checkpoint stores a zero point that becomes zero
    # after Ascend unpacks and recenters it.
    offsets = torch.zeros_like(scales)
    return (lhs.npu(), weights.npu(), scales.npu()), dict(
        antiquant_offset=offsets.npu(),
        antiquant_group_size=group_size,
        bias=None,
    )


@pytest.mark.parametrize(
    "shape",
    [pytest.param((2, 2048, 2048), id="qwen3-1p7b-gptq-int8-o-proj-tp1-decode")],
)
def test_weight_quant_batchmatmul_int8(shape):
    from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul

    args, kwargs = generate_gptq_int8_data(*shape, group_size=128)
    executor = check_accuracy(
        weight_quant_batchmatmul,
        args,
        kwargs=kwargs,
        standard={"torch.float16": AccuracyStandard(atol=0.002, rtol=0.002)},
    )
    check_performance(executor)


def generate_mxfp8_data(m, groups, k, n):
    # WoA batches two attention groups on each TP4 rank.
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randint(
        -3, 4, (m, groups, k), dtype=torch.int8, generator=generator, device="cpu"
    )
    rhs = torch.randint(
        -3, 4, (groups, k, n), dtype=torch.int8, generator=generator, device="cpu"
    )
    x1_scale = torch.randint(
        125,
        128,
        (m, groups, k // 64, 2),
        dtype=torch.uint8,
        generator=generator,
        device="cpu",
    )
    # The checkpoint has one E8M0 scale per 128x128 block. SGLang expands
    # it over output channels and 32-wide MX groups, then pairs K scales.
    weight_block_size = 128
    x2_scale = torch.randint(
        125,
        128,
        (groups, k // weight_block_size, n // weight_block_size),
        dtype=torch.uint8,
        generator=generator,
        device="cpu",
    )
    x2_scale = (
        x2_scale.repeat_interleave(weight_block_size // 32, dim=1)
        .repeat_interleave(weight_block_size, dim=2)
        .unflatten(1, (-1, 2))
        .transpose(-1, -2)
        .contiguous()
    )
    return (
        lhs.to(torch.float8_e4m3fn).npu(),
        rhs.to(torch.float8_e4m3fn).npu(),
        torch.bfloat16,
    ), dict(
        bias=None,
        x1_scale=x1_scale.npu().view(torch.float8_e8m0fnu),
        x2_scale=x2_scale.npu().view(torch.float8_e8m0fnu),
        group_sizes=(0, 0, 32),
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )


@pytest.mark.parametrize(
    "shape",
    [pytest.param((2, 2, 4096, 1024), id="deepseek-v4-flash-0731-wo-a-attn-tp4")],
)
def test_transpose_quant_batchmatmul_mxfp8(shape):
    from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul

    require_test_impl(transpose_quant_batchmatmul, get_test_impls()[0])
    args, kwargs = generate_mxfp8_data(*shape)
    executor = check_accuracy(
        transpose_quant_batchmatmul, args, kwargs=kwargs, standard=_BF16_EXACT
    )
    check_performance(executor)


def generate_w4a4_data(m, k, n):
    import torch_npu

    generator = torch.Generator(device="cpu").manual_seed(7)
    # Packed E2M1 activation [M, K/2] and untransposed NZ weight [N, K/2].
    x1 = torch.randint(
        0, 256, (m, k // 2), dtype=torch.uint8, generator=generator, device="cpu"
    )
    rhs = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, generator=generator, device="cpu"
    )
    x2 = torch_npu.npu_format_cast(
        rhs.view(torch.int8).npu(), 29, customize_dtype=torch.int8
    )
    # Powers of two distinguish both scale levels while keeping the reference exact.
    l0 = 2.0 ** -torch.randint(0, 2, (m, k // 512), generator=generator, device="cpu")
    w_l0 = 2.0 ** -torch.randint(0, 2, (k // 512, n), generator=generator, device="cpu")
    l1 = torch.randint(
        126, 128, (m, k // 64, 2), dtype=torch.uint8, generator=generator, device="cpu"
    )
    w_l1 = torch.randint(
        126, 128, (n, k // 64, 2), dtype=torch.uint8, generator=generator, device="cpu"
    )
    return x1.npu(), x2, l0.npu(), w_l0.npu(), l1.npu(), w_l1.npu()


@pytest.mark.parametrize(
    "shape",
    [pytest.param((2, 1024, 4096), id="qwen3-8b-online-w4a4-o-proj-tp4-decode")],
)
def test_dual_level_quant_matmul_w4a4(shape, request):
    from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul

    require_test_impl(dual_level_quant_matmul, get_test_impls()[0])
    request.getfixturevalue("allow_internal_weight_format")
    executor = check_accuracy(
        dual_level_quant_matmul,
        generate_w4a4_data(*shape),
        kwargs={"bias": None, "output_dtype": torch.bfloat16},
        standard=_BF16_EXACT,
    )
    check_performance(executor)

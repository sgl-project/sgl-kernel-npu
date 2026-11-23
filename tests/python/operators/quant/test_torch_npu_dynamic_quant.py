"""Compare dynamic quantizers at SGLang model-layer shapes."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_FP8_EXACT = {"torch.float8_e4m3fn": AccuracyStandard(atol=0, rtol=0)}


def generate_int8_data(shape):
    tokens, width = shape
    values = torch.arange(tokens * width, device="cpu").remainder(255).sub(127)
    # Power-of-two row scales keep the INT8 cast exact and exercise per-token scales.
    scales = 2.0 ** -torch.arange(tokens, device="cpu").view(-1, 1)
    return (values.reshape(shape) * scales).to(torch.bfloat16).npu()


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 1536), id="qwen2-5-1p5b-w8a8-o-proj-tp1-decode")]
)
def test_dynamic_quant_int8(shape):
    from sgl_kernel_npu.operators.quant import dynamic_quant

    # RedHatAI/Qwen2.5-1.5B-quantized.w8a8 uses per-token INT8 activations.
    executor = check_accuracy(dynamic_quant, (generate_int8_data(shape),))
    check_performance(executor)


def generate_block_data(shape):
    rows, width = shape
    payload = torch.tensor([448, -64, 32, 0], dtype=torch.float32, device="cpu")
    scales = 2.0 ** -(torch.arange(rows, device="cpu") % 4).view(-1, 1)
    return (payload.repeat(rows, width // 4) * scales).to(torch.bfloat16).npu()


@pytest.mark.parametrize(
    "shape", [pytest.param((2 * 64, 128), id="deepseek-v4-flash-0731-c4-indexer-q")]
)
def test_dynamic_block_quant(shape):
    from sgl_kernel_npu.operators.quant import dynamic_block_quant
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH35:
        pytest.skip("DeepSeek-V4-Flash C4 indexer block FP8 requires Ascend 950")

    # The Ascend C4 indexer flattens 64 replicated heads per token into rows.
    executor = check_accuracy(
        dynamic_block_quant,
        (generate_block_data(shape),),
        kwargs={"dst_type": torch.float8_e4m3fn},
        standard=_FP8_EXACT,
    )
    check_performance(executor)


def generate_mx_data(shape):
    groups = shape[-1] // 32
    exponents = torch.arange(shape[0] * shape[1] * groups, device="cpu").reshape(
        *shape[:-1], groups
    )
    scale = 2.0 ** (exponents.remainder(4) - 8)
    values = torch.tensor([256, -128, 64, 0], dtype=torch.float32, device="cpu")
    return (
        (scale.unsqueeze(-1) * values.repeat(8)).reshape(shape).to(torch.bfloat16).npu()
    )


@pytest.mark.parametrize(
    "shape",
    [pytest.param((2, 2, 4096), id="deepseek-v4-flash-0731-wo-a-attn-tp4")],
)
def test_dynamic_mx_quant_fp8(shape):
    from sgl_kernel_npu.operators.quant import dynamic_mx_quant

    require_test_impl(dynamic_mx_quant, get_test_impls()[0])
    # Attention TP4 leaves 8 / 4 local groups, each with 64 * 512 / 8 features.
    executor = check_accuracy(
        dynamic_mx_quant,
        (generate_mx_data(shape),),
        kwargs={"dst_type": torch.float8_e4m3fn},
        standard=_FP8_EXACT,
    )
    check_performance(executor)


def generate_dual_level_data(shape):
    rows, width = shape
    outer = 2.0 ** (torch.arange(rows * (width // 512), device="cpu") % 4 - 2)
    outer = outer.reshape(rows, -1)
    inner = 2.0 ** -(torch.arange(width // 32, device="cpu") % 2)
    payload = torch.tensor([6, 3, -6, -3], dtype=torch.float32, device="cpu").repeat(8)
    groups = (outer.repeat_interleave(16, dim=-1) * inner).unsqueeze(-1) * payload
    return groups.reshape(shape).to(torch.bfloat16).npu()


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 1024), id="qwen3-8b-online-w4a4-o-proj-tp4-decode")]
)
def test_dynamic_dual_level_mx_quant_fp4(shape):
    from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant

    require_test_impl(dynamic_dual_level_mx_quant, get_test_impls()[0])
    # Qwen3-8B's row-parallel o_proj receives 32 * 128 / 4 features on each rank.
    executor = check_accuracy(
        dynamic_dual_level_mx_quant,
        (generate_dual_level_data(shape),),
        kwargs={"smooth_scale": None},
    )
    check_performance(executor)

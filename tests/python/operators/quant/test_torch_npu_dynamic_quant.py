"""Compare the dynamic quantizers with independent PyTorch references."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_FP8_EXACT = {"torch.float8_e4m3fn": AccuracyStandard(atol=0, rtol=0)}


def generate_int8_data(width):
    # A row maximum of 127 gives scale 1 and avoids cast midpoint ties.
    x = torch.arange(width, device="cpu").remainder(255).sub(127)
    return x.reshape(1, width).to(device="npu", dtype=torch.bfloat16)


@pytest.mark.parametrize("width", [pytest.param(4096, id="w8a8-linear-decode")])
def test_dynamic_quant_int8(width):
    from sgl_kernel_npu.operators.quant import dynamic_quant

    executor = check_accuracy(
        dynamic_quant, (generate_int8_data(width),), kwargs={"dst_type": torch.int8}
    )
    check_performance(executor)


def generate_block_data(shape, dtype):
    maximum = 127 if dtype == torch.int8 else 448
    payload = torch.tensor([maximum, -64, 32, 0], dtype=torch.float32, device="cpu")
    scale = torch.tensor([0.125, 0.25, 0.5, 1.0], device="cpu").view(-1, 1)
    return (payload.repeat(shape[0], shape[1] // 4) * scale).to(torch.bfloat16).npu()


@pytest.mark.parametrize("shape", [pytest.param((4, 128), id="dsa-indexer-head")])
def test_dynamic_block_quant(shape):
    from sgl_kernel_npu.operators.quant import dynamic_block_quant
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    # Atlas A2/A3 can run block INT8. The DSA FP8 variant requires Ascend 950.
    dtype = (
        torch.float8_e4m3fn
        if get_hardware().generation is Generation.ARCH35
        else torch.int8
    )
    executor = check_accuracy(
        dynamic_block_quant,
        (generate_block_data(shape, dtype),),
        kwargs={"dst_type": dtype, "row_block_size": 1, "col_block_size": 128},
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
    "shape", [pytest.param((2, 2, 512), id="deepseek-v4-wo-a-mxfp8")]
)
def test_dynamic_mx_quant_fp8(shape):
    from sgl_kernel_npu.operators.quant import dynamic_mx_quant

    require_test_impl(dynamic_mx_quant, get_test_impls()[0])
    executor = check_accuracy(
        dynamic_mx_quant,
        (generate_mx_data(shape),),
        kwargs={"dst_type": torch.float8_e4m3fn},
        standard=_FP8_EXACT,
    )
    check_performance(executor)


def generate_dual_level_data(shape):
    rows, width = shape
    outer = 2.0 ** (torch.arange(width // 512, device="cpu") % 4 - 2)
    inner = 2.0 ** -(torch.arange(width // 32, device="cpu") % 2)
    payload = torch.tensor([6, 3, -6, -3], dtype=torch.float32, device="cpu").repeat(8)
    groups = (outer.repeat_interleave(16) * inner).unsqueeze(-1) * payload
    return (
        groups.reshape(1, width).expand(rows, -1).to(torch.bfloat16).contiguous().npu()
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 4096), id="qwen3-w4a4-linear-decode")]
)
def test_dynamic_dual_level_mx_quant_fp4(shape):
    from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant

    require_test_impl(dynamic_dual_level_mx_quant, get_test_impls()[0])
    executor = check_accuracy(
        dynamic_dual_level_mx_quant, (generate_dual_level_data(shape),)
    )
    check_performance(executor)

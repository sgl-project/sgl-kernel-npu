import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.activation.swiglu_mxfp8_quant import swiglu_quant


def _reference(x: torch.Tensor, do_limit: bool, limit: float) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    if do_limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (F.silu(gate) * up).to(x.dtype)


def _dequantize_mxfp8(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = payload.shape
    return (
        payload.float()
        .reshape(rows, cols // 32, 32)
        .mul(scale.float().unsqueeze(-1))
        .reshape_as(payload)
    )


@pytest.mark.parametrize("group_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("do_limit", [False, True])
def test_swiglu_mxfp8_quant_unquantized(group_dtype, do_limit):
    torch.manual_seed(0)
    x = torch.randn(64, 3072, dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([8, 0, 5, 0, 7], dtype=group_dtype, device="npu")

    actual, _ = swiglu_quant(
        x,
        group_list,
        group_list_type=1,
        need_quant=False,
        do_limit=do_limit,
        limit=7.0,
    )

    expected = _reference(x, do_limit, 7.0)
    torch.testing.assert_close(actual[:20], expected[:20], rtol=2e-2, atol=2e-2)


def test_swiglu_mxfp8_quant_unquantized_cumulative_group_list():
    torch.manual_seed(2)
    x = torch.randn(64, 3072, dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([0, 8, 8, 13, 13, 20], dtype=torch.int64, device="npu")

    actual, _ = swiglu_quant(
        x,
        group_list,
        group_list_type=0,
        need_quant=False,
        do_limit=True,
        limit=7.0,
    )

    expected = _reference(x, True, 7.0)
    torch.testing.assert_close(actual[:20], expected[:20], rtol=2e-2, atol=2e-2)


def test_swiglu_mxfp8_quant():
    torch.manual_seed(1)
    x = torch.randn(64, 3072, dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([8, 0, 5, 0, 7], dtype=torch.int64, device="npu")

    payload, scale = swiglu_quant(
        x,
        group_list,
        group_list_type=1,
        need_quant=True,
        do_limit=True,
        limit=7.0,
    )

    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (64, 48)

    actual = _dequantize_mxfp8(payload, scale)
    expected = _reference(x, True, 7.0)
    torch.testing.assert_close(actual[:20], expected[:20], rtol=0.05, atol=0.1)


def test_swiglu_mxfp8_quant_rounds_scale_up_at_saturation_boundary():
    x = torch.full((1, 3072), 7.0, dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([0, 1], dtype=torch.int64, device="npu")

    payload, scale = swiglu_quant(
        x,
        group_list,
        group_list_type=0,
        need_quant=True,
        do_limit=True,
        limit=7.0,
    )

    expected = _reference(x, True, 7.0)
    actual = _dequantize_mxfp8(payload, scale)
    torch.testing.assert_close(scale.float(), torch.full_like(scale.float(), 0.125))
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.25)

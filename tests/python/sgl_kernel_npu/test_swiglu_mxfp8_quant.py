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

    actual = payload.float() * scale.float().unsqueeze(-1)
    expected = _reference(x, True, 7.0)
    torch.testing.assert_close(actual[:20], expected[:20], rtol=0.15, atol=0.2)


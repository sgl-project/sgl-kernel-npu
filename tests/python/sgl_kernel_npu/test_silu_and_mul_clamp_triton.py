import pytest
import torch
import torch.nn.functional as F

from sgl_kernel_npu.activation.silu_and_mul_clamp_triton import silu_and_mul_clamp_triton


def _reference(gate_up, limit, weights):
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit is not None and limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    output = F.silu(gate) * up
    if weights is not None:
        output = output * weights.reshape(-1, 1).float()
    return output.to(gate_up.dtype)


@pytest.mark.parametrize("shape", [(1, 256), (7, 512), (33, 2048)])
@pytest.mark.parametrize("limit", [None, 7.0])
@pytest.mark.parametrize("with_weights", [False, True])
def test_silu_and_mul_clamp_triton(shape, limit, with_weights):
    torch.manual_seed(0)
    gate_up = torch.randn(*shape, dtype=torch.bfloat16, device="npu")
    weights = torch.rand(shape[0], dtype=torch.bfloat16, device="npu") if with_weights else None
    actual = silu_and_mul_clamp_triton(gate_up, swiglu_limit=limit, weights=weights)
    expected = _reference(gate_up, limit, weights)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

import pytest
import torch
import torch_npu  # noqa: F401

from sgl_kernel_npu.activation.situ import situ_and_mul


def _reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    return (gate * up).to(x.dtype)


@pytest.mark.parametrize(
    ("tokens", "intermediate"),
    [(1, 3072), (16, 3072), (1, 33792), (16, 33792), (64, 33792)],
)
@torch.no_grad()
def test_situ_dense_tile_matches_reference(tokens: int, intermediate: int):
    torch.manual_seed(20260820 + tokens)
    host = torch.randn(tokens, 2 * intermediate, dtype=torch.float32).clamp_(-8, 8)
    x = host.to(dtype=torch.bfloat16, device="npu")

    actual = situ_and_mul(x, beta=4.0, linear_beta=25.0)
    expected = _reference(host.to(torch.bfloat16))

    torch.testing.assert_close(actual.cpu(), expected, atol=3e-2, rtol=1e-2)

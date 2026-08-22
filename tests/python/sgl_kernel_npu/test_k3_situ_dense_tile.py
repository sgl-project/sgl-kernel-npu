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


@pytest.mark.parametrize("tokens", [1, 3, 16, 49])
@torch.no_grad()
def test_situ_grouped_hidden_tiles_match_reference(tokens: int):
    intermediate = 33792
    padded_rows = 64
    torch.manual_seed(20260821 + tokens)
    host = torch.randn(
        padded_rows, 2 * intermediate, dtype=torch.float32
    ).clamp_(-8, 8)
    x = host.to(dtype=torch.bfloat16, device="npu")
    counts = torch.zeros(64, dtype=torch.int32, device="npu")
    counts[0] = tokens

    actual = situ_and_mul(
        x,
        group_list=counts,
        group_list_type=1,
        beta=4.0,
        linear_beta=25.0,
    )
    expected = _reference(host[:tokens].to(torch.bfloat16))

    torch.testing.assert_close(
        actual[:tokens].cpu(), expected, atol=3e-2, rtol=1e-2
    )

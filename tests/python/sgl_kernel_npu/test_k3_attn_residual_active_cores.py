import pytest
import torch
import torch_npu  # noqa: F401

from sgl_kernel_npu.kimi_k3.attn_residual import mix_fused


def _reference(prefix, bank, num_valid_blocks, combined_weight, eps):
    values = torch.cat(
        [bank[:, :num_valid_blocks].float(), prefix[:, None].float()], dim=1
    )
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + eps)
    scores = (values * inverse_rms[..., None] * combined_weight.float()).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=1)
    return (probabilities[..., None] * values).sum(dim=1).to(prefix.dtype)


@pytest.mark.parametrize(
    "tokens,num_valid_blocks", [(1, 1), (12, 4), (16, 8), (49, 8)]
)
@torch.no_grad()
def test_mix_fused_small_token_grid_matches_reference(tokens, num_valid_blocks):
    hidden = 7168
    torch.manual_seed(101 + tokens)
    prefix = (torch.randn(tokens, hidden) * 0.25).to(
        dtype=torch.bfloat16, device="npu"
    )
    bank = (torch.randn(tokens, 8, hidden) * 0.25).to(
        dtype=torch.bfloat16, device="npu"
    )
    weight = torch.randn(hidden, device="npu", dtype=torch.float32) / hidden**0.5

    actual = mix_fused(prefix, bank, num_valid_blocks, weight, 1e-6)
    expected = _reference(prefix, bank, num_valid_blocks, weight, 1e-6)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=4e-3, rtol=1e-2)

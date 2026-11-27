import pytest
import sgl_kernel_npu  # noqa: F401  registers npu ops before pytestmark
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.ops.npu namespace available
from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _recurrent_reference(q, k, v, beta, g, scale):
    """Gated delta rule: decay -> predict -> subtractive correction."""
    _, t, H, d = q.shape
    S = torch.zeros(H, d, d, device=q.device)
    o_ref = torch.zeros_like(v, dtype=torch.float32)
    ge, be = g[0].exp(), beta[0].float()
    for ti in range(t):
        S = S * ge[ti].view(H, 1, 1)
        pred = torch.einsum("hd,hde->he", k[0, ti].float(), S)
        u = be[ti].view(H, 1) * (v[0, ti].float() - pred)
        S = S + torch.einsum("hd,he->hde", k[0, ti].float(), u)
        o_ref[0, ti] = torch.einsum(
            "hd,hde->he", q[0, ti].float() * scale, S
        )
    return o_ref, S


def _rel_error(actual, expected):
    diff = (actual.float() - expected.float()).flatten()
    base = expected.float().flatten()
    return (diff.norm() / base.norm().clamp_min(1e-8)).item()


@pytest.mark.parametrize("t,H,d", [(512, 8, 128), (64, 4, 128)])
def test_chunk_gated_delta_rule_npu_fixed_length(t, H, d):
    """cu_seqlens=None selects the fixed-length kernel path.

    Regression for the NameError('i_t is not defined') this path used to hit
    at compile time: the varlen branch assigns i_t from chunk_indices, the
    fixed-length branch must take it from the program id.
    """
    torch.manual_seed(0)
    device = "npu"
    q = F.normalize(torch.rand((1, t, H, d), dtype=torch.bfloat16, device=device), dim=-1)
    k = F.normalize(torch.rand((1, t, H, d), dtype=torch.bfloat16, device=device), dim=-1)
    v = torch.rand((1, t, H, d), dtype=torch.bfloat16, device=device)
    beta = torch.rand((1, t, H), dtype=torch.bfloat16, device=device)
    g = -1 + torch.rand((1, t, H), dtype=torch.float32, device=device)
    scale = d**-0.5

    o, final_state, _ = chunk_gated_delta_rule_npu(
        q, k, v, g, beta, scale=scale, output_final_state=True
    )

    o_ref, S = _recurrent_reference(q, k, v, beta, g, scale)
    assert _rel_error(o, o_ref) < 0.02, f"output rel error {_rel_error(o, o_ref):.4f}"
    assert (
        _rel_error(final_state, S) < 0.02
    ), f"final state rel error {_rel_error(final_state, S):.4f}"

import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla.kda_chunk_delta_h import (
    chunk_gated_delta_rule_fwd_affine_npu,
    merge_kda_cp_affine_states,
)

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="KDA CP affine kernels require an NPU"
)


@requires_npu
@pytest.mark.parametrize("key_dim", [64, 128, 192, 256])
def test_kda_cp_affine_preprocess_identity(key_dim):
    device = torch.device("npu")
    batch, tokens, heads, value_dim = 1, 128, 2, 128
    k = torch.zeros(
        (batch, tokens, heads, key_dim), dtype=torch.bfloat16, device=device
    )
    w = torch.zeros_like(k)
    u = torch.zeros(
        (batch, tokens, heads, value_dim), dtype=torch.bfloat16, device=device
    )
    g = torch.zeros((batch, tokens, heads, key_dim), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)

    affine = chunk_gated_delta_rule_fwd_affine_npu(k, w, u, g, cu_seqlens)
    torch.npu.synchronize()

    torch.testing.assert_close(
        affine[..., :value_dim], torch.zeros_like(affine[..., :value_dim])
    )
    expected_transition = (
        torch.eye(key_dim, dtype=torch.float32, device=device)
        .view(1, 1, key_dim, key_dim)
        .expand_as(affine[..., value_dim:])
    )
    torch.testing.assert_close(affine[..., value_dim:], expected_transition)


@requires_npu
def test_kda_cp_fused_merge_identity_plan():
    device = torch.device("npu")
    cp_size, max_segments = 2, 2
    heads, key_dim, value_dim = 2, 128, 128
    gathered = torch.zeros(
        (
            cp_size,
            max_segments,
            heads,
            key_dim,
            value_dim + key_dim,
        ),
        dtype=torch.float32,
        device=device,
    )
    gathered[..., value_dim:].copy_(
        torch.eye(key_dim, dtype=torch.float32, device=device).view(
            1, 1, 1, key_dim, key_dim
        )
    )
    initial = torch.randn(
        (1, heads, key_dim, value_dim), dtype=torch.float32, device=device
    )
    local_initial = torch.empty(
        (2, heads, key_dim, value_dim), dtype=torch.float32, device=device
    )
    final = torch.empty_like(initial)

    merge_kda_cp_affine_states(
        gathered,
        initial,
        local_initial,
        final,
        cp_rank=0,
        owner_ranks=torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=device),
        source_segments=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device),
        local_indices=torch.tensor([0, -1, -1, 1], dtype=torch.int32, device=device),
        local_steps=(0, 3),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(final, initial)
    torch.testing.assert_close(local_initial[0], initial[0])
    torch.testing.assert_close(local_initial[1], initial[0])


def test_kda_recompute_preserves_public_return_contract(monkeypatch):
    from sgl_kernel_npu.fla import kda_prefill

    expected = tuple(torch.empty(0) for _ in range(3))

    def fake_head_major(**kwargs):
        assert kwargs["beta"].shape == (1, 2, 4)
        return expected

    monkeypatch.setattr(kda_prefill, "recompute_w_u_fwd_head_major", fake_head_major)
    result = kda_prefill.recompute_w_u_fwd_npu(
        k=torch.empty(1, 4, 2, 8),
        v=torch.empty(1, 4, 2, 8),
        beta=torch.empty(1, 4, 2),
        A=torch.empty(1, 4, 2, 4),
        gk=torch.empty(1, 4, 2, 8),
    )

    assert result == expected

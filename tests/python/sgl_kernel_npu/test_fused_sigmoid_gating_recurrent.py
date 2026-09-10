import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  registers the torch.npu backend
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update_npu,
)


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _reference(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    q,
    k,
    v,
    b,
    state,
    state_indices,
    scale,
):
    batch_size, num_tokens, num_query_heads, key_dim = k.shape
    num_value_heads, value_dim = v.shape[2:]
    output = torch.empty_like(v, dtype=torch.float32)
    updated_state = state.float().clone()

    for batch in range(batch_size):
        slot = int(state_indices[batch])
        for token in range(num_tokens):
            for value_head in range(num_value_heads):
                query_head = value_head // (num_value_heads // num_query_heads)
                hidden = (
                    torch.zeros(key_dim, value_dim, dtype=torch.float32)
                    if slot < 0
                    else updated_state[slot, value_head].transpose(0, 1).clone()
                )
                decay = -torch.exp(A_log[value_head]) * F.softplus(
                    a[batch, token, value_head] + dt_bias[value_head],
                    beta=softplus_beta,
                )
                hidden *= torch.exp(decay)
                value = v[batch, token, value_head].float().clone()
                value -= torch.sum(
                    hidden * k[batch, token, query_head].float()[:, None], dim=0
                )
                value *= torch.sigmoid(b[batch, token, value_head].float())
                hidden += k[batch, token, query_head].float()[:, None] * value[None, :]
                output[batch, token, value_head] = torch.sum(
                    hidden * (q[batch, token, query_head].float() * scale)[:, None],
                    dim=0,
                )
                if slot >= 0:
                    updated_state[slot, value_head] = hidden.transpose(0, 1)

    return output, updated_state


@pytest.mark.parametrize(
    ("initial_state_kind", "state_index"),
    [
        pytest.param("zero", 1, id="zero-state"),
        pytest.param("random", 1, id="random-state"),
        pytest.param("random", -1, id="padding-slot"),
    ],
)
@torch.no_grad()
def test_state_pool_uses_v_major_k_minor_layout(initial_state_kind, state_index):
    torch.manual_seed(640)
    batch_size, num_tokens = 1, 2
    num_query_heads, num_value_heads = 1, 2
    key_dim, value_dim = 16, 8
    scale = 1.0

    A_log = torch.zeros(num_value_heads, dtype=torch.float32)
    a = torch.full((batch_size, num_tokens, num_value_heads), -30.0)
    dt_bias = torch.zeros(num_value_heads)
    q = torch.randn(batch_size, num_tokens, num_query_heads, key_dim) * 0.1
    k = torch.randn(batch_size, num_tokens, num_query_heads, key_dim) * 0.1
    v = torch.randn(batch_size, num_tokens, num_value_heads, value_dim) * 0.1
    b = torch.zeros(batch_size, num_tokens, num_value_heads)
    state = (
        torch.zeros(3, num_value_heads, value_dim, key_dim)
        if initial_state_kind == "zero"
        else torch.randn(3, num_value_heads, value_dim, key_dim) * 0.1
    )
    state_before = state.clone()
    state_indices = torch.tensor([state_index], dtype=torch.int32)

    expected_output, expected_state = _reference(
        A_log,
        a,
        dt_bias,
        1.0,
        q,
        k,
        v,
        b,
        state,
        state_indices,
        scale,
    )
    state_npu = state.npu()
    actual_output = fused_sigmoid_gating_delta_rule_update_npu(
        A_log.npu(),
        a.npu(),
        dt_bias.npu(),
        1.0,
        20.0,
        q.npu(),
        k.npu(),
        v.npu(),
        b.npu(),
        state_npu,
        state_indices.npu(),
        scale=scale,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_output.cpu().float(), expected_output, rtol=0, atol=1e-3
    )
    torch.testing.assert_close(
        state_npu.cpu().float(), expected_state, rtol=0, atol=1e-3
    )
    if state_index < 0:
        torch.testing.assert_close(state_npu.cpu(), state_before, rtol=0, atol=0)

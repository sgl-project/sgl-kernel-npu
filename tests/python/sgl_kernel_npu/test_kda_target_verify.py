import pytest
import torch
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")
def test_kda_target_verify_uses_key_value_state_layout():
    """Use K != V so a transposed speculative state cannot pass silently."""
    torch.manual_seed(7)
    device = torch.device("npu")
    batch, steps = 2, 3
    h_q, h_k, h_v = 1, 1, 2
    key_dim, value_dim = 8, 6
    dtype = torch.bfloat16
    tokens = batch * steps

    q = torch.randn(1, tokens, h_q, key_dim, device=device, dtype=dtype)
    k = torch.randn(1, tokens, h_k, key_dim, device=device, dtype=dtype)
    v = torch.randn(1, tokens, h_v, value_dim, device=device, dtype=dtype)
    a = -torch.rand(1, tokens, h_k, key_dim, device=device)
    b = torch.rand(1, tokens, h_v, device=device)
    initial = torch.randn(4, h_v, key_dim, value_dim, device=device, dtype=dtype)
    initial_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    snapshots = torch.zeros(
        batch, steps, h_v, key_dim, value_dim, device=device, dtype=dtype
    )
    snapshot_indices = torch.arange(batch, device=device, dtype=torch.int32)

    actual = kda_target_verify_npu(
        A_log=torch.empty(h_k, device=device),
        dt_bias=torch.empty(h_k, key_dim, device=device),
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=snapshots,
        intermediate_state_indices=snapshot_indices,
        cache_steps=steps,
        gates_are_preactivated=True,
    )

    scale = key_dim**-0.5
    q_ref = q.squeeze(0).view(batch, steps, h_q, key_dim).float()
    k_ref = k.squeeze(0).view(batch, steps, h_k, key_dim).float()
    v_ref = v.squeeze(0).view(batch, steps, h_v, value_dim).float()
    a_ref = a.squeeze(0).view(batch, steps, h_k, key_dim).float()
    b_ref = b.squeeze(0).view(batch, steps, h_v).float()
    state = initial.index_select(0, initial_indices.long()).float()
    outputs = []
    expected_snapshots = []
    for step in range(steps):
        q_step = q_ref[:, step]
        k_step = k_ref[:, step]
        q_step = q_step / (q_step.norm(dim=-1, keepdim=True) + 1e-6)
        k_step = k_step / (k_step.norm(dim=-1, keepdim=True) + 1e-6)
        q_step = q_step.repeat_interleave(h_v // h_q, dim=1) * scale
        k_step = k_step.repeat_interleave(h_v // h_k, dim=1)
        gate = a_ref[:, step].exp().repeat_interleave(h_v // h_k, dim=1)
        state = state * gate.unsqueeze(-1)
        value = v_ref[:, step] - torch.matmul(k_step.unsqueeze(2), state).squeeze(2)
        value = value * b_ref[:, step].unsqueeze(-1)
        state = state + k_step.unsqueeze(-1) * value.unsqueeze(-2)
        outputs.append(torch.matmul(q_step.unsqueeze(2), state).squeeze(2))
        persisted_state = state.to(dtype)
        expected_snapshots.append(persisted_state)
        state = persisted_state.float()

    expected = torch.stack(outputs, dim=1).reshape_as(actual)
    expected_snapshots = torch.stack(expected_snapshots, dim=1)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        snapshots.float(), expected_snapshots.float(), rtol=2e-2, atol=2e-2
    )

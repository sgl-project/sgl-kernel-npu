import torch
from sgl_kernel_npu.fla.kda_gate import fused_kda_gate_npu
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu


def test_kda_target_verify_raw_gates_match_preactivated_gates():
    device = torch.device("npu")
    batch, steps, heads, key_dim, value_dim = 2, 3, 2, 8, 8
    tokens = batch * steps
    q = torch.randn(1, tokens, heads, key_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, dtype=torch.bfloat16, device=device)
    raw_a = torch.randn_like(q)
    raw_b = torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(1, 1, heads, 1, dtype=torch.float32, device=device)
    dt_bias = torch.randn(heads * key_dim, dtype=torch.float32, device=device)
    initial_state = torch.randn(
        batch, heads, value_dim, key_dim, dtype=torch.bfloat16, device=device
    )
    initial_indices = torch.arange(batch, dtype=torch.int32, device=device)
    intermediate_indices = torch.arange(batch, dtype=torch.int32, device=device)
    raw_scratch = torch.empty(
        batch,
        steps,
        heads,
        value_dim,
        key_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    preactivated_scratch = torch.empty_like(raw_scratch)
    lower_bound = -5.0

    raw_output = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_a,
        b=raw_b,
        initial_state_source=initial_state,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=raw_scratch,
        intermediate_state_indices=intermediate_indices,
        cache_steps=steps,
        gates_are_preactivated=False,
        lower_bound=lower_bound,
    )
    preactivated_a = fused_kda_gate_npu(
        raw_a.flatten(-2),
        A_log,
        key_dim,
        gate_bias=dt_bias,
        lower_bound=lower_bound,
    )
    preactivated_b = raw_b.float().sigmoid()
    preactivated_output = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=preactivated_a,
        b=preactivated_b,
        initial_state_source=initial_state,
        initial_state_indices=initial_indices,
        intermediate_states_buffer=preactivated_scratch,
        intermediate_state_indices=intermediate_indices,
        cache_steps=steps,
        gates_are_preactivated=True,
    )
    torch.testing.assert_close(
        raw_output.float(), preactivated_output.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        raw_scratch.float(), preactivated_scratch.float(), rtol=2e-2, atol=2e-2
    )

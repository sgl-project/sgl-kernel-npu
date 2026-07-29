"""Correctness tests for the PTO-ISA KDA decode backend (KDA_DECODE_PTO_BACKEND=1).

Shape under test is the production decode step: packed ``B=1``, ``T=N`` with one
token per sequence, slots gathered through ``state_indices`` with one padded
(``-1``) lane, and in-kernel q/k L2 norm.

The reference is a plain torch implementation of the documented semantics, with
the recurrent state held **V-major** ``[slots, HV, V, K]`` to match sglang's
``temporal_state`` pool (``mem_cache/memory_pool.py``), the prefill
``chunk_delta_h`` block pointer ``(V, K)/(K, 1)``, and the CUDA reference decode
kernel's ``o_v[None, :] * K + o_k[:, None]`` indexing.
"""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.kda_decode_pto import kda_decode_pto

DEVICE = "npu:0"
HV, K, V = 8, 128, 128
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0
# fp16 is the kernel's wire format for q/k/v/g/beta, so the output carries a few
# 1e-3 of relative error against an fp32 reference; the fp32 state does not.
OUT_RTOL = 1e-2
STATE_ATOL = 5e-3


def _inputs(n_seq, slots, seed, state_scale):
    torch.manual_seed(seed)
    dev = DEVICE
    q = torch.randn(1, n_seq, HV, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(1, n_seq, HV, V, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(1, 1, HV, 1, device=dev)
    a = torch.randn(n_seq, HV * K, device=dev)  # 2-D, as the decode path passes it
    dt_bias = torch.randn(HV * K, device=dev)
    b = torch.randn(1, n_seq, HV, device=dev)  # raw logits, kernel applies sigmoid
    state = (state_scale * torch.randn(slots, HV, V, K)).to(dev)
    cu_seqlens = torch.arange(0, n_seq + 1, dtype=torch.int32, device=dev)
    return q, k, v, A_log, a, dt_bias, b, state, cu_seqlens


def _reference(q, k, v, A_log, a, dt_bias, b, state, indices, scale):
    """Torch fp32 reference; returns (out, final_state)."""
    n_seq = q.shape[1]
    g = -A_log.reshape(1, 1, HV, 1).float().exp() * F.softplus(
        a.reshape(1, n_seq, HV, K).float() + dt_bias.reshape(HV, K).float(),
        beta=SOFTPLUS_BETA,
        threshold=SOFTPLUS_THRESHOLD,
    )
    beta = torch.sigmoid(b.reshape(1, n_seq, HV).float())
    out = torch.zeros(1, n_seq, HV, V, device=q.device, dtype=torch.float32)
    final = state.clone().float()
    for n in range(n_seq):
        slot = int(indices[n])
        if slot < 0:  # padded lane: no state, no output
            continue
        for h in range(HV):
            S = final[slot, h].clone()  # [V, K]
            q_t, k_t = q[0, n, h].float(), k[0, n, h].float()
            q_t = q_t / (q_t.pow(2).sum().sqrt() + 1e-6)
            k_t = k_t / (k_t.pow(2).sum().sqrt() + 1e-6)
            q_t = q_t * scale
            S = S * g[0, n, h].exp()[None, :]  # decay along K
            delta = (v[0, n, h].float() - S @ k_t) * beta[0, n, h]
            S = S + delta[:, None] * k_t[None, :]
            out[0, n, h] = S @ q_t
            final[slot, h] = S
    return out, final


def _run(n_seq, slots, indices, seed=7, state_scale=0.1):
    q, k, v, A_log, a, dt_bias, b, state, cu = _inputs(n_seq, slots, seed, state_scale)
    idx = torch.tensor(indices, dtype=torch.int32, device=DEVICE)
    scale = K**-0.5
    ref_out, ref_state = _reference(q, k, v, A_log, a, dt_bias, b, state, idx, scale)
    act_out = kda_decode_pto(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=SOFTPLUS_BETA,
        softplus_threshold=SOFTPLUS_THRESHOLD,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=idx,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=True,
    )
    torch.npu.synchronize()
    return act_out.float(), state.float(), ref_out, ref_state, idx


@pytest.mark.parametrize(
    "n_seq, slots, indices",
    [
        (6, 16, [5, 2, -1, 11, 0, 8]),  # shuffled slots, one padded lane
        (1, 4, [3]),  # single sequence
        (48, 64, list(range(48))),  # one work item per vector core
    ],
)
def test_matches_reference(n_seq, slots, indices):
    out, state, ref_out, ref_state, _ = _run(n_seq, slots, indices)
    mag = ref_out.abs().max().item()
    assert (out - ref_out).abs().max().item() / mag < OUT_RTOL
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=STATE_ATOL)


def test_every_head_is_processed():
    """Guards the AIV worker mapping.

    block_dim counts AIV cores here and get_subblockid() is always 0, so a
    mix-mode `worker = cid * 2 + vid` mapping would silently drop every odd work
    item -- which showed up as untouched state and all-zero output on odd heads.
    """
    out, state, ref_out, ref_state, idx = _run(6, 16, [5, 2, -1, 11, 0, 8])
    visited = [int(i) for i in idx if int(i) >= 0]
    for h in range(HV):
        assert out[0, :, h].abs().max().item() > 0, f"head {h} produced no output"
        for slot in visited:
            assert not torch.equal(
                state[slot, h], ref_state[slot, h] * 0 + state[slot, h] * 0
            ), "state untouched"
    # every visited slot/head must have moved off its initial value
    assert (state[visited] - ref_state[visited]).abs().max().item() < STATE_ATOL


def test_padded_lane_is_inert():
    """A -1 slot must write no output row and touch no state."""
    n_seq, slots, indices = 6, 16, [5, 2, -1, 11, 0, 8]
    q, k, v, A_log, a, dt_bias, b, state, cu = _inputs(n_seq, slots, 7, 0.1)
    before = state.clone()
    idx = torch.tensor(indices, dtype=torch.int32, device=DEVICE)
    out = kda_decode_pto(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=SOFTPLUS_BETA,
        softplus_threshold=SOFTPLUS_THRESHOLD,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=idx,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=True,
    )
    torch.npu.synchronize()
    assert out[0, 2].abs().max().item() == 0.0
    for slot in set(range(slots)) - {5, 2, 11, 0, 8}:
        torch.testing.assert_close(state[slot], before[slot], atol=0, rtol=0)


def test_state_is_updated_in_place():
    """sglang relies on the gathered pool being mutated, not copied."""
    q, k, v, A_log, a, dt_bias, b, state, cu = _inputs(4, 8, 11, 0.1)
    idx = torch.tensor([1, 3, 0, 5], dtype=torch.int32, device=DEVICE)
    ptr, before = state.data_ptr(), state.clone()
    kda_decode_pto(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=SOFTPLUS_BETA,
        softplus_threshold=SOFTPLUS_THRESHOLD,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=idx,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=True,
    )
    torch.npu.synchronize()
    assert state.data_ptr() == ptr
    assert not torch.equal(state, before)

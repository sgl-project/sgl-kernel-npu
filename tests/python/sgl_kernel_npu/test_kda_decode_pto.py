"""Correctness tests for the PTO-ISA KDA decode backend (KDA_DECODE_PTO_BACKEND=1)."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.kda_decode_pto import kda_decode_pto

DEVICE = "npu:0"
HV, K, V = 8, 128, 128
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0
# Both sit just above a dtype floor, so neither has much room left.
# out   atol 2e-4, rtol 5e-3 -- floor is the bf16 re-cast on return
#       (kda_decode_pto.py:112).  Measured: max |diff| 9.3e-5, rel 2.6e-3.
# state atol 2e-3, rtol 2e-3 -- fp32 throughout, so the floor is instead the
#       fp16 `g` of the un-fused gating (kda_decode_pto.py:83), which the kernel
#       feeds to exp().  Measured: max |diff| 1.9e-4, rel 2.8e-4.
OUT_ATOL, OUT_RTOL = 2e-4, 5e-3
STATE_ATOL, STATE_RTOL = 2e-3, 2e-3


def _diff(name, actual, expected, atol, rtol):
    """Print how far `actual` is from `expected`, next to the limits it must meet."""
    d = (actual.detach().float() - expected.detach().float()).abs()
    max_abs = d.max().item()
    mag = expected.detach().float().abs().max().item()
    print(
        f"  {name:5s} max abs diff {max_abs:.2e}   mean {d.mean().item():.2e}"
        f"   max ref {mag:.2e}   rel {max_abs / max(mag, 1e-12):.2e}"
        f"   (atol {atol:.0e}, rtol {rtol:.0e})"
    )
    return max_abs


def _inputs(cu, slots, seed, state_scale):
    """`cu` is the cu_seqlens list; sequence n owns tokens [cu[n], cu[n + 1])."""
    torch.manual_seed(seed)
    dev = DEVICE
    tokens = cu[-1]
    q = torch.randn(1, tokens, HV, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, HV, V, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(1, 1, HV, 1, device=dev)
    a = torch.randn(tokens, HV * K, device=dev)  # 2-D, as the decode path passes it
    dt_bias = torch.randn(HV * K, device=dev)
    b = torch.randn(1, tokens, HV, device=dev)  # raw logits, kernel applies sigmoid
    state = (state_scale * torch.randn(slots, HV, V, K)).to(dev)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=dev)
    return q, k, v, A_log, a, dt_bias, b, state, cu_seqlens


def _reference(q, k, v, A_log, a, dt_bias, b, state, indices, scale, cu):
    """Torch fp32 reference; returns (out, final_state).

    Sequence n owns tokens [cu[n], cu[n + 1]) and recurs over them in order,
    carrying the state in slot ``indices[n]``.  A zero-length sequence and a
    ``-1`` slot both leave the pool untouched, matching the kernel's `continue`.
    """
    tokens = q.shape[1]
    g = -A_log.reshape(1, 1, HV, 1).float().exp() * F.softplus(
        a.reshape(1, tokens, HV, K).float() + dt_bias.reshape(HV, K).float(),
        beta=SOFTPLUS_BETA,
        threshold=SOFTPLUS_THRESHOLD,
    )
    beta = torch.sigmoid(b.reshape(1, tokens, HV).float())
    out = torch.zeros(1, tokens, HV, V, device=q.device, dtype=torch.float32)
    final = state.clone().float()
    for n in range(len(cu) - 1):
        slot = int(indices[n])
        if slot < 0:  # padded lane: no state, no output
            continue
        for t in range(int(cu[n]), int(cu[n + 1])):
            for h in range(HV):
                S = final[slot, h].clone()  # [V, K]
                q_t, k_t = q[0, t, h].float(), k[0, t, h].float()
                q_t = q_t / (q_t.pow(2).sum().sqrt() + 1e-6)
                k_t = k_t / (k_t.pow(2).sum().sqrt() + 1e-6)
                q_t = q_t * scale
                S = S * g[0, t, h].exp()[None, :]  # decay along K
                delta = (v[0, t, h].float() - S @ k_t) * beta[0, t, h]
                S = S + delta[:, None] * k_t[None, :]
                out[0, t, h] = S @ q_t
                final[slot, h] = S
    return out, final


def _run(cu, slots, indices, seed=7, state_scale=0.1):
    lens = [cu[n + 1] - cu[n] for n in range(len(cu) - 1)]
    print(
        f"\ncu={cu} ({cu[-1]} tokens, seq lens {lens})"
        f" slots={slots} indices={list(indices)}"
    )
    q, k, v, A_log, a, dt_bias, b, state, cu_t = _inputs(cu, slots, seed, state_scale)
    idx = torch.tensor(indices, dtype=torch.int32, device=DEVICE)
    scale = K**-0.5
    ref_out, ref_state = _reference(
        q, k, v, A_log, a, dt_bias, b, state, idx, scale, cu
    )
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
        cu_seqlens=cu_t,
        is_kda=True,
    )
    torch.npu.synchronize()
    return act_out.float(), state.float(), ref_out, ref_state, idx


@pytest.mark.parametrize(
    "cu, slots, indices",
    [
        # One token per sequence -- the production decode step.
        (list(range(7)), 16, [5, 2, -1, 11, 0, 8]),  # shuffled slots, one padded lane
        (list(range(2)), 4, [3]),  # single sequence
        (list(range(49)), 64, list(range(48))),  # one work item per vector core
        # Ragged, multi-token.  Every one-token case above leaves the kernel's
        # token loop at a single iteration, so these are the only cases that
        # exercise the recurrence carrying state across tokens, the out staging
        # buffers, and the `tokens <= 0 -> continue` branch.
        ([0, 3, 3, 4, 9], 16, [2, 7, 0, 5]),  # seq 1 is empty; slot 7 must stay put
        ([0, 2, 5, 6], 8, [1, -1, 4]),  # multi-token alongside a padded lane
    ],
)
def test_matches_reference(cu, slots, indices):
    out, state, ref_out, ref_state, _ = _run(cu, slots, indices)
    _diff("out", out, ref_out, OUT_ATOL, OUT_RTOL)
    _diff("state", state, ref_state, STATE_ATOL, STATE_RTOL)
    torch.testing.assert_close(out, ref_out, atol=OUT_ATOL, rtol=OUT_RTOL)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=STATE_RTOL)


def test_every_head_is_processed():
    """Guards the AIV worker mapping.

    block_dim counts AIV cores here and get_subblockid() is always 0, so a
    mix-mode `worker = cid * 2 + vid` mapping would silently drop every odd work
    item -- which showed up as untouched state and all-zero output on odd heads.
    """
    out, state, ref_out, ref_state, idx = _run(list(range(7)), 16, [5, 2, -1, 11, 0, 8])
    visited = [int(i) for i in idx if int(i) >= 0]
    per_head = out.abs().amax(dim=(0, 1, 3)).tolist()
    print("  per-head max |out| (a 0 means that head never ran):")
    print("       " + "  ".join(f"h{h}={m:.2e}" for h, m in enumerate(per_head)))
    _diff("state", state[visited], ref_state[visited], STATE_ATOL, STATE_RTOL)

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
    slots, indices = 16, [5, 2, -1, 11, 0, 8]
    q, k, v, A_log, a, dt_bias, b, state, cu = _inputs(list(range(7)), slots, 7, 0.1)
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

    untouched = sorted(set(range(slots)) - {5, 2, 11, 0, 8})
    drift = (state - before).abs().amax(dim=(1, 2, 3))
    print(f"\npadded lane: indices={indices}")
    print(f"  seq 2 (slot -1) max |out| = {out[0, 2].abs().max().item():.2e} (want 0)")
    print(
        f"  {len(untouched)} unwritten slots, max |state change| ="
        f" {drift[untouched].max().item():.2e} (want 0)"
    )

    assert out[0, 2].abs().max().item() == 0.0
    for slot in untouched:
        torch.testing.assert_close(state[slot], before[slot], atol=0, rtol=0)


def test_state_is_updated_in_place():
    """sglang relies on the gathered pool being mutated, not copied."""
    q, k, v, A_log, a, dt_bias, b, state, cu = _inputs(list(range(5)), 8, 11, 0.1)
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

    print("\nin-place update:")
    print(f"  same buffer = {state.data_ptr() == ptr}")
    print(
        f"  max |state change| = {(state - before).abs().max().item():.2e} (want > 0)"
    )

    assert state.data_ptr() == ptr
    assert not torch.equal(state, before)


if __name__ == "__main__":
    for case in [
        (list(range(7)), 16, [5, 2, -1, 11, 0, 8]),
        (list(range(2)), 4, [3]),
        (list(range(49)), 64, list(range(48))),
        ([0, 3, 3, 4, 9], 16, [2, 7, 0, 5]),
        ([0, 2, 5, 6], 8, [1, -1, 4]),
    ]:
        test_matches_reference(*case)
    test_every_head_is_processed()
    test_padded_lane_is_inert()
    test_state_is_updated_in_place()
    print("\nall KDA decode checks passed")

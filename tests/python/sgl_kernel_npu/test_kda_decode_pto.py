"""Correctness tests for the PTO-ISA KDA decode backend (KDA_DECODE_PTO_BACKEND=1)."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.kda_decode_pto import kda_decode_pto

DEVICE = "npu:0"
HV, K, V = 8, 128, 128
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0
# out   atol 3e-4, rtol 5e-3 -- floor is the kernel's fp32 -> bf16 store, worth
#       2**-9 relative.  Measured: max |diff| 1.2e-4, rel 2.9e-3.
# state atol 1e-6, rtol 1e-6 -- fp32 the whole way now that the gating is fused,
#       so the floor is plain fp32 rounding: 1.2e-7 on the plain cases (exactly
#       2**-23) and 3.6e-7 worst, at softplus_beta=0.5.  Before the fusion this
#       was 1.9e-4, set by `g` being rounded to fp16 by torch and then fed to
#       exp(); tightened ~2000x.
OUT_ATOL, OUT_RTOL = 3e-4, 5e-3
STATE_ATOL, STATE_RTOL = 1e-6, 1e-6


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


def _inputs(cu, slots, seed, state_scale, gate_scale=1.0):
    """`cu` is the cu_seqlens list; sequence n owns tokens [cu[n], cu[n + 1]).

    `gate_scale` widens `a` and `dt_bias`, which is how the softplus branch gets
    exercised: the kernel evaluates it as relu(x) + log1p(exp(-|x|)) rather than
    torch's `beta*x > threshold ? x : ...`, so the two must agree out where a
    naive log1p(exp(x)) would have overflowed.
    """
    torch.manual_seed(seed)
    dev = DEVICE
    tokens = cu[-1]
    q = torch.randn(1, tokens, HV, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, HV, V, dtype=torch.bfloat16, device=dev)
    # Dtypes as sglang hands them over: per-token activations (q, k, v, a, b) in
    # the model's bf16, per-head parameters (A_log, dt_bias) fp32.
    A_log = torch.randn(1, 1, HV, 1, device=dev)
    a = (gate_scale * torch.randn(tokens, HV * K, device=dev)).to(torch.bfloat16)
    dt_bias = gate_scale * torch.randn(HV * K, device=dev)
    b = torch.randn(
        1, tokens, HV, dtype=torch.bfloat16, device=dev
    )  # kernel applies sigmoid
    state = (state_scale * torch.randn(slots, HV, V, K)).to(dev)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=dev)
    return q, k, v, A_log, a, dt_bias, b, state, cu_seqlens


def _reference(
    q, k, v, A_log, a, dt_bias, b, state, indices, scale, cu, sp_beta=SOFTPLUS_BETA
):
    """Torch fp32 reference; returns (out, final_state).

    Sequence n owns tokens [cu[n], cu[n + 1]) and recurs over them in order,
    carrying the state in slot ``indices[n]``.  A zero-length sequence and a
    ``-1`` slot both leave the pool untouched, matching the kernel's `continue`.
    """
    tokens = q.shape[1]
    g = -A_log.reshape(1, 1, HV, 1).float().exp() * F.softplus(
        a.reshape(1, tokens, HV, K).float() + dt_bias.reshape(HV, K).float(),
        beta=sp_beta,
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


def _run(
    cu, slots, indices, seed=7, state_scale=0.1, gate_scale=1.0, sp_beta=SOFTPLUS_BETA
):
    lens = [cu[n + 1] - cu[n] for n in range(len(cu) - 1)]
    print(
        f"\ncu={cu} ({cu[-1]} tokens, seq lens {lens})"
        f" slots={slots} indices={list(indices)}"
        f" gate_scale={gate_scale} softplus_beta={sp_beta}"
    )
    q, k, v, A_log, a, dt_bias, b, state, cu_t = _inputs(
        cu, slots, seed, state_scale, gate_scale
    )
    idx = torch.tensor(indices, dtype=torch.int32, device=DEVICE)
    scale = K**-0.5
    ref_out, ref_state = _reference(
        q, k, v, A_log, a, dt_bias, b, state, idx, scale, cu, sp_beta
    )
    act_out = kda_decode_pto(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=sp_beta,
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


@pytest.mark.parametrize("gate_scale", [1.0, 40.0, 200.0])
def test_fused_gating_matches_torch(gate_scale):
    """The in-kernel g = -exp(A_log) * softplus(a + dt_bias) against torch's.

    `gate_scale=200` drives `a + dt_bias` past +-500, where the textbook
    log1p(exp(x)) overflows fp32 (exp(89) is already inf).  The kernel's
    relu(x) + log1p(exp(-|x|)) form never exponentiates a positive number, so it
    has to stay finite and keep matching F.softplus's linear branch.
    """
    out, state, ref_out, ref_state, _ = _run(
        [0, 3, 3, 4, 9], 16, [2, 7, 0, 5], gate_scale=gate_scale
    )
    assert torch.isfinite(out).all(), "kernel produced inf/nan in out"
    assert torch.isfinite(state).all(), "kernel produced inf/nan in state"
    _diff("out", out, ref_out, OUT_ATOL, OUT_RTOL)
    _diff("state", state, ref_state, STATE_ATOL, STATE_RTOL)
    torch.testing.assert_close(out, ref_out, atol=OUT_ATOL, rtol=OUT_RTOL)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=STATE_RTOL)


@pytest.mark.parametrize("sp_beta", [0.5, 1.0, 2.5])
def test_fused_gating_softplus_beta(sp_beta):
    """softplus_beta reaches the kernel and is applied on both sides of the log."""
    out, state, ref_out, ref_state, _ = _run(
        [0, 2, 5, 6], 8, [1, -1, 4], gate_scale=5.0, sp_beta=sp_beta
    )
    _diff("out", out, ref_out, OUT_ATOL, OUT_RTOL)
    _diff("state", state, ref_state, STATE_ATOL, STATE_RTOL)
    torch.testing.assert_close(out, ref_out, atol=OUT_ATOL, rtol=OUT_RTOL)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=STATE_RTOL)


def test_launcher_runs_no_torch_math():
    """The whole point of fusing: nothing elementwise runs before the launch.

    Every tensor the kernel receives must alias the caller's buffer -- a dtype
    cast or a gating op would have produced a fresh allocation with a different
    data_ptr.  `out` is the one legitimate allocation.
    """
    q, k, v, A_log, a, dt_bias, b, state, cu = _inputs(list(range(7)), 16, 7, 0.1)
    idx = torch.tensor([5, 2, -1, 11, 0, 8], dtype=torch.int32, device=DEVICE)
    seen = {}

    orig = torch.ops.npu.kda_decode
    try:
        torch.ops.npu.kda_decode = lambda *args: seen.update(args=args)
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
    finally:
        torch.ops.npu.kda_decode = orig

    names = ["q", "k", "v", "A_log", "a", "dt_bias", "b", "state"]
    callers = [q, k, v, A_log, a, dt_bias, b, state]
    print("\nlauncher marshalling (all must be views, not copies):")
    for name, passed, caller in zip(names, seen["args"], callers):
        aliased = passed.data_ptr() == caller.data_ptr()
        print(f"  {name:8s} dtype={str(passed.dtype):16s} aliases caller: {aliased}")
        assert (
            aliased
        ), f"{name} was copied before the launch (dtype cast or gating op?)"


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
    for gs in (1.0, 40.0, 200.0):
        test_fused_gating_matches_torch(gs)
    for spb in (0.5, 1.0, 2.5):
        test_fused_gating_softplus_beta(spb)
    test_launcher_runs_no_torch_math()
    test_every_head_is_processed()
    test_padded_lane_is_inert()
    test_state_is_updated_in_place()
    print("\nall KDA decode checks passed")

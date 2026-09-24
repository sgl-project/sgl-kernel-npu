"""AscendC recurrent_gated_delta_rule vs NPU Triton: single-op accuracy and latency.

Both sides implement the same recurrence and sit at the same production call site
(SGLang's ``TritonGDNKernel.target_verify``); the Triton side ships in this repo.

Two contracts differ and are reconciled here:

* Gate. AscendC takes it pre-activated (``g`` is the log-decay, ``beta`` the
  pre-sigmoid logit); Triton takes raw ``a``/``A_log``/``dt_bias`` and evaluates
  softplus/sigmoid in-kernel. With ``A_log = dt_bias = 0`` the bridge is
  ``a = log(exp(-g) - 1)``, exact in fp32 -- hence the g range below, which keeps
  ``g`` away from 0 where the bridge degrades.
* State slots. Both sides seed from ``ssm_state_indices[i, 0]``. For ``S > 1`` the
  operator writes every token's state to ``intermediate_state`` and leaves the
  recurrent pool alone; for ``S == 1`` it writes the seed pool directly. Triton
  always writes back the slot it was seeded from. Only the applied slot is compared.

The printed ratio is a lower bound, since Triton evaluates the gates in-kernel
while the operator is handed them.

Run on NPU:
    python tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule_vs_triton.py
    python tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule_vs_triton.py \\
        --b 4 --mtp 4 --nv 16 --iters 100 --rounds 5
"""

import argparse
import statistics
import time

import pytest
import sgl_kernel_npu  # noqa: F401  registers npu ops before pytestmark
import torch
import torch_npu  # noqa: F401  makes torch.ops.npu namespace available
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update_npu,
)
from utils import require_npu_op

pytestmark = require_npu_op("recurrent_gated_delta_rule")

# Keep -g away from 0, where the bridge a = log(exp(-g) - 1) loses precision.
GATE_MIN, GATE_SPAN = 0.05, 0.95

# Triton's launch hardcodes BHV = 2.
TRITON_HV_MULTIPLE = 2


def make_inputs(b, mtp, nk, nv, dk, dv, seed, device):
    """Build one problem for both sides from the same q/k/v/g/beta, plus the state
    bookkeeping: which slot each side writes, and what to restore between calls.
    """
    if nv % TRITON_HV_MULTIPLE:
        raise ValueError(
            f"nv={nv} must be a multiple of {TRITON_HV_MULTIPLE} for Triton"
        )

    torch.manual_seed(seed)
    slots = b * mtp

    q = torch.randn(b, mtp, nk, dk, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, mtp, nk, dk, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, mtp, nv, dv, dtype=torch.bfloat16, device=device)

    # AscendC eats a packed [B, S, 2*nk*dk + nv*dv] buffer; Triton eats q/k/v.
    mix_qkv = torch.cat(
        [
            q.reshape(b, mtp, nk * dk),
            k.reshape(b, mtp, nk * dk),
            v.reshape(b, mtp, nv * dv),
        ],
        dim=-1,
    ).contiguous()

    # Batch i owns the scratch block [i*mtp, (i+1)*mtp); both sides seed from
    # ssm_state_indices[i, 0].
    seed_pool = torch.randn(
        slots, nv, dv, dk, dtype=torch.bfloat16, device=device
    ) * 0.1
    cache_indices = torch.arange(b, dtype=torch.int32, device=device)
    offsets = torch.arange(mtp, dtype=torch.int32, device=device)
    ssm_state_indices = (cache_indices[:, None] * mtp + offsets[None, :]).contiguous()
    seed_slots = ssm_state_indices[:, 0].contiguous()
    # Pre-seed the scratch pool as the correctness golden does, so both pools hold
    # the same seed for S > 1.
    scratch = torch.zeros_like(seed_pool)
    if mtp > 1:
        scratch[seed_slots.to(torch.int64)] = seed_pool[cache_indices.to(torch.int64)]
    actual_seq_lengths = torch.full((b,), mtp, dtype=torch.int32, device=device)

    # AscendC gate: log-decay, fp32, one scalar per value head.
    gate_rand = torch.rand(b, mtp, nv, dtype=torch.float32, device=device)
    g = -(GATE_MIN + GATE_SPAN * gate_rand)
    # AscendC beta: pre-sigmoid logit; Triton applies sigmoid to the same values.
    beta_logit = torch.randn(b, mtp, nv, dtype=torch.bfloat16, device=device)

    # Bridge a = log(exp(-g) - 1). On the non-KDA path a is [B, T, HV] and
    # dt_bias/A_log are [HV], so GQA (nk != nv) works.
    triton_a = torch.log(torch.expm1(-g)).reshape(1, slots, nv).contiguous()
    triton_A_log = torch.zeros(nv, dtype=torch.float32, device=device)
    triton_dt_bias = torch.zeros(nv, dtype=torch.float32, device=device)

    ascendc = dict(
        mix_qkv=mix_qkv,
        recurrent_state=seed_pool,
        beta=beta_logit,
        scale=dk**-0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        nk=nk,
        nv=nv,
        intermediate_state=scratch if mtp > 1 else None,
        cache_indices=cache_indices if mtp > 1 else None,
        g=g,
    )
    triton = dict(
        A_log=triton_A_log,
        a=triton_a,
        dt_bias=triton_dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.reshape(1, slots, nk, dk),
        k=k.reshape(1, slots, nk, dk),
        v=v.reshape(1, slots, nv, dv),
        b=beta_logit.reshape(1, slots, nv),
        initial_state_source=scratch if mtp > 1 else seed_pool,
        initial_state_indices=seed_slots,
        scale=dk**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.arange(0, slots + 1, mtp, dtype=torch.int32, device=device),
        is_kda=False,
    )

    triton_pool = triton["initial_state_source"]
    # Assert only on the slot the operator actually writes (see module docstring).
    candidates = {"recurrent[cache]": (seed_pool, cache_indices)}
    if mtp > 1:
        candidates["scratch[i,S-1]"] = (scratch, ssm_state_indices[:, mtp - 1])
    meta = dict(
        ascendc_candidates=candidates,
        triton_pool=triton_pool,
        triton_final_slots=seed_slots,
        pools=[seed_pool, scratch],
    )
    return ascendc, triton, meta


def restore(snapshot):
    for tensor, saved in snapshot:
        tensor.copy_(saved)


def run_ascendc(args, snapshot):
    """Call the AscendC operator (it writes state in place)."""
    restore(snapshot)
    return torch.ops.npu.recurrent_gated_delta_rule(
        args["mix_qkv"],
        args["recurrent_state"],
        beta=args["beta"],
        scale=args["scale"],
        actual_seq_lengths=args["actual_seq_lengths"],
        ssm_state_indices=args["ssm_state_indices"],
        nk=args["nk"],
        nv=args["nv"],
        intermediate_state=args["intermediate_state"],
        cache_indices=args["cache_indices"],
        num_accepted_tokens=None,
        g=args["g"],
        gk=None,
    )


def run_triton(args, snapshot):
    """Call the NPU Triton reference."""
    restore(snapshot)
    return fused_sigmoid_gating_delta_rule_update_npu(**args)


def bench(fn, warmup, iters, rounds):
    """Time fn: best and median ms per call, over `rounds` rounds of `iters` calls."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    per_round = []
    for _ in range(rounds):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        per_round.append((time.perf_counter() - t0) / iters * 1e3)
    return min(per_round), statistics.median(per_round)


def report_out(got, want, atol):
    """Compare per-token outputs; per-step and per-batch maxima localise divergence."""
    diff = (got.float() - want.float()).abs()
    max_diff = diff.max().item()
    print(f"    out  : max={max_diff:.6f} mean={diff.mean().item():.6f}")
    if got.shape[1] > 1:
        per_step = [f"{s}:{diff[:, s].max().item():.6f}" for s in range(got.shape[1])]
        print(f"      per step: {' '.join(per_step)}")
        per_batch = [f"{i}:{diff[i, 0].max().item():.6f}" for i in range(got.shape[0])]
        print(f"      step 0 per batch: {' '.join(per_batch)}")
    return max_diff, max_diff <= atol


def report_state(candidates, want, atol):
    """Compare Triton's write-back slot against every slot the operator may use."""
    best_diff, best_name = float("inf"), None
    for name, got in candidates.items():
        diff = (got.float() - want.float()).abs()
        max_diff = diff.max().item()
        print(f"    state vs {name}: max={max_diff:.6f} mean={diff.mean().item():.6f}")
        if max_diff < best_diff:
            best_diff, best_name = max_diff, name
    print(f"    state: best match {best_name} max={best_diff:.6f}")
    return best_diff, best_diff <= atol


def run_case(b, mtp, nk, nv, dk, dv, seed, atol, warmup, iters, rounds, device):
    print(
        f"\n=== B={b} S={mtp} nk={nk} nv={nv} dk={dk} dv={dv} "
        f"iters={iters} rounds={rounds} ==="
    )
    ascendc, triton, meta = make_inputs(b, mtp, nk, nv, dk, dv, seed, device)
    snapshot = [(t, t.clone()) for t in meta["pools"]]

    # ---- accuracy: same inputs, same seed slots, pools restored before each call ----
    out_ascendc = run_ascendc(ascendc, snapshot).to(torch.float32)
    state_ascendc = {
        name: pool[slots].to(torch.float32).clone()
        for name, (pool, slots) in meta["ascendc_candidates"].items()
    }
    out_triton = run_triton(triton, snapshot).to(torch.float32)
    state_triton = (
        meta["triton_pool"][meta["triton_final_slots"]].to(torch.float32).clone()
    )
    torch.npu.synchronize()

    print(f"  shapes: out asc={tuple(out_ascendc.shape)} tri={tuple(out_triton.shape)}")
    ok = True
    max_out, ok_out = report_out(
        out_ascendc, out_triton.reshape(out_ascendc.shape), atol
    )
    ok &= ok_out
    max_state, ok_state = report_state(state_ascendc, state_triton, atol)
    ok &= ok_state
    print(f"  [{'PASS' if ok else 'FAIL'}] atol={atol}")

    # ---- performance ----
    best_asc, med_asc = bench(
        lambda: run_ascendc(ascendc, snapshot), warmup, iters, rounds
    )
    best_tri, med_tri = bench(
        lambda: run_triton(triton, snapshot), warmup, iters, rounds
    )
    print(
        f"  ascendc: best={best_asc:.4f}ms median={med_asc:.4f}ms\n"
        f"  triton : best={best_tri:.4f}ms median={med_tri:.4f}ms\n"
        f"  speedup (triton/ascendc, best): {best_tri / best_asc:.2f}x"
        f"   (ascendc is handed pre-activated gates; triton computes them in-kernel)"
    )
    return {
        "b": b,
        "mtp": mtp,
        "nk": nk,
        "nv": nv,
        "dk": dk,
        "dv": dv,
        "ok": ok,
        "max_out": max_out,
        "max_state": max_state,
        "ascendc_best": best_asc,
        "triton_best": best_tri,
        "speedup": best_tri / best_asc,
    }


DEFAULT_SHAPES = [
    # (b, mtp, nk, nv, dk, dv)
    (2, 1, 8, 16, 128, 128),  # production verify shape, single token
    (4, 4, 8, 16, 128, 128),  # MTP = 4
    (8, 8, 8, 16, 128, 128),  # MTP = 8, the kernel's MAX_MTP
    (2, 4, 8, 16, 64, 32),  # dk != dv: the shape that exposes layout bugs
]


@pytest.mark.parametrize(
    "shape",
    DEFAULT_SHAPES,
    ids=lambda s: "B{}-S{}-nk{}-nv{}-dk{}-dv{}".format(*s),
)
def test_recurrent_gated_delta_rule_vs_triton(shape):
    if not hasattr(torch.ops.npu, "recurrent_gated_delta_rule"):
        pytest.skip("torch.ops.npu.recurrent_gated_delta_rule is not registered")
    result = run_case(
        *shape, seed=42, atol=0.01, warmup=5, iters=20, rounds=3, device="npu"
    )
    assert result["ok"], (
        f"accuracy mismatch: max_out={result['max_out']}, "
        f"max_state={result['max_state']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AscendC recurrent_gated_delta_rule vs NPU Triton: accuracy and latency"
        )
    )
    parser.add_argument(
        "--b", type=int, help="batch size (default: run the built-in shapes)"
    )
    parser.add_argument("--mtp", type=int, default=4, help="tokens per request, <= 8")
    parser.add_argument("--nk", type=int, default=8)
    parser.add_argument("--nv", type=int, default=16)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.01,
        help="per-element tolerance; the observed spread is ~4e-03",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100, help="calls per timing round")
    parser.add_argument("--rounds", type=int, default=5, help="timing rounds")
    args = parser.parse_args()

    if not hasattr(torch.ops.npu, "recurrent_gated_delta_rule"):
        raise SystemExit(
            "torch.ops.npu.recurrent_gated_delta_rule is not registered. "
            "Build sgl-kernel-npu and install the wheel first."
        )
    if not hasattr(torch, "npu") or torch.npu.device_count() <= 0:
        raise SystemExit("NPU device is not available")

    if args.b is None:
        shapes = DEFAULT_SHAPES
    else:
        shapes = [(args.b, args.mtp, args.nk, args.nv, args.dk, args.dv)]

    results = [
        run_case(
            *shape, args.seed, args.atol, args.warmup, args.iters, args.rounds, "npu"
        )
        for shape in shapes
    ]

    print("\n=== summary ===")
    print(f"{'shape':<34} {'status':>8} {'ascendc':>10} {'triton':>10} {'speedup':>9}")
    for r in results:
        label = f"B{r['b']}-S{r['mtp']}-nk{r['nk']}-nv{r['nv']}-dk{r['dk']}-dv{r['dv']}"
        print(
            f"{label:<34} {'PASS' if r['ok'] else 'FAIL':>8} "
            f"{r['ascendc_best']:>9.4f}ms {r['triton_best']:>9.4f}ms "
            f"{r['speedup']:>8.2f}x"
        )

    failed = [r for r in results if not r["ok"]]
    passed = len(results) - len(failed)
    print(f"\n{passed}/{len(results)} shapes within atol={args.atol}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Microbenchmark: fused_shared_expert_mlp vs the eager chain (NPU only).

Same-machine, same-input comparison — the acceptance criterion is RELATIVE:
the fused path must beat the original eager chain on this machine.
Absolute us / GB/s numbers are reported for reference only (they vary across
machines); do not gate on them.

Usage:
  python benchmark/fused_shared_expert_bench.py                 # default sweep
  python benchmark/fused_shared_expert_bench.py --m 32 --iters 200 --repeat 5
"""

import argparse
import statistics
import time

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

from sgl_kernel_npu.activation.fused_sigmoid_mul import fused_sigmoid_mul_broadcast
from sgl_kernel_npu.fused.fused_shared_expert import fused_shared_expert_mlp

IN, HALF, N = 2048, 512, 2048
# gate_up 4.19MB + down 2.10MB + gate 4KB (reference-only bandwidth model)
WEIGHT_BYTES = (2 * HALF * IN + N * HALF + IN) * 2


def eager_chain(hidden, wgu, wd, wg):
    shared = torch_npu.npu_swiglu(hidden @ wgu.t())
    shared = shared @ wd.t()
    gate = hidden @ wg.t()
    return fused_sigmoid_mul_broadcast(shared, gate)


def eager_chain_breakdown(hidden, wgu, wd, wg):
    """Per-op timings of the original chain (for diagnosis)."""
    stages = {
        "gate_up_gemm": lambda: hidden @ wgu.t(),
    }
    gu = stages["gate_up_gemm"]()
    stages["swiglu"] = lambda: torch_npu.npu_swiglu(gu)
    inter = stages["swiglu"]()
    stages["down_gemm"] = lambda: inter @ wd.t()
    down = stages["down_gemm"]()
    stages["gate_gemv"] = lambda: hidden @ wg.t()
    gate = stages["gate_gemv"]()
    stages["sigmoid_mul"] = lambda: fused_sigmoid_mul_broadcast(down, gate)
    return stages


def time_fn(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def time_repeated(fn, iters, repeat):
    """Return (median, min) over `repeat` rounds of per-iter us."""
    rounds = [time_fn(fn, iters) for _ in range(repeat)]
    return statistics.median(rounds), min(rounds)


def graph_time(fn, calls=20, replays=30):
    """Per-invocation device time of `fn` measured inside an NPU graph.

    Eager-mode wall timing on this host is dominated by Triton/torch_npu
    host dispatch (~40-80us per launch), which disappears under graph
    capture — the way the op actually runs in production decode. Capture
    `calls` invocations once and time `replays` graph replays.
    """
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(g):
            for _ in range(calls):
                fn()
    except Exception as e:
        print(f"      graph capture failed ({type(e).__name__}: {e}); "
              f"graph timing skipped for this M")
        return None
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(replays):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / (calls * replays) * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="*", default=[1, 8, 32, 128, 256])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="also print per-op timings of the eager chain",
    )
    parser.add_argument(
        "--stage-time",
        action="store_true",
        help="also time K1 and K2 separately (device diagnosis)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also measure in-graph device time via NPU graph capture; "
             "this is the production-relevant comparison and the primary "
             "acceptance criterion",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise SystemExit("NPU not available")

    print(f"{'M':>5} {'eager us':>10} {'fused us':>10} {'speedup':>8} "
          f"{'fused GB/s':>11} {'verdict':>8}")
    all_pass = True
    graph_all_pass = True
    for M in args.m:
        torch.manual_seed(0)
        hidden = torch.randn(M, IN).to(torch.bfloat16).npu()
        wgu = (torch.randn(2 * HALF, IN) / IN**0.5).to(torch.bfloat16).npu()
        wd = (torch.randn(N, HALF) / HALF**0.5).to(torch.bfloat16).npu()
        wg = (torch.randn(1, IN) / IN**0.5).to(torch.bfloat16).npu()

        # Same-input sanity: the perf comparison is only meaningful if both
        # paths compute the same thing on these exact inputs.
        ref = eager_chain(hidden, wgu, wd, wg)
        out = fused_shared_expert_mlp(hidden, wgu, wd, wg)
        max_abs = (out.float() - ref.float()).abs().max().item()
        if not torch.allclose(out.float(), ref.float(), rtol=1.6e-2, atol=1e-2):
            print(f"{M:>5}  NUMERIC MISMATCH (max abs diff {max_abs}), skip timing")
            all_pass = False
            continue

        e_med, e_min = time_repeated(lambda: eager_chain(hidden, wgu, wd, wg),
                                     args.iters, args.repeat)
        f_med, f_min = time_repeated(lambda: fused_shared_expert_mlp(hidden, wgu, wd, wg),
                                     args.iters, args.repeat)
        # Verdict on the least noisy statistic (min), confirmed by median.
        speedup_min = e_min / f_min
        verdict = "FASTER" if speedup_min > 1.0 else "SLOWER"
        if speedup_min <= 1.0:
            all_pass = False
        gbs = WEIGHT_BYTES / (f_min * 1e-6) / 1e9
        print(f"{M:>5} {e_min:>10.2f} {f_min:>10.2f} "
              f"{speedup_min:>7.2f}x {gbs:>10.1f} {verdict:>8}  "
              f"(median {e_med / f_med:.2f}x, max abs diff {max_abs:.2e})")

        if args.breakdown:
            stages = eager_chain_breakdown(hidden, wgu, wd, wg)
            parts = {name: time_repeated(fn, args.iters, args.repeat)[1]
                     for name, fn in stages.items()}
            total = sum(parts.values())
            detail = "  ".join(f"{k}={v:.2f}" for k, v in parts.items())
            print(f"      eager breakdown: {detail}  (sum={total:.2f}us)")

        if args.stage_time:
            from sgl_kernel_npu.fused.fused_shared_expert import (
                _launch_k1,
                _launch_k2,
                _pack_gate_up_weight,
                _padded_gate_weight,
            )

            wgp = _padded_gate_weight(wg)
            wpack = _pack_gate_up_weight(wgu, HALF)
            inter = torch.empty((M, HALF), dtype=hidden.dtype, device=hidden.device)
            gate = torch.empty((M,), dtype=hidden.dtype, device=hidden.device)
            out_st = torch.empty((M, N), dtype=hidden.dtype, device=hidden.device)
            k1_min = time_repeated(
                lambda: _launch_k1(hidden, wpack, wgp, inter, gate, M, IN, HALF),
                args.iters, args.repeat,
            )[1]
            k2_min = time_repeated(
                lambda: _launch_k2(inter, wd, gate, out_st, None, M, HALF, N),
                args.iters, args.repeat,
            )[1]
            print(f"      stage time: K1={k1_min:.2f}us K2={k2_min:.2f}us "
                  f"(includes ~per-launch host dispatch)")

        if args.graph:
            e_g = graph_time(lambda: eager_chain(hidden, wgu, wd, wg))
            f_g = graph_time(lambda: fused_shared_expert_mlp(hidden, wgu, wd, wg))
            if e_g is not None and f_g is not None:
                sp_g = e_g / f_g
                graph_all_pass = graph_all_pass and (sp_g > 1.0)
                print(f"      graph: eager={e_g:.2f}us fused={f_g:.2f}us "
                      f"speedup={sp_g:.2f}x  <- production-relevant")

    print("\nRELATIVE ACCEPTANCE (eager-mode wall time; host-dispatch bound,"
          " reference only):", "PASS" if all_pass else "FAIL")
    if args.graph:
        print("GRAPH ACCEPTANCE (in-graph device time, production-relevant"
              " criterion):", "PASS" if graph_all_pass else "FAIL")
    try:
        from sgl_kernel_npu.fused.fused_shared_expert import (
            _fused_shared_expert_k1,
            _fused_shared_expert_k2,
        )

        print(f"chosen configs: K1={_fused_shared_expert_k1.best_config} "
              f"K2={_fused_shared_expert_k2.best_config}")
    except Exception:
        pass
    print("note: absolute us / GB/s are machine-specific, reference only.")


if __name__ == "__main__":
    main()

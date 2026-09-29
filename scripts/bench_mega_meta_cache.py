#!/usr/bin/env python3
"""A/B benchmark for mega_chunk_gdn wrapper overhead on Ascend NPU.

Variant A "upstream": per-call allocation logic of the ORIGINAL wrapper
(every buffer fresh per layer, ~4 GB/layer at 128k tokens incl. a dead 1 GB
fp32 A_inv_f32, plus a cu_seqlens D2H sync).

Variant B "optimized": the patched sgl_kernel_npu.fla.mega_chunk_gdn wrapper
  - cross-layer scratch cache (kkt/wy/h/o workspaces, keyed by shape)
  - grow-only token slab reused across layers AND forward steps
    (g_sum/g_t/beta_t/A/A_inv/w/u/v_new), SGLANG_NPU_GDN_POOL_OUTPUTS
  - dropped dead op argument A_inv_f32 (kernel never reads it; dummy scalar)
  - zeros -> empty where the kernel provably covers every element
  - cu32 / num_chunks cached by content, one D2H per distinct cu_seqlens

Both variants call the SAME AscendC op torch.ops.npu.mega_chunk_gdn, so the
wall delta isolates wrapper overhead (alloc / memset / D2H / launch churn).

Run on an Ascend host (sgl_kernel_npu installed):

  python scripts/bench_mega_meta_cache.py
  python scripts/bench_mega_meta_cache.py --seq 16384,65536,131072 --layers 30
  python scripts/bench_mega_meta_cache.py --num-seqs 4            # packed batch
  python scripts/bench_mega_meta_cache.py --clone-cu              # fresh cu object/layer

Notes:
  - With pooling ON, buffers that escape the call but are discarded by every
    caller in the serving stack (A_inv/w/v_new; u is not even returned) are
    served from the shared slab. Disable via SGLANG_NPU_GDN_POOL_OUTPUTS=0.
  - Do not name this file profile.py (cProfile stdlib clash).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

HEAD_DIM = 128
CHUNK_SIZE = 128
GB = 1024**3

ZEROS_UPSTREAM = ("A", "A_inv_f32", "A_inv", "h", "final_state", "kkt",
                  "wy_a1", "wy_a2", "h_ws", "o_qk", "o_qs", "o_gated")


def sync() -> None:
    torch.npu.synchronize()


def fmt_us(us: float) -> str:
    if us >= 1e6:
        return f"{us / 1e6:.3f} s"
    if us >= 1e3:
        return f"{us / 1e3:.3f} ms"
    return f"{us:.1f} us"


def parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_gdn_inputs(total_tokens, num_seqs, qh, vh, dim, device, dtype):
    assert total_tokens % num_seqs == 0, "tokens must be divisible by num_seqs"
    seg = total_tokens // num_seqs
    cu = None
    if num_seqs > 1:
        cu = torch.arange(0, total_tokens + 1, seg, device=device, dtype=torch.int64)
    q = torch.randn(1, total_tokens, qh, dim, device=device, dtype=dtype)
    k = F.normalize(
        torch.randn(1, total_tokens, qh, dim, device=device, dtype=torch.float32),
        p=2, dim=-1,
    ).to(dtype)
    v = torch.randn(1, total_tokens, vh, dim, device=device, dtype=dtype)
    g = F.logsigmoid(torch.rand(1, total_tokens, vh, device=device, dtype=torch.float32))
    beta = torch.rand(1, total_tokens, vh, device=device, dtype=dtype).sigmoid()
    h0 = torch.randn(num_seqs, vh, dim, dim, device=device, dtype=dtype)
    return dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0, cu=cu, scale=dim**-0.5)


# ---------------------------------------------------------------------------
# Variant A: original upstream wrapper, verbatim allocation logic
# ---------------------------------------------------------------------------

def run_upstream(inp):
    q = inp["q"].half()
    k = inp["k"].half()
    v = inp["v"].half()
    beta = inp["beta"].half()
    g = inp["g"]
    initial_state = inp["h0"]
    cu_seqlens = inp["cu"]

    _, total_tokens, _, head_dim = q.shape
    num_value_heads = v.shape[-2]
    device = q.device

    if cu_seqlens is None:
        cu32 = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
        num_sequences, num_chunks = 1, -(-total_tokens // CHUNK_SIZE)
    else:
        cu32 = cu_seqlens.to(torch.int32)
        cu = cu_seqlens.cpu().tolist()  # D2H sync every call (upstream behavior)
        num_sequences = len(cu) - 1
        num_chunks = sum(-(-(b - a) // CHUNK_SIZE) for a, b in zip(cu, cu[1:]))
    num_matrices = num_chunks * num_value_heads

    from sgl_kernel_npu.fla.mega_chunk_gdn import (
        _block_dim, _masks, _minus_identity,
    )

    device_type, device_index = (device.type, 0 if device.index is None else device.index)
    mask_lower, mask_full = _masks(device_type, device_index)
    minus_identity = _minus_identity(device_type, device_index)

    g_sum = torch.empty_like(g, dtype=torch.float32)
    g_t = torch.empty(num_value_heads, total_tokens, device=device, dtype=torch.float32)
    beta_t = torch.empty(num_value_heads, total_tokens, device=device, dtype=torch.float16)
    A = torch.zeros(1, total_tokens, num_value_heads, CHUNK_SIZE, device=device,
                    dtype=torch.float16)
    A_inv_f32 = torch.zeros_like(A, dtype=torch.float32)  # dead op argument
    A_inv = torch.zeros_like(A)
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    h = torch.zeros(num_chunks * num_value_heads, head_dim, head_dim, device=device,
                    dtype=torch.float16)
    v_new = torch.empty_like(v)
    final_state = torch.zeros(num_sequences * num_value_heads, head_dim, head_dim,
                              device=device, dtype=torch.float16)
    has_initial_state = initial_state is not None
    initial_state = initial_state.half() if has_initial_state else final_state
    out = torch.empty_like(v)

    block_dim = _block_dim(device)
    kkt = torch.zeros(block_dim * 2, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
    wy_a1 = torch.zeros(block_dim, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
    wy_a2 = torch.zeros_like(wy_a1)
    h_ws = torch.zeros(block_dim * 4, head_dim, head_dim, device=device, dtype=torch.float16)
    o_qk = torch.zeros(block_dim, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
    o_qs = torch.zeros(block_dim, CHUNK_SIZE, head_dim, device=device, dtype=torch.float16)
    o_gated = torch.zeros_like(o_qk)

    torch.ops.npu.mega_chunk_gdn(
        q, k, v, g, beta, mask_lower, mask_full, minus_identity, cu32,
        out, g_sum, g_t, beta_t, A, A_inv_f32, A_inv, w, u, h, v_new,
        final_state, initial_state, has_initial_state,
        kkt, wy_a1, wy_a2, h_ws, o_qk, o_qs, o_gated,
        block_dim, num_sequences, total_tokens, total_tokens, num_matrices,
    )

    # Same post-processing as the ORIGINAL wrapper so the A/B comparison is
    # apples-to-apples: upstream casts A_inv/w/v_new/h back to the model dtype
    # on return (they are dead outputs at default SUPPRESS_LEVEL, but the
    # upstream wrapper still pays for the copies).
    scale = inp["scale"]
    model_dtype = inp["q"].dtype
    A_inv = A_inv.to(model_dtype)
    w = w.to(model_dtype)
    v_new = v_new.to(model_dtype)
    h = h.view(1, num_chunks, num_value_heads, head_dim, head_dim).to(model_dtype)
    final_state = final_state.view(
        num_sequences, num_value_heads, head_dim, head_dim
    ).to(torch.float32)
    return (out * scale).to(model_dtype), final_state, h


# ---------------------------------------------------------------------------
# Variant B: patched wrapper from the installed package
# ---------------------------------------------------------------------------

def _wrapper_supports_return_internals():
    import inspect
    from sgl_kernel_npu.fla.mega_chunk_gdn import run_mega_chunk_gdn
    try:
        return "return_internals" in inspect.signature(run_mega_chunk_gdn).parameters
    except (TypeError, ValueError):
        return False


def make_run_optimized(clone_cu: bool, return_internals: bool):
    from sgl_kernel_npu.fla.mega_chunk_gdn import run_mega_chunk_gdn

    # Older installs of the wrapper have no cast-skip optimization, so passing
    # return_internals would raise TypeError; detect and fall back to the
    # full-return call (which is exactly what those servers run).
    supports = _wrapper_supports_return_internals()

    def _run(inp):
        cur = inp
        if clone_cu and inp["cu"] is not None:
            cur = dict(inp)
            cur["cu"] = inp["cu"].clone()
        if supports:
            g_sum, o, A_inv, final_state, w, h, v_new = run_mega_chunk_gdn(
                inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
                inp["scale"], inp["h0"], True, cur["cu"],
                return_internals=return_internals,
            )
        else:
            g_sum, o, A_inv, final_state, w, h, v_new = run_mega_chunk_gdn(
                inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
                inp["scale"], inp["h0"], True, cur["cu"],
            )
        return o, final_state, h

    return _run


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def memory_model(total_tokens, vh, num_seqs, block_dim):
    fp16, fp32 = 2, 4
    # Packed sequences chunk independently: ceil per segment, not over the total.
    seg = total_tokens // num_seqs
    chunks = num_seqs * (-(-seg // CHUNK_SIZE))
    per_tensor = {
        "out": total_tokens * vh * HEAD_DIM * fp16,
        "g_sum": total_tokens * vh * fp32,
        "g_t": vh * total_tokens * fp32,
        "beta_t": vh * total_tokens * fp16,
        "A": total_tokens * vh * CHUNK_SIZE * fp16,
        "A_inv_f32": total_tokens * vh * CHUNK_SIZE * fp32,
        "A_inv": total_tokens * vh * CHUNK_SIZE * fp16,
        "w": total_tokens * vh * HEAD_DIM * fp16,
        "u": total_tokens * vh * HEAD_DIM * fp16,
        "h": chunks * vh * HEAD_DIM * HEAD_DIM * fp16,
        "v_new": total_tokens * vh * HEAD_DIM * fp16,
        "final_state": num_seqs * vh * HEAD_DIM * HEAD_DIM * fp16,
        "kkt": block_dim * 2 * CHUNK_SIZE * CHUNK_SIZE * fp16,
        "wy_a1": block_dim * CHUNK_SIZE * CHUNK_SIZE * fp16,
        "wy_a2": block_dim * CHUNK_SIZE * CHUNK_SIZE * fp16,
        "h_ws": block_dim * 4 * HEAD_DIM * HEAD_DIM * fp16,
        "o_qk": block_dim * CHUNK_SIZE * CHUNK_SIZE * fp16,
        "o_qs": block_dim * CHUNK_SIZE * HEAD_DIM * fp16,
        "o_gated": block_dim * CHUNK_SIZE * CHUNK_SIZE * fp16,
    }
    total_up = sum(per_tensor.values())
    zeros_up = sum(per_tensor[n] for n in ZEROS_UPSTREAM)
    # optimized steady state: only out + h + final_state escape per layer
    per_call_opt = per_tensor["out"] + per_tensor["h"] + per_tensor["final_state"]
    return per_tensor, total_up, zeros_up, per_call_opt


def make_multilayer_fn(run_fn: Callable, inp: dict, n_layers: int):
    """Run n_layers forward passes; only the last layer's outputs stay alive,
    like the serving stack where each layer's outputs are consumed/freed."""

    def _fn():
        last = None
        for _ in range(n_layers):
            last = run_fn(inp)
        return last

    return _fn


def bench_host(fn: Callable, n_layers: int, iters: int) -> float:
    """Host-side time per layer (no sync inside: exposes alloc/memset/D2H/launch)."""
    fn()
    sync()
    best = float("inf")
    for _ in range(max(3, iters)):
        t0 = time.perf_counter()
        fn()
        host_us = (time.perf_counter() - t0) * 1e6
        sync()
        best = min(best, host_us)
    return best / n_layers


def bench_wall(fn: Callable, warmup: int, active: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(active):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e6)
    return {"mean_us": statistics.mean(times), "p50_us": statistics.median(times),
            "min_us": min(times)}


def bench_memory(fn: Callable, iters: int = 2) -> Dict[str, float]:
    """Allocator-level HBM accounting around repeated multi-layer forwards.

    churn/forward counts every byte the caching allocator hands out across one
    whole forward (fn call) — the traffic this PR removes. Peak allocated/
    reserved show the footprint kept around. Input tensors live through both
    variants' windows, so they cancel out in the base-vs-opt delta.
    """
    fn()
    sync()  # one-time slab/cache build happens here, outside the window
    torch.npu.empty_cache()
    try:
        torch.npu.reset_peak_memory_stats()
    except Exception:
        pass
    try:
        stats0 = torch.npu.memory_stats()
    except Exception:
        stats0 = None

    for _ in range(iters):
        fn()
        sync()

    result = {
        "peak_alloc_gb": torch.npu.max_memory_allocated() / GB,
        "peak_res_gb": torch.npu.max_memory_reserved() / GB,
        "churn_gb": None,
        "allocs": None,
    }
    if stats0 is not None:
        try:
            stats1 = torch.npu.memory_stats()

            def delta(key):
                return stats1.get(key, 0) - stats0.get(key, 0)

            result["churn_gb"] = delta("allocated_bytes.all.allocated") / iters / GB
            result["allocs"] = delta("allocation.all.allocated") / iters
        except Exception:
            pass
    return result


def check_outputs(up, opt) -> List[str]:
    errs = []
    for name, a, b in zip(("o", "final_state", "h"), up, opt):
        if (a is None) != (b is None):
            errs.append(f"{name}: None-ness mismatch")
            continue
        if a is None:
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            errs.append(f"{name}: shape/dtype {tuple(a.shape)}/{a.dtype} "
                        f"vs {tuple(b.shape)}/{b.dtype}")
            continue
        if torch.equal(a, b):
            print(f"    {name:<12} bit-equal")
            continue
        a32, b32 = a.float(), b.float()
        diff = (a32 - b32).abs()
        max_abs = diff.max().item()
        # allclose-style threshold; both variants call the same op with the
        # same inputs, so any real divergence here is a wrapper bug.
        ok = torch.allclose(a32, b32, rtol=1e-3, atol=1e-3)
        print(f"    {name:<12} maxdiff={max_abs:.3e} allclose={'yes' if ok else 'NO'}")
        if not ok:
            errs.append(f"{name}: maxdiff={max_abs:.3e}")
    return errs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", type=str, default="8192,32768,131072",
                        help="packed total tokens per scenario")
    parser.add_argument("--num-seqs", type=int, default=1,
                        help="sequences packed in the request (1 = single long seq)")
    parser.add_argument("--layers", type=int, default=30,
                        help="GDN layers per forward (Qwen3.6-35B-A3B: 30)")
    parser.add_argument("--q-heads", type=int, default=8, help="key/query heads (TP=2)")
    parser.add_argument("--v-heads", type=int, default=16, help="value heads (TP=2)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument("--clone-cu", action="store_true",
                        help="clone cu_seqlens each layer (content-keyed cache still hits)")
    parser.add_argument("--suppress-level", type=int, default=0,
                        help="GDN_RECOMPUTE_SUPPRESS_LEVEL as used by the serving "
                             "stack; >=3 keeps A_inv/w/v_new (wrapper casts them), "
                             "0-2 discards them (wrapper skips the casts)")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    import torch_npu  # noqa: F401  # registers the npu device on torch
    import sgl_kernel_npu  # noqa: F401  # loads libsgl_kernel_npu, registers ops

    if not torch.npu.is_available():
        raise RuntimeError("NPU not available; run this script on an Ascend host.")
    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        raise RuntimeError(
            "torch.ops.npu.mega_chunk_gdn missing; rebuild sgl_kernel_npu with the "
            "mega_chunk_gdn kernel enabled."
        )

    from sgl_kernel_npu.fla.mega_chunk_gdn import _block_dim
    from sgl_kernel_npu.fla.utils import (
        clear_gdn_meta_cache, is_gdn_meta_cache_enabled, set_gdn_meta_cache,
    )

    device = torch.device("npu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    block_dim = _block_dim(device)
    seqs = parse_ints(args.seq)

    print("=" * 78)
    print("mega_chunk_gdn wrapper A/B: upstream alloc vs optimized (meta cache + slab)")
    print(f"  device={device} cube_block_dim={block_dim} H_qk={args.q_heads} "
          f"H_v={args.v_heads} D={HEAD_DIM} CHUNK={CHUNK_SIZE} dtype={args.dtype}")
    print(f"  layers={args.layers} num_seqs={args.num_seqs} clone_cu={args.clone_cu} "
          f"pool_outputs={os.getenv('SGLANG_NPU_GDN_POOL_OUTPUTS', '1')}")
    if _wrapper_supports_return_internals():
        mode = "serving path (A_inv/w/v_new cast-skip)" if args.suppress_level < 3 \
            else "full-return path"
        print(f"  wrapper cast-skip: supported, running {mode}")
    else:
        print("  wrapper cast-skip: NOT installed (old build); both arms use the")
        print("  full-return path, i.e. A_inv/w/v_new/h casts are still paid.")
    print("=" * 78)

    rows: List[dict] = []
    for seq in seqs:
        if seq % args.num_seqs != 0:
            print(f"\n[skip] seq={seq} not divisible by num_seqs={args.num_seqs}")
            continue
        per_tensor, total_up, zeros_up, per_call_opt = memory_model(
            seq, args.v_heads, args.num_seqs, block_dim)
        free_b, _ = torch.npu.mem_get_info()
        # Peak ~= inputs + one layer's allocations (outputs freed per layer),
        # so upstream and optimized differ by roughly total_up - slab.
        print(f"\n{'=' * 78}")
        print(f"tokens={seq}")
        print(f"  upstream alloc/layer : {total_up / GB:6.2f} GB  "
              f"(zeros: {zeros_up / GB:5.2f} GB, x{args.layers} = "
              f"{total_up * args.layers / GB:.1f} GB churn/forward)")
        print(f"  optimized alloc/layer: {per_call_opt / GB:6.2f} GB steady-state "
              f"(out+h+final_state; slab buffers reused)")
        if total_up + per_call_opt + 2 * GB > free_b:
            print("  [warn] tight free memory; reduce --seq")
            continue

        inp = build_gdn_inputs(seq, args.num_seqs, args.q_heads, args.v_heads,
                               HEAD_DIM, device, dtype)
        # Match the serving stack: chunk.py passes return_internals=
        # SUPPRESS_LEVEL >= 3 (False by default), so A_inv/w/v_new casts are
        # skipped. Use --suppress-level 3 to benchmark the full-return path.
        return_internals = args.suppress_level >= 3
        run_opt = make_run_optimized(args.clone_cu, return_internals)

        if not args.skip_correctness:
            # Verify via the full-return path so dtypes/shapes match upstream
            # exactly; the serving path (return_internals=False) only skips
            # casts of dead outputs and cannot change numerics.
            print("  correctness (upstream vs optimized, full-return path):")
            clear_gdn_meta_cache()
            up = run_upstream(inp)
            sync()
            clear_gdn_meta_cache()
            set_gdn_meta_cache(True)
            run_opt_full = make_run_optimized(args.clone_cu, True)
            opt = run_opt_full(inp)
            sync()
            errs = check_outputs(up, opt)
            del up, opt, run_opt_full
            if errs:
                raise AssertionError("optimized diverged:\n" + "\n".join(errs))

        # host-only overhead per layer
        fn_up = make_multilayer_fn(run_upstream, inp, args.layers)
        host_up = bench_host(fn_up, args.layers, args.active)

        clear_gdn_meta_cache()
        set_gdn_meta_cache(True)
        fn_opt = make_multilayer_fn(run_opt, inp, args.layers)
        fn_opt()  # fill slab + caches before timing
        sync()
        host_opt = bench_host(fn_opt, args.layers, args.active)

        # full xL wall (kernel included, one sync per xL)
        wall_up = bench_wall(fn_up, args.warmup, args.active)
        wall_opt = bench_wall(fn_opt, args.warmup, args.active)
        assert is_gdn_meta_cache_enabled()

        # allocator-level HBM accounting: churn = bytes the caching allocator
        # hands out across ONE whole forward (all layers), i.e. the traffic
        # this change removes. Input tensors are live for both windows, so the
        # base-vs-opt delta is unaffected by them.
        mem_up = bench_memory(fn_up)
        mem_opt = bench_memory(fn_opt)

        print(f"  {'host/layer':<24} upstream={fmt_us(host_up):>10}  "
              f"optimized={fmt_us(host_opt):>10}  saved={fmt_us(host_up - host_opt)}  "
              f"({host_up / max(host_opt, 1e-9):.2f}x)")
        print(f"  {'xL wall (kernel incl.)':<24} upstream={fmt_us(wall_up['mean_us']):>10}  "
              f"optimized={fmt_us(wall_opt['mean_us']):>10}  "
              f"saved={fmt_us(wall_up['mean_us'] - wall_opt['mean_us'])}  "
              f"({wall_up['mean_us'] / max(wall_opt['mean_us'], 1e-9):.2f}x)  "
              f"[min: {fmt_us(wall_up['min_us'])} / {fmt_us(wall_opt['min_us'])}]")
        if mem_up["churn_gb"] is not None:
            print(f"  {'churn/forward (meas.)':<24} upstream={mem_up['churn_gb']:>8.2f} GB  "
                  f"optimized={mem_opt['churn_gb']:>8.2f} GB  "
                  f"saved={mem_up['churn_gb'] - mem_opt['churn_gb']:.2f} GB  "
                  f"(allocs: {mem_up['allocs']:.0f} -> {mem_opt['allocs']:.0f}/forward)")
        print(f"  {'peak allocated':<24} upstream={mem_up['peak_alloc_gb']:>8.2f} GB  "
              f"optimized={mem_opt['peak_alloc_gb']:>8.2f} GB")
        print(f"  {'peak reserved':<24} upstream={mem_up['peak_res_gb']:>8.2f} GB  "
              f"optimized={mem_opt['peak_res_gb']:>8.2f} GB")
        rows.append({"tokens": seq, "host_us_up": host_up, "host_us_opt": host_opt,
                     "wall_us_up": wall_up["mean_us"], "wall_us_opt": wall_opt["mean_us"],
                     "alloc_gb_up": total_up / GB, "alloc_gb_opt": per_call_opt / GB,
                     "mem_up": mem_up, "mem_opt": mem_opt})

        del inp
        clear_gdn_meta_cache()

    if rows:
        print(f"\n{'=' * 78}\nsummary")
        for r in rows:
            print(f"  tokens={r['tokens']:>7}  alloc/layer {r['alloc_gb_up']:.2f} GB -> "
                  f"{r['alloc_gb_opt']:.2f} GB  host/layer {fmt_us(r['host_us_up']):>9} -> "
                  f"{fmt_us(r['host_us_opt']):>9}  xL wall {fmt_us(r['wall_us_up']):>9} -> "
                  f"{fmt_us(r['wall_us_opt']):>9}")
        if all(r["mem_up"]["churn_gb"] is not None for r in rows):
            print(f"\n{'=' * 78}\nPR-ready HBM table (allocator-measured):\n")
            print("| tokens/forward | churn upstream | churn optimized | saved | "
                  "peak alloc up/opt | peak reserved up/opt |")
            print("| ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in rows:
                mu, mo = r["mem_up"], r["mem_opt"]
                print(f"| {r['tokens']} | {mu['churn_gb']:.1f} GB | "
                      f"{mo['churn_gb']:.1f} GB | "
                      f"{mu['churn_gb'] - mo['churn_gb']:.1f} GB | "
                      f"{mu['peak_alloc_gb']:.2f} / {mo['peak_alloc_gb']:.2f} GB | "
                      f"{mu['peak_res_gb']:.2f} / {mo['peak_res_gb']:.2f} GB |")
    print(
        """
How to read:
  - host/layer: allocator churn + torch.zeros memsets + D2H + launch churn.
    The optimized variant should collapse to ~constant (cache hits).
  - xL wall includes the AscendC kernel; on long sequences the kernel dilutes
    the host saving, so compare wall at small tokens and host at large tokens.
  - env toggles: SGLANG_NPU_GDN_META_CACHE=0, SGLANG_NPU_GDN_POOL_OUTPUTS=0,
    SGLANG_NPU_GDN_SKIP_OUTPUT_ZERO=0 (restore zero-init for isolation).
"""
    )


if __name__ == "__main__":
    main()

"""Tuning sweep for the fused shared-expert GEMM stage (F1 diagnostics, NPU only).

Round-trips with the NPU box are expensive, so this script explores the whole
tile/loop space in ONE run. All timings are in-graph device time (NPU graph
capture around back-to-back calls) — the production-relevant metric; eager
wall time on this host is dominated by launch overhead.

Part 0  baselines: torch MatMulV2 for the K1 shape, full eager chain.
Part 1  probe: minimal 1-dot GEMM shaped like K1's gate_up projection
        ([32, 2048] x [2048, 1024]) swept over BLOCK_K x BLOCK_OUT x
        num_warps x num_stages x weight layout (row-major [N, K] strided
        load vs pre-transposed [K, N] contiguous).
Part 2  persistent/grid-by-core probe: fixed program count (NP), each walks
        its share of n-blocks.
Part 3  real autotuned kernels: K1 / K2 / K1+K2 in-graph time.
Part 4  mechanism probes: dots-per-iter effect, SwiGLU epilogue cost, and the
        K2 GEMM shape floor ([32, HALF] x [HALF, N]).

Usage: python benchmark/fused_shared_expert_tune.py [--quick]
"""

import argparse
import time

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

import triton
import triton.language as tl

IN, HALF, N = 2048, 512, 2048  # 35B-A3B dims
WGU_MB = 2 * HALF * IN * 2 / 1e6


@triton.jit
def _probe_gemm(h_ptr, w_ptr, wt_ptr, o_ptr, M,
                K: tl.constexpr, Nn: tl.constexpr,
                MB: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                NS: tl.constexpr, W_T: tl.constexpr,
                PERS: tl.constexpr, NP: tl.constexpr,
                NDOTS: tl.constexpr, EPI: tl.constexpr):
    pid = tl.program_id(0)
    num_n = Nn // BN
    if PERS:
        # grid = NP programs (core-count tiling); each walks n-blocks with
        # stride NP, m-block 0 only (sweep uses M <= MB).
        offs_m = tl.arange(0, MB)
        mm = offs_m < M
        for nb in range(pid, num_n, NP):
            acc = tl.zeros((MB, BN), dtype=tl.float32)
            offs_n = nb * BN + tl.arange(0, BN)
            acc = _dot_loop(h_ptr, w_ptr, wt_ptr, acc, offs_m, offs_n, mm,
                            K, Nn, BK, NS, W_T, NDOTS)
            if EPI:
                xe = acc.to(tl.bfloat16).to(tl.float32)
                acc = xe * tl.sigmoid(xe)
            tl.store(o_ptr + offs_m[:, None] * Nn + offs_n[None, :],
                     acc.to(tl.bfloat16), mask=mm[:, None])
    else:
        mb = pid // num_n
        nb = pid % num_n
        offs_m = mb * MB + tl.arange(0, MB)
        mm = offs_m < M
        offs_n = nb * BN + tl.arange(0, BN)
        acc = tl.zeros((MB, BN), dtype=tl.float32)
        acc = _dot_loop(h_ptr, w_ptr, wt_ptr, acc, offs_m, offs_n, mm,
                        K, Nn, BK, NS, W_T, NDOTS)
        if EPI:
            xe = acc.to(tl.bfloat16).to(tl.float32)
            acc = xe * tl.sigmoid(xe)
        tl.store(o_ptr + offs_m[:, None] * Nn + offs_n[None, :],
                 acc.to(tl.bfloat16), mask=mm[:, None])


@triton.jit
def _dot_loop(h_ptr, w_ptr, wt_ptr, acc, offs_m, offs_n, mm,
              K: tl.constexpr, Nn: tl.constexpr,
              BK: tl.constexpr, NS: tl.constexpr, W_T: tl.constexpr,
              NDOTS: tl.constexpr):
    for k0 in tl.range(0, K, BK, num_stages=NS):
        offs_k = k0 + tl.arange(0, BK)
        h = tl.load(h_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=mm[:, None], other=0.0)
        if W_T:
            w = tl.load(wt_ptr + offs_k[:, None] * Nn + offs_n[None, :])
        else:
            w = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None])
        acc += tl.dot(h, w)
        if NDOTS >= 2:
            acc += tl.dot(h, w)
        if NDOTS >= 3:
            acc += tl.dot(h, w)
    return acc


def graph_us(fn, calls=10, replays=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(g):
            for _ in range(calls):
                fn()
    except Exception as e:
        print(f"    capture failed: {type(e).__name__}: {e}")
        return None
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(replays):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / (calls * replays) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip part 2/3, small part 1 (smoke run)")
    args = ap.parse_args()

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise SystemExit("NPU not available")
    print(f"device: {torch.npu.get_device_name(0)}")

    torch.manual_seed(0)
    M = 32
    h = torch.randn(M, IN).to(torch.bfloat16).npu()
    wgu = (torch.randn(2 * HALF, IN) / IN**0.5).to(torch.bfloat16).npu()
    wgu_t = wgu.t().contiguous()
    o = torch.empty((M, 2 * HALF), dtype=torch.bfloat16, device="npu")
    wgp = torch.zeros((16, IN), dtype=torch.bfloat16, device="npu")
    inter = torch.empty((M, HALF), dtype=torch.bfloat16, device="npu")
    gate = torch.empty((M,), dtype=torch.bfloat16, device="npu")

    print("\n== Part 0: baselines (in-graph us, weight GB/s) ==")
    t = graph_us(lambda: torch.matmul(h, wgu.t()))
    print(f"  matmul [32,{IN}]x[{IN},{2*HALF}] : {t:.2f}us  {WGU_MB/1e3/(t*1e-6):.0f} GB/s")

    import torch_npu as _tn
    from sgl_kernel_npu.activation.fused_sigmoid_mul import fused_sigmoid_mul_broadcast
    wd = (torch.randn(N, HALF) / HALF**0.5).to(torch.bfloat16).npu()
    wg = (torch.randn(1, IN) / IN**0.5).to(torch.bfloat16).npu()

    def eager():
        gu = torch.matmul(h, wgu.t())
        i = _tn.npu_swiglu(gu)
        d = torch.matmul(i, wd.t())
        g = torch.matmul(h, wg.t())
        return fused_sigmoid_mul_broadcast(d, g)

    t = graph_us(eager)
    print(f"  eager chain (5 ops)              : {t:.2f}us")

    print("\n== Part 1: probe GEMM sweep (1 dot/iter, M=32, gate_up shape) ==")
    print("    BK  BN  NW  NS  W_T   us     GB/s")
    rows = []
    bks = (64, 128, 256) if not args.quick else (64, 256)
    for BK in bks:
        for BN in (16, 64, 128):
            for NW in (4, 8):
                for NS in (1, 3):
                    for W_T in (False, True):
                        grid = (triton.cdiv(M, 32) * (2 * HALF // BN),)
                        fn = lambda: _probe_gemm[grid](
                            h, wgu, wgu_t, o, M,
                            K=IN, Nn=2 * HALF, MB=32, BK=BK, BN=BN,
                            NS=NS, W_T=W_T, PERS=0, NP=0, NDOTS=1, EPI=0,
                            num_warps=NW, multibuffer=True)
                        t = graph_us(fn)
                        if t is None:
                            continue
                        gbs = WGU_MB / 1e3 / (t * 1e-6)
                        rows.append((t, BK, BN, NW, NS, W_T))
                        print(f"    {BK:>3} {BN:>3} {NW:>3} {NS:>3} "
                              f"{'T' if W_T else 'F':>4} {t:>7.2f} {gbs:>7.0f}")
    if rows:
        rows.sort()
        print("  TOP 5:")
        for t, BK, BN, NW, NS, W_T in rows[:5]:
            print(f"    {t:>7.2f}us  BK={BK} BN={BN} NW={NW} NS={NS} W_T={W_T}")

    if not args.quick:
        print("\n== Part 2: persistent grid-by-core probe (BK=128, NW=8, NS=3) ==")
        print("    BN  NP   us     GB/s")
        for BN in (32, 64):
            for NPv in (20, 40, 80):
                grid = (NPv,)
                fn = lambda: _probe_gemm[grid](
                    h, wgu, wgu_t, o, M,
                    K=IN, Nn=2 * HALF, MB=32, BK=128, BN=BN,
                    NS=3, W_T=False, PERS=1, NP=NPv, NDOTS=1, EPI=0,
                    num_warps=8, multibuffer=True)
                t = graph_us(fn)
                if t is None:
                    continue
                print(f"    {BN:>3} {NPv:>3} {t:>7.2f} "
                      f"{WGU_MB/1e3/(t*1e-6):>7.0f}")

        print("\n== Part 3: real autotuned K1 / K2 / K1+K2 (in-graph us) ==")
        import sgl_kernel_npu.fused.fused_shared_expert as fs

        out_k2 = torch.empty((M, N), dtype=torch.bfloat16, device="npu")
        wpack = fs._pack_gate_up_weight(wgu, HALF)
        t_k1 = graph_us(lambda: fs._launch_k1(h, wpack, wgp, inter, gate, M, IN, HALF))
        print(f"    K1           : {t_k1 if t_k1 is None else round(t_k1, 2)}")
        t_k2 = graph_us(lambda: fs._launch_k2(inter, wd, gate, out_k2, None, M, HALF, N))
        print(f"    K2           : {t_k2 if t_k2 is None else round(t_k2, 2)}")
        t_all = graph_us(lambda: fs.fused_shared_expert_mlp(h, wgu, wd, wg, None))
        print(f"    K1+K2        : {t_all if t_all is None else round(t_all, 2)}")
        print(f"    chosen K1 cfg: {fs._fused_shared_expert_k1.best_config}")
        print(f"    chosen K2 cfg: {fs._fused_shared_expert_k2.best_config}")

        print("\n== Part 4: mechanism probes ==")
        wd_mb = N * HALF * 2 / 1e6

        def probe(Kd, Nd, w, wt, o_buf, BK, BN, NW, NS, ndots, epi, tag):
            grid = (triton.cdiv(M, 32) * (Nd // BN),)
            fn = lambda: _probe_gemm[grid](
                h, w, wt, o_buf, M,
                K=Kd, Nn=Nd, MB=32, BK=BK, BN=BN,
                NS=NS, W_T=False, PERS=0, NP=0, NDOTS=ndots, EPI=epi,
                num_warps=NW, multibuffer=True)
            t = graph_us(fn)
            gbs = (Nd * Kd * 2 / 1e6) / 1e3 / (t * 1e-6) if t else 0
            print(f"    {tag:<44} {t if t is None else round(t, 2)}  {gbs:.0f} GB/s")

        print("    -- dot-count effect on the K1 shape (BK=256, BN=64) --")
        probe(IN, 2 * HALF, wgu, wgu_t, o, 256, 64, 8, 2, 1, 0, "1 dot/iter (plain)")
        probe(IN, 2 * HALF, wgu, wgu_t, o, 256, 64, 8, 2, 2, 0, "2 dots/iter")
        probe(IN, 2 * HALF, wgu, wgu_t, o, 256, 64, 8, 2, 3, 0, "3 dots/iter (K1-like)")
        print("    -- epilogue effect (BK=256, BN=64) --")
        probe(IN, 2 * HALF, wgu, wgu_t, o, 256, 64, 8, 2, 1, 0, "no epilogue")
        probe(IN, 2 * HALF, wgu, wgu_t, o, 256, 64, 8, 2, 1, 1, "+ swiglu epilogue (sigmoid)")
        probe(IN, 2 * HALF, wgu, wgu_t, o, 256, 64, 8, 2, 3, 1, "3 dots + swiglu epilogue")
        print("    -- K2 GEMM shape floor: [32,512]x[512,2048], 2MB --")
        o_k2p = torch.empty((M, N), dtype=torch.bfloat16, device="npu")
        probe(HALF, N, wd, wd.t().contiguous(), o_k2p, 64, 64, 8, 2, 1, 0, "BK=64  BN=64")
        probe(HALF, N, wd, wd.t().contiguous(), o_k2p, 128, 64, 8, 2, 1, 0, "BK=128 BN=64")
        probe(HALF, N, wd, wd.t().contiguous(), o_k2p, 256, 64, 8, 2, 1, 0, "BK=256 BN=64")
        probe(HALF, N, wd, wd.t().contiguous(), o_k2p, 256, 128, 8, 2, 1, 0, "BK=256 BN=128")
        probe(HALF, N, wd, wd.t().contiguous(), o_k2p, 512, 32, 8, 2, 1, 0, "BK=512 BN=32")
        probe(HALF, N, wd, wd.t().contiguous(), o_k2p, 512, 64, 4, 1, 1, 0, "BK=512 BN=64 NS=1")


if __name__ == "__main__":
    main()

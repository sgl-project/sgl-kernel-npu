"""NPU correctness check for the sgl-kernel-npu W4A4 fused mega-MoE op.

Compares torch.ops.npu.mega_moe_w4a4 (via the sgl wrapper) against a pure-torch
W4A4 + block-diag-64 Hadamard MoE oracle.

Run: ASCEND_RT_VISIBLE_DEVICES=0 python tests/python/sgl_kernel_npu/test_mega_moe_w4a4_npu.py
"""

import math
import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../python/sgl_kernel_npu")
)

import sgl_kernel_npu  # noqa: F401  (loads libsgl_kernel_npu.so -> torch.ops.npu)
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.moe.mega_moe_w4a4 import (
    mega_moe_forward,
    pack_nz_int4,
    pack_scale_uint64,
    routing_prep,
)

DEV = "npu:0"


def _hadamard64(dev, dtype):
    n = 64
    rows = [[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)]
    return (torch.tensor(rows, dtype=torch.float32, device=dev) / math.sqrt(n)).to(
        dtype
    )


def _block_hadamard(x, h64):
    m, k = x.shape
    return (x.reshape(m, k // 64, 64) @ h64).reshape(m, k)


def _int4_fakequant_per_token(x):
    s = (x.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-6)
    return ((x / s).round().clamp_(-8, 7)) * s


def w4a4_moe_torch(x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights):
    T, H = x.shape
    E, twoI, _ = w13.shape
    I = twoI // 2
    K = topk_ids.shape[1]
    h64 = _hadamard64(x.device, x.dtype)
    xr = _block_hadamard(x, h64)
    w13f = w13.to(torch.float32) * w13_scale.unsqueeze(-1)
    w2f = w2.to(torch.float32) * w2_scale.unsqueeze(-1)
    flat_x = xr.unsqueeze(1).expand(T, K, H).reshape(T * K, H)
    flat_e = topk_ids.reshape(-1)
    y = torch.zeros(T * K, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        idx = (flat_e == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xe = _int4_fakequant_per_token(flat_x[idx]).to(torch.float32)
        gu = xe @ w13f[e].t()
        sw = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        iq = _int4_fakequant_per_token(sw.to(x.dtype)).to(torch.float32)
        y[idx] = iq @ w2f[e].t()
    y = (y * topk_weights.reshape(-1, 1).to(torch.float32)).reshape(T, K, H).sum(1)
    return y.to(x.dtype)


def run(E, T, K, seed=0):
    H, I = 2048, 128  # kernel is compiled for these per-rank dims
    torch.manual_seed(seed)
    x = torch.randn(T, H, dtype=torch.float16, device=DEV) * 0.5
    w13 = torch.randint(-8, 8, (E, 2 * I, H), dtype=torch.int8, device=DEV)
    w2 = torch.randint(-8, 8, (E, H, I), dtype=torch.int8, device=DEV)
    w13s = (torch.rand(E, 2 * I, device=DEV) * 0.02 + 0.005).to(torch.float32)
    w2s = (torch.rand(E, H, device=DEV) * 0.02 + 0.005).to(torch.float32)
    rnd = torch.rand(T, E, device=DEV)
    topk_ids = rnd.topk(K, dim=1).indices.to(torch.int32)
    topk_w = torch.softmax(torch.randn(T, K, device=DEV), dim=1).to(torch.float16)

    torch.npu.synchronize()
    t0 = time.time()
    y_ref = w4a4_moe_torch(x, w13, w13s, w2, w2s, topk_ids, topk_w)
    torch.npu.synchronize()
    t_ref = time.time() - t0

    w13_nz = pack_nz_int4(w13.transpose(1, 2).contiguous())  # [E,H,2I] -> NZ
    w2_nz = pack_nz_int4(w2.transpose(1, 2).contiguous())  # [E,I,H] -> NZ
    w13s_u64 = pack_scale_uint64(w13s)
    w2s_u64 = pack_scale_uint64(w2s)
    gl, eri, sort_idx = routing_prep(topk_ids, E)

    torch.npu.synchronize()
    t0 = time.time()
    y_k = mega_moe_forward(
        x,
        w13_nz,
        w13s_u64,
        w2_nz,
        w2s_u64,
        gl,
        eri,
        sort_idx,
        topk_w,
        top_k=K,
        H=H,
        i_dim=I,
        n_gu=2 * I,
    ).clone()
    torch.npu.synchronize()
    t_k = time.time() - t0

    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten(), y_k.float().flatten(), dim=0
    ).item()
    print(
        f"[E={E} T={T} top_k={K}] torch={t_ref*1000:.1f}ms mega={t_k*1000:.1f}ms "
        f"finite={torch.isfinite(y_k).all().item()} nonzero={(y_k != 0).any().item()} "
        f"cos={cos:.4f}"
    )
    assert torch.isfinite(y_k).all(), "mega output contains NaN/Inf"
    assert (
        cos >= 0.99
    ), f"cosine similarity {cos:.4f} below 0.99 (expected ~0.999 vs torch reference)"
    return cos


if __name__ == "__main__":
    print("=== small smoke ===")
    run(E=16, T=8, K=2)
    print("=== production-like ===")
    run(E=256, T=64, K=8)

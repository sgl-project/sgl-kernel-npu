# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Host orchestration for the fused W4A4 INT4 MoE "mega" kernel (Qwen3.x-MoE, Ascend 910B).
# One launch of torch.ops.npu.mega_moe_w4a4 runs the whole routed-expert path. This mirrors
# the vllm-ascend JIT loader (vllm_ascend/ops/mega_moe_w4a4.py) but targets the build-time
# kernel registered by libsgl_kernel_npu.so: tilings / workspaces / weight repack / scale
# pack are prepared here, and the op_host does only shape checks + EXEC_KERNEL_CMD.

import math
from functools import lru_cache

import numpy as np
import torch

# Per-rank shapes the kernel is COMPILED for (see csrc/.../op_kernel: H_DIM, I_DIM_OVERRIDE).
KERNEL_H_DIM = 2048
KERNEL_I_DIM = 128
HADAMARD_BLOCK_SIZE = 64

_BLOCK_DIM = 24  # 910B (A2) cube core count
_M_TILE = 32  # cube writes in 32-row chunks; pad workspaces to this
_M_TILE_GU = 128  # gate_up cube baseM (== singleCoreM)
_M_TILE_DN = 64  # down cube baseM


def routing_prep(topk_ids: torch.Tensor, num_experts: int):
    """Sort-by-expert permutation, replacing ``npu_moe_init_routing_v2``.

    Returns ``(group_list, expanded_row_idx, sort_idx)``:
      * ``group_list`` [E] int64 — cumulative per-expert token counts.
      * ``expanded_row_idx`` [M] int32 — vendor convention; for original flat index
        ``i`` it is the expanded slot token ``i`` landed at (combine stage).
      * ``sort_idx`` [M] int32 — inverse: expanded slot -> original flat index
        (the quant stage reads ``x[sort_idx[m] // top_k]``).

    Done on fp32 so both argsorts stay on AICore (int sort falls back to AICpu).
    """
    flat = topk_ids.reshape(-1).to(torch.float32)
    sorted_ids, sort_idx = torch.sort(flat)
    eri = torch.argsort(sort_idx.to(torch.float32))
    expert_ids = torch.arange(num_experts, device=topk_ids.device, dtype=torch.float32)
    group_list = torch.searchsorted(sorted_ids, expert_ids, right=True).to(torch.int64)
    return group_list, eri.to(torch.int32), sort_idx.to(torch.int32)


def pack_nz_int4(w_kn: torch.Tensor) -> torch.Tensor:
    """``[E, K, N]`` int8 (one nibble per byte, range [-8, 7]) -> flat FRACTAL_NZ
    int4 bytes, the layout the AscendC ``CubeFormat::NZ`` int4 B matrix expects:
    ``[E, N/64, K/16, 16, 32]`` packed along the inner N0=64."""
    E, K, N = w_kn.shape
    assert K % 16 == 0 and N % 64 == 0, f"NZ pack needs K%16==0, N%64==0 (K={K}, N={N})"
    r = w_kn.reshape(E, K // 16, 16, N // 64, 64).permute(0, 3, 1, 2, 4).contiguous()
    lo = r[..., 0::2].to(torch.int32) & 0xF
    hi = r[..., 1::2].to(torch.int32) & 0xF
    return ((hi << 4) | lo).to(torch.int8).reshape(-1).contiguous()


# TCubeTiling: 50 int32 fields, in AscendC kernel_tiling.h order.
_TILING_FIELDS = [
    "usedCoreNum",
    "M",
    "N",
    "Ka",
    "Kb",
    "singleCoreM",
    "singleCoreN",
    "singleCoreK",
    "baseM",
    "baseN",
    "baseK",
    "depthA1",
    "depthB1",
    "stepM",
    "stepN",
    "isBias",
    "transLength",
    "iterateOrder",
    "shareMode",
    "shareL1Size",
    "shareL0CSize",
    "shareUbSize",
    "batchM",
    "batchN",
    "singleBatchM",
    "singleBatchN",
    "stepKa",
    "stepKb",
    "depthAL1CacheUB",
    "depthBL1CacheUB",
    "dbL0A",
    "dbL0B",
    "dbL0C",
    "ALayoutInfoB",
    "ALayoutInfoS",
    "ALayoutInfoN",
    "ALayoutInfoG",
    "ALayoutInfoD",
    "BLayoutInfoB",
    "BLayoutInfoS",
    "BLayoutInfoN",
    "BLayoutInfoG",
    "BLayoutInfoD",
    "CLayoutInfoB",
    "CLayoutInfoS1",
    "CLayoutInfoN",
    "CLayoutInfoG",
    "CLayoutInfoS2",
    "BatchNum",
    "mxTypePara",
]


def _make_tiling(m_tile: int, K: int, N: int, n_tile: int, base_k: int, base_m: int):
    """Build a 50-int32 TCubeTiling array. baseM MUST equal singleCoreM (=m_tile):
    the int4 ``MatmulImpl`` corrupts when baseM < singleCoreM."""
    base_k = min(base_k, K)
    n_ka = (K + base_k - 1) // base_k
    t = dict.fromkeys(_TILING_FIELDS, 0)
    t["usedCoreNum"] = _BLOCK_DIM
    t["M"], t["N"], t["Ka"], t["Kb"] = m_tile, N, K, K
    t["singleCoreM"], t["singleCoreN"], t["singleCoreK"] = m_tile, n_tile, K
    t["baseM"], t["baseN"], t["baseK"] = max(16, base_m), n_tile, base_k
    t["depthA1"], t["depthB1"] = n_ka, n_ka
    t["stepM"], t["stepN"] = 1, 1
    t["stepKa"], t["stepKb"] = n_ka, n_ka
    t["dbL0A"], t["dbL0B"], t["dbL0C"] = 2, 2, 1
    return np.array([t[f] for f in _TILING_FIELDS], dtype=np.int32)


@lru_cache(maxsize=8)
def _get_tilings(device_key, H: int, i_dim: int, n_gu: int):
    """``(tiling_gu, tiling_dn)`` int32 device tensors. gate_up: K=H, N=n_gu.
    down: K=i_dim, N=H (512-tiled). Cached per layer shape."""
    device = torch.device(device_key[0], device_key[1])
    tg = _make_tiling(_M_TILE_GU, H, n_gu, n_gu, base_k=256, base_m=_M_TILE_GU)
    td = _make_tiling(
        _M_TILE_DN, i_dim, H, min(512, H), base_k=min(256, i_dim), base_m=_M_TILE_DN
    )
    tg_t = torch.from_numpy(tg).contiguous().to(device)
    td_t = torch.from_numpy(td).contiguous().to(device)
    return tg_t, td_t


@lru_cache(maxsize=8)
def _get_b1(device_key, H: int) -> torch.Tensor:
    """Stage-1 Hadamard weight: normalized 64x64 Walsh-Hadamard replicated H/64
    times, ``[H, 64]`` fp16. Generated at load (fixed matrix, not in the checkpoint).
    The Sylvester matrix is symmetric, so it equals its own DN-transposed block."""
    device = torch.device(device_key[0], device_key[1])
    n = HADAMARD_BLOCK_SIZE
    rows = [[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)]
    hn = torch.tensor(rows, dtype=torch.float32) / math.sqrt(n)
    return hn.repeat(H // n, 1).to(torch.float16).contiguous().to(device)


def pack_scale_uint64(scale: torch.Tensor) -> torch.Tensor:
    """Per-channel fp32 scale -> uint64 (fp32 bits in the low 32) for the cube FIXPIPE
    ``SetQuantVector`` dequant. Represented as an int64 tensor (torch has no uint64).
    Must be materialized before the cube reads it (cold launch reads zero otherwise)."""
    f32 = scale.to(torch.float32).contiguous()
    u64 = (f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).contiguous()
    torch.npu.synchronize()
    return u64


def mega_moe_forward(
    x: torch.Tensor,
    w13_nz: torch.Tensor,
    w13_scale_u64: torch.Tensor,
    w2_nz: torch.Tensor,
    w2_scale_u64: torch.Tensor,
    group_list: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    sort_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    H: int = KERNEL_H_DIM,
    i_dim: int = KERNEL_I_DIM,
    n_gu: int = None,
) -> torch.Tensor:
    """One fused launch of the full expert path. ``x`` is the UN-permuted ``[T, H]`` fp16
    input (raw routed activations; the kernel applies the block-diagonal Hadamard on the
    cube and scatters by ``sort_idx``). Weights are FRACTAL_NZ int4 with uint64-packed
    scales. Returns ``[T, H]`` fp16."""
    assert H == KERNEL_H_DIM and i_dim == KERNEL_I_DIM, (
        f"kernel is compiled for H={KERNEL_H_DIM}, I={KERNEL_I_DIM}; got H={H}, I={i_dim}. "
        "Use SGLANG_NPU_MEGA_MOE_W4A4=0 for the vendor-separated path."
    )
    n_gu = n_gu if n_gu is not None else 2 * i_dim
    device = x.device
    T_orig = x.shape[0]
    E = int(group_list.shape[0])
    M_total = expanded_row_idx.shape[0]
    m_pad = ((M_total + _M_TILE - 1) // _M_TILE) * _M_TILE

    h_act = H // 2  # int4-packed activation bytes
    i_act = i_dim // 2
    xq_ws = torch.empty(m_pad, h_act, dtype=torch.int8, device=device)
    xs_ws = torch.empty(m_pad * 32, dtype=torch.float32, device=device)
    gu_ws = torch.empty(m_pad, n_gu, dtype=torch.float16, device=device)
    iq_ws = torch.empty(m_pad, i_act, dtype=torch.int8, device=device)
    is_ws = torch.empty(m_pad * 32, dtype=torch.float32, device=device)
    d_ws = torch.empty(m_pad, H, dtype=torch.float16, device=device)
    xrot_ws = torch.empty(T_orig, H, dtype=torch.float16, device=device)
    y = torch.zeros(
        T_orig, H, dtype=torch.float16, device=device
    )  # combine atomic-adds

    device_key = (device.type, 0 if device.index is None else device.index)
    tiling_gu, tiling_dn = _get_tilings(device_key, H, i_dim, n_gu)
    b1 = _get_b1(device_key, H)

    torch.ops.npu.mega_moe_w4a4(
        x,
        w13_nz,
        w13_scale_u64,
        w2_nz,
        w2_scale_u64,
        group_list,
        expanded_row_idx,
        sort_idx,
        topk_weights,
        xq_ws,
        xs_ws,
        gu_ws,
        iq_ws,
        is_ws,
        d_ws,
        y,
        tiling_gu,
        tiling_dn,
        b1,
        xrot_ws,
        M_total,
        E,
        top_k,
        T_orig,
        _BLOCK_DIM,
    )
    return y

"""Triton-Ascend fused input builder for the LPLB interior-point solver.

The GPU counterpart in ``sglang/kernels/jit/csrc/lplb/lp_prep.cuh`` replaces
eight torch ops with one launch:

    total       = global_counts.sum()
    counts_norm = global_counts / total.clamp(min=1.0)
    t1          = counts_norm[log_single]                  # (NUM_SINGLE,)
    b1          = counts_norm[log_replicated]              # (NUM_RED_LOG,)
    b2          = -(B1 @ t1)                               # (NUM_GPUS,)
    b           = cat(b1, b2)                              # (NC,)
    A_full[:, -1] = b - A_base_row_sum

``A_full`` is pre-allocated by the caller; its first ``NV - 1`` columns already
hold ``A_base`` and are not touched here. ``b`` and ``t1`` are overwritten in
place, so the caller can reuse long-lived buffers across calls.

The kernel keeps the same single-program (``grid=(1,)``) shape as the CUDA
implementation and performs every reduction inside the vector unit.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def lp_prep_kernel(
    A_full_ptr,
    b_ptr,
    t1_ptr,
    global_counts_ptr,
    log_single_ptr,
    log_replicated_ptr,
    B1_ptr,
    A_base_row_sum_ptr,
    NUM_SINGLE: tl.constexpr,
    NUM_RED_LOG: tl.constexpr,
    NUM_GPUS: tl.constexpr,
    NV: tl.constexpr,
    NUM_LOGICAL: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
    BT: tl.constexpr,
    GCHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    SCHUNK: tl.constexpr,
):
    # ---------------- Stage 1: gather counts, total, inv_total ----------------
    o_s = tl.arange(0, BS)
    m_s = o_s < NUM_SINGLE
    o_r = tl.arange(0, BR)
    m_r = o_r < NUM_RED_LOG

    # int64 logical ids -> int32 before entering the vector path.
    idx_s = tl.load(log_single_ptr + o_s, mask=m_s, other=0).to(tl.int32)
    idx_r = tl.load(log_replicated_ptr + o_r, mask=m_r, other=0).to(tl.int32)

    # The count table is small: load it contiguously into UB once, then select
    # with in-UB gather instead of random global reads.
    o_t = tl.arange(0, BT)
    m_t = o_t < NUM_LOGICAL
    counts_tab = tl.load(global_counts_ptr + o_t, mask=m_t, other=0.0)

    c_s = tl.gather(counts_tab, idx_s, 0)
    c_r = tl.gather(counts_tab, idx_r, 0)
    c_s = tl.where(m_s, c_s, 0.0)
    c_r = tl.where(m_r, c_r, 0.0)

    # Masked lanes contribute exactly 0 to the sums (padding never enters exp).
    sum_s = tl.sum(c_s, axis=0)
    sum_r = tl.sum(c_r, axis=0)
    total = sum_s + sum_r
    inv_total = 1.0 / tl.maximum(total, 1.0)

    # ---------------- Stage 2: scale t1 and b[:NUM_RED_LOG] ----------------
    t1_s = c_s * inv_total
    b1_s = c_r * inv_total

    tl.store(t1_ptr + o_s, t1_s, mask=m_s)
    tl.store(b_ptr + o_r, b1_s, mask=m_r)

    # ---------------- Stage 4a: A_full[:NUM_RED_LOG, NV-1] ----------------
    base_r = tl.load(A_base_row_sum_ptr + o_r, mask=m_r, other=0.0)
    tl.store(A_full_ptr + o_r * NV + (NV - 1), b1_s - base_r, mask=m_r)

    # ---------------- Stage 3 + 4b: b2 = -(B1 @ t1), GPU row chunks ----------------
    # Both the GPU rows and the ``NUM_SINGLE`` columns are chunked. Some CANN
    # versions pad/blockify a UB tile by up to 8x per element, and the tile,
    # its product and the reduction temp buffer can be live at the same time;
    # a full 16x256 FP32 tile then needs ~265 KiB UB and fails to compile.
    # Keeping both tile dimensions small bounds the padded footprint.
    o_g = tl.arange(0, GCHUNK)
    o_c = tl.arange(0, SCHUNK)
    n_col_chunks: tl.constexpr = (NUM_SINGLE + SCHUNK - 1) // SCHUNK
    for g0 in tl.static_range(0, NCHUNK):
        rows = g0 * GCHUNK + o_g
        m_g = rows < NUM_GPUS

        # Issue the row-sum load before the big B1 tile: it only depends on the
        # loop variable and hides latency behind the tile load.
        base_g = tl.load(A_base_row_sum_ptr + NUM_RED_LOG + rows, mask=m_g, other=0.0)

        part = tl.zeros([GCHUNK], tl.float32)
        for s0 in tl.static_range(0, n_col_chunks):
            cols = s0 * SCHUNK + o_c
            m_c = cols < NUM_SINGLE

            # Re-select this column chunk from the in-UB count table. Padded
            # columns contribute exactly 0 to the row dot.
            idx_c = tl.load(log_single_ptr + cols, mask=m_c, other=0).to(tl.int32)
            c_c = tl.gather(counts_tab, idx_c, 0)
            t_c = tl.where(m_c, c_c, 0.0) * inv_total

            b1_tile = tl.load(
                B1_ptr + rows[:, None] * NUM_SINGLE + cols[None, :],
                mask=m_g[:, None] & m_c[None, :],
                other=0.0,
            )
            part += tl.sum(b1_tile * t_c[None, :], axis=1)

        b2 = -part

        tl.store(b_ptr + NUM_RED_LOG + rows, b2, mask=m_g)

        tl.store(
            A_full_ptr + (NUM_RED_LOG + rows) * NV + (NV - 1),
            b2 - base_g,
            mask=m_g,
        )


def _next_pow2(n, floor=16):
    """Smallest power of two >= n, floored at ``floor``."""
    p = floor
    while p < n:
        p *= 2
    return p


def lp_prep(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
):
    """Build the IPM inputs in place and return ``(A_full, b, t1)``.

    All tensors must be contiguous float32 NPU tensors, except the two index
    vectors which may be int32 or int64. ``A_full`` must be 2-D ``(NC, NV)``
    with ``NC == log_replicated.numel() + B1.shape[0]``; only its last column
    is written.
    """
    if A_full.dim() != 2 or A_full.dtype != torch.float32:
        raise TypeError("A_full must be a 2-D float32 tensor")
    if b.dim() != 1 or b.dtype != torch.float32:
        raise TypeError("b must be a 1-D float32 tensor")
    if t1.dim() != 1 or t1.dtype != torch.float32:
        raise TypeError("t1 must be a 1-D float32 tensor")
    if global_counts.dim() != 1 or global_counts.dtype != torch.float32:
        raise TypeError("global_counts must be a 1-D float32 tensor")
    if log_single.dim() != 1 or log_single.dtype not in (torch.int32, torch.int64):
        raise TypeError("log_single must be a 1-D int32/int64 tensor")
    if log_replicated.dim() != 1 or log_replicated.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("log_replicated must be a 1-D int32/int64 tensor")
    if B1.dim() != 2 or B1.dtype != torch.float32:
        raise TypeError("B1 must be a 2-D float32 tensor")
    if A_base_row_sum.dim() != 1 or A_base_row_sum.dtype != torch.float32:
        raise TypeError("A_base_row_sum must be a 1-D float32 tensor")

    for name, tensor in (
        ("A_full", A_full),
        ("b", b),
        ("t1", t1),
        ("global_counts", global_counts),
        ("log_single", log_single),
        ("log_replicated", log_replicated),
        ("B1", B1),
        ("A_base_row_sum", A_base_row_sum),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    nc, nv = A_full.shape
    num_single = t1.numel()
    num_red_log = log_replicated.numel()
    num_gpus, b1_single = B1.shape
    num_logical = global_counts.numel()
    if num_logical == 0:
        raise ValueError("global_counts must not be empty")
    if b1_single != num_single:
        raise ValueError(
            f"B1 must have shape [num_gpus, num_single], got {(num_gpus, b1_single)} "
            f"and num_single={num_single}"
        )
    if b.shape != (nc,):
        raise ValueError(f"b must have shape {(nc,)}, got {tuple(b.shape)}")
    if A_base_row_sum.shape != (nc,):
        raise ValueError(
            f"A_base_row_sum must have shape {(nc,)}, "
            f"got {tuple(A_base_row_sum.shape)}"
        )
    if nc != num_red_log + num_gpus:
        raise ValueError(
            f"NC must equal num_red_log + num_gpus, got {nc} != "
            f"{num_red_log} + {num_gpus}"
        )
    if nv == 0:
        raise ValueError("A_full must have at least one column")

    BS = _next_pow2(max(16, num_single))
    BR = _next_pow2(max(16, num_red_log))
    BT = _next_pow2(max(16, num_logical))

    # Size the B1 tile so that the tile, its product and the reduction temp
    # buffer stay inside the UB even on toolchains that pad/blockify a UB tile
    # by 8x per element (observed with some CANN versions on Ascend 910). A
    # full (GCHUNK, BS) tile needs ~265 KiB there and fails to compile. Keep
    # the nominal FP32 tile at <= 2048 elements and the column chunk <= 64.
    SCHUNK = min(BS, 64)
    GCHUNK = _next_pow2(max(16, num_gpus))
    if GCHUNK * SCHUNK > 2048:
        GCHUNK = max(16, 2048 // SCHUNK)
    NCHUNK = triton.cdiv(num_gpus, GCHUNK)

    lp_prep_kernel[(1,)](
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
        NUM_SINGLE=num_single,
        NUM_RED_LOG=num_red_log,
        NUM_GPUS=num_gpus,
        NV=nv,
        NUM_LOGICAL=num_logical,
        BS=BS,
        BR=BR,
        BT=BT,
        GCHUNK=GCHUNK,
        NCHUNK=NCHUNK,
        SCHUNK=SCHUNK,
    )
    return A_full, b, t1

"""Triton-Ascend implementation of the LPLB ``lp_post`` operator.

The GPU counterpart in ``sglang/kernels/jit/csrc/lplb/lp_post.cuh`` is a single
block that runs four stages serially:

    Stage 1  phy_prob[:] = 0
    Stage 2  phy_prob[phy_replicated] = max(x, 0)
    Stage 3  phy_prob[phy_single] = t1
    Stage 4  out[i] = phy_prob[wrap(log2phy[i])]

Design notes specific to Ascend:

* **No global scatter.** On Ascend, ``tl.store(ptr + idx_vec)`` lowers to a
  per-lane scalar path, so stages 1-3 are replaced by a *compare-sum inverse*:
  every physical slot sums the values of the source lanes that point at it.
  Because ``phy_single`` and ``phy_replicated`` are disjoint, the result is
  bit-identical to a sequential scatter and has no write conflict at all.
* **Two kernels.** The table must be visible to every output program and there
  is no grid-wide barrier, so it has to go through global memory.
* **Stage 4 uses a UB preload + ``tl.gather``**: the whole table is loaded into
  UB once per program and selected with the hardware gather path.
"""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties

# ``tl.arange`` requires a power-of-two length and the UB resident table is
# bounded by the vector unit's element limit.
_MAX_TABLE_ELEMENTS = 32768


@triton.jit
def build_tab_kernel(
    x_ptr,
    t1_ptr,
    ps_ptr,
    pr_ptr,
    tab_ptr,
    NR: tl.constexpr,
    NS: tl.constexpr,
    P: tl.constexpr,
    NCH: tl.constexpr,
    NC: tl.constexpr,
    PCH: tl.constexpr,
    SCH: tl.constexpr,
):
    """Equivalent of stages 1-3: build ``phy_prob`` (zero-init + two scatters).

    Each program owns ``PCH`` consecutive physical slots (contiguous write side)
    and accumulates the contributions of every source lane whose index matches
    the slot. Slots that no source points at keep the zero-initialised value,
    which also covers the ``-1`` sink slot ``P - 1``.
    """
    pid = tl.program_id(0).to(tl.int32)
    o_c = tl.arange(0, PCH)
    o_s = tl.arange(0, SCH)

    n_single_chunks: tl.constexpr = (NS + SCH - 1) // SCH
    n_red_chunks: tl.constexpr = (NR + SCH - 1) // SCH

    for c in tl.static_range(0, (NCH + NC - 1) // NC):
        ci = pid + c * NC
        slots = ci * PCH + o_c
        acc = tl.zeros([PCH], tl.float32)

        # --- source 1: phy_single / t1 (stage 3, no clamp) ---
        for j in tl.static_range(0, n_single_chunks):
            src = j * SCH + o_s
            ms = src < NS
            tv = tl.load(t1_ptr + src, mask=ms, other=0.0)
            si = tl.load(ps_ptr + src, mask=ms, other=-1).to(tl.int32)
            acc += tl.sum(
                tl.where(slots[:, None] == si[None, :], tv[None, :], 0.0), axis=1
            )

        # --- source 2: phy_replicated / max(x, 0) (stage 2) ---
        for j in tl.static_range(0, n_red_chunks):
            src = j * SCH + o_s
            mr = src < NR
            xv = tl.load(x_ptr + src, mask=mr, other=0.0)
            xv = tl.where(xv > 0.0, xv, 0.0)  # clamp(min=0)
            ri = tl.load(pr_ptr + src, mask=mr, other=-1).to(tl.int32)
            acc += tl.sum(
                tl.where(slots[:, None] == ri[None, :], xv[None, :], 0.0), axis=1
            )

        # Full coverage of [0, P): this replaces the explicit zero-init kernel.
        tl.store(tab_ptr + slots, acc, mask=(ci < NCH) & (slots < P))


@triton.jit
def gather_out_kernel(
    out_ptr,
    l2p_ptr,
    tab_ptr,
    P: tl.constexpr,
    W: tl.constexpr,
    TO: tl.constexpr,
    NB: tl.constexpr,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Stage 4: ``out[i] = phy_prob[wrap(log2phy[i])]`` via a UB gather."""
    pid = tl.program_id(0).to(tl.int32)

    # Whole table into UB once per program.
    o_p = tl.arange(0, W)
    tab = tl.load(tab_ptr + o_p, mask=o_p < P, other=0.0)

    o = tl.arange(0, BLOCK)
    for i in tl.static_range(0, (NB + NC - 1) // NC):
        blk = pid + i * NC
        oo = blk * BLOCK + o
        m = (blk < NB) & (oo < TO)
        k = tl.load(l2p_ptr + oo, mask=m, other=0).to(tl.int32)
        kw = tl.where(k < 0, k + P, k)  # -1 -> sink slot P-1 (always zero)
        tl.store(out_ptr + oo, tl.gather(tab, kw, 0), mask=m)


def _next_pow2(n, floor=16):
    p = floor
    while p < n:
        p *= 2
    return p


def lp_post(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
):
    """Overwrite ``log2phy_prob`` from the IPM output and return it.

    ``x`` holds the replica ratios (``x[:num_red_phy]`` of the IPM solution),
    ``t1`` the single-copy shares from ``lp_prep``. ``log2phy`` may contain
    ``-1`` for unused replicas; those entries read the always-zero sink slot.
    """
    if log2phy_prob.dim() != 2 or log2phy_prob.dtype != torch.float32:
        raise TypeError("log2phy_prob must be a 2-D float32 tensor")
    if log2phy.dim() != 2 or log2phy.dtype not in (torch.int32, torch.int64):
        raise TypeError("log2phy must be a 2-D int32/int64 tensor")
    if log2phy.shape != log2phy_prob.shape:
        raise ValueError(
            f"log2phy shape {tuple(log2phy.shape)} must match log2phy_prob "
            f"shape {tuple(log2phy_prob.shape)}"
        )
    if x.dim() != 1 or x.dtype != torch.float32:
        raise TypeError("x must be a 1-D float32 tensor")
    if t1.dim() != 1 or t1.dtype != torch.float32:
        raise TypeError("t1 must be a 1-D float32 tensor")
    if phy_single.dim() != 1 or phy_single.dtype not in (torch.int32, torch.int64):
        raise TypeError("phy_single must be a 1-D int32/int64 tensor")
    if phy_replicated.dim() != 1 or phy_replicated.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("phy_replicated must be a 1-D int32/int64 tensor")

    for name, tensor in (
        ("log2phy_prob", log2phy_prob),
        ("log2phy", log2phy),
        ("x", x),
        ("t1", t1),
        ("phy_single", phy_single),
        ("phy_replicated", phy_replicated),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    num_single = t1.numel()
    num_red_phy = x.numel()
    if phy_single.numel() != num_single:
        raise ValueError(
            f"phy_single must have {num_single} entries, got {phy_single.numel()}"
        )
    if phy_replicated.numel() != num_red_phy:
        raise ValueError(
            f"phy_replicated must have {num_red_phy} entries, "
            f"got {phy_replicated.numel()}"
        )

    total_out = log2phy.numel()
    if total_out == 0:
        return log2phy_prob

    phy_prob_size = num_single + num_red_phy + 1  # +1: always-zero sink slot
    pad_p = _next_pow2(phy_prob_size)
    if pad_p > _MAX_TABLE_ELEMENTS:
        raise ValueError(
            f"physical probability table ({phy_prob_size}) exceeds the UB "
            f"budget of {_MAX_TABLE_ELEMENTS} elements"
        )

    _, num_vectorcore = get_device_properties()
    nvec = num_vectorcore
    tab = torch.empty(pad_p, dtype=torch.float32, device=log2phy_prob.device)

    # --- kernel 1: build the physical probability table ---
    pch = 16 if phy_prob_size <= 320 else 32
    nch = (phy_prob_size + pch - 1) // pch
    nc1 = min(nvec, nch)
    build_tab_kernel[(nc1,)](
        x,
        t1,
        phy_single,
        phy_replicated,
        tab,
        NR=num_red_phy,
        NS=num_single,
        P=phy_prob_size,
        NCH=nch,
        NC=nc1,
        PCH=pch,
        SCH=128,
    )

    # --- kernel 2: gather the logical outputs ---
    block = 512 if total_out <= 2560 else 1024
    nb = (total_out + block - 1) // block
    nc2 = min(nvec, nb)
    gather_out_kernel[(nc2,)](
        log2phy_prob,
        log2phy,
        tab,
        P=phy_prob_size,
        W=pad_p,
        TO=total_out,
        NB=nb,
        NC=nc2,
        BLOCK=block,
    )

    return log2phy_prob

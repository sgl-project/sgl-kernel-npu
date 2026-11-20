"""Triton-Ascend LPLB physical-expert dispatch.

Collapses the torch reference

    topk_probs = log2phy_prob[topk_ids]
    row_sums   = topk_probs.sum(dim=-1)
    fallback   = (log2phy_map[topk_ids] >= 0).float()
    topk_probs = where(row_sums > 0, topk_probs, fallback)
    chosen     = multinomial(topk_probs, 1)
    out        = log2phy_map[topk_ids, chosen]

into one launch. Each program handles one 64-entry token block at a time and
strides over the remaining blocks.

The whole ``[num_logical, max_copies]`` table is loaded into UB once per
program (transposed to ``[max_copies, num_logical]`` so the copy axis is the
reduction axis) and the per-row inverse-CDF sample is a single gather plus a
compare-sum. Tables whose logical/copy counts are not powers of two are padded
with masked lanes, so no alignment restriction is imposed on the caller; the
wrapper rejects tables whose padded size does not fit the UB budget.

The CANN 9.2 variant (``dispatch_probability_cann92``) keeps the table geometry
as compile-time constants and transposes the resident table with ``tl.trans``
instead of building the transposed view from a strided flat index, so no
padding is needed.
"""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties

_DISPATCH_BLOCK = 64

# The resident probability/map tables and the derived [max_copies, BLOCK]
# gather tile must fit in the vector-core UB. 32768 FP32 elements is the same
# single-tile budget lp_post uses.
_MAX_TABLE_ELEMENTS = 32768


@triton.jit(do_not_specialize=["N", "NC", "NL"])
def dispatch_probability_kernel(
    topk_ptr,
    prob_ptr,
    map_ptr,
    rnd_ptr,
    out_ptr,
    N,
    NC,
    NL,
    NLD: tl.constexpr,
    MCD: tl.constexpr,
    MC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: whole table resident in UB, once per program ---
    # Padded flat index: row-major over the transposed [MCD, NLD] view, so the
    # copy axis (axis 0) is the reduction axis.
    f = tl.arange(0, NLD * MCD)
    cidx = f // NLD
    lidx = f % NLD
    tmask = (cidx < MC) & (lidx < NL)
    tab_off = lidx * MC + cidx
    p = tl.reshape(
        tl.load(prob_ptr + tab_off, mask=tmask, other=0.0),
        (MCD, NLD),
    )
    mt = tl.reshape(
        tl.load(map_ptr + tab_off, mask=tmask, other=-1).to(tl.int32),
        (MCD, NLD),
    )
    # Probabilities are nonnegative, so "row sum > 0" is independent of the
    # summation order and a tree reduction is enough to detect a zero row.
    has_p = tl.sum(p, axis=0) > 0.0
    valid = (mt >= 0).to(tl.float32)
    # A zero-probability row falls back to uniform sampling over valid copies:
    # the scan source becomes 0/1 valid, so one cumsum covers both the
    # probability CDF and the valid-copy prefix count.
    src = tl.where(has_p[None, :], p, valid)
    cum = tl.cumsum(src, axis=0)
    # Use the last scanned column as the total to keep row_sum == cum_last,
    # avoiding a 1-ulp boundary flip between tl.sum and the cumsum order.
    tot = tl.max(cum, axis=0)
    cum_f = tl.reshape(cum, (NLD * MCD,))
    # ``tl.gather`` only accepts fp16/fp32/bf16/fp8/int8 data, so the physical
    # ids travel through float32. Ids are bounded by the physical expert count
    # (hundreds in production), far below the 2^24 exact-integer range.
    map_f = tl.reshape(mt, (NLD * MCD,)).to(tl.float32)

    # --- stage 1: per token block, gather + count only ---
    c = tl.arange(0, MCD)
    cmask = c < MC
    o = tl.arange(0, BLOCK)
    nb = tl.cdiv(N, BLOCK)
    for i in tl.range(0, tl.cdiv(nb, NC)):
        offs = (pid + i * NC) * BLOCK + o
        msk = offs < N
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # The in-UB gather has no mask argument: clamp out-of-range ids into
        # [0, NL-1] so the index stays legal. Valid inputs never hit the clamp.
        lid = tl.minimum(tl.maximum(lid, 0), NL - 1)
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        thr = r0 * tl.gather(tot, lid, 0)
        # Gather the whole [MCD, BLOCK] block at once, then count the columns
        # whose CDF value is <= threshold along axis 0.
        ig = tl.reshape(c[:, None] * NLD + lid[None, :], (MCD * BLOCK,))
        row = tl.reshape(tl.gather(cum_f, ig, 0), (MCD, BLOCK))
        chosen = tl.sum(((row <= thr[None, :]) & cmask[:, None]).to(tl.int32), axis=0)
        chosen = tl.minimum(chosen, MC - 1)
        res = tl.gather(map_f, chosen * NLD + lid, 0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)


def dispatch_probability(topk_ids, log2phy_prob, log2phy_map, random_vals=None):
    """Sample one physical expert per ``(token, slot)``.

    Args:
        topk_ids (torch.Tensor): Integer logical expert ids of any shape.
        log2phy_prob (torch.Tensor): LP probabilities [num_logical, max_copies],
            float32. A row that sums to zero falls back to a uniform choice.
        log2phy_map (torch.Tensor): Physical ids [num_logical, max_copies],
            int32/int64; ``-1`` marks an unused replica.
        random_vals (torch.Tensor | None): Optional [numel(topk_ids)] float32
            uniform samples in [0, 1). Fresh values are drawn when omitted.

    Returns:
        torch.Tensor: Physical ids with ``topk_ids``' shape and dtype.
    """
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32/int64")
    if log2phy_prob.dim() != 2 or log2phy_prob.dtype != torch.float32:
        raise TypeError("log2phy_prob must be a 2-D float32 tensor")
    if log2phy_map.shape != log2phy_prob.shape:
        raise ValueError(
            f"log2phy_map shape {tuple(log2phy_map.shape)} must match "
            f"log2phy_prob shape {tuple(log2phy_prob.shape)}"
        )
    if log2phy_map.dtype not in (torch.int32, torch.int64):
        raise TypeError("log2phy_map must be int32/int64")
    if not log2phy_prob.is_contiguous():
        raise ValueError("log2phy_prob must be contiguous")
    if not log2phy_map.is_contiguous():
        raise ValueError("log2phy_map must be contiguous")

    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous()
    num_logical, max_copies = log2phy_prob.shape
    if num_logical == 0 or max_copies == 0:
        raise ValueError("log2phy_prob must have at least one row and one column")

    n = flat_ids.numel()
    if n == 0:
        return torch.empty_like(flat_ids).reshape(original_shape)

    nld = triton.next_power_of_2(num_logical)
    mcd = triton.next_power_of_2(max_copies)
    table_elements = nld * mcd
    gather_elements = mcd * _DISPATCH_BLOCK
    if table_elements > _MAX_TABLE_ELEMENTS or gather_elements > _MAX_TABLE_ELEMENTS:
        raise ValueError(
            f"padded dispatch tables ({table_elements} table and "
            f"{gather_elements} gather elements) exceed the UB budget of "
            f"{_MAX_TABLE_ELEMENTS}"
        )

    if random_vals is None:
        random_vals = torch.rand(n, dtype=torch.float32, device=flat_ids.device)
    else:
        if random_vals.shape != (n,) or random_vals.dtype != torch.float32:
            raise TypeError(f"random_vals must be a float32 tensor of shape {(n,)}")
        random_vals = random_vals.contiguous()

    out = torch.empty_like(flat_ids)
    _, num_vectorcore = get_device_properties()
    grid = min(num_vectorcore, triton.cdiv(n, _DISPATCH_BLOCK))

    dispatch_probability_kernel[(grid,)](
        flat_ids,
        log2phy_prob,
        log2phy_map,
        random_vals,
        out,
        n,
        grid,
        num_logical,
        nld,
        mcd,
        max_copies,
        _DISPATCH_BLOCK,
    )
    return out.reshape(original_shape)


# ---------------------------------------------------------------------------
# CANN 9.2: same contract, table geometry compiled in.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["N", "NC"])
def dispatch_probability_kernel_cann92(
    topk_ptr,
    prob_ptr,
    map_ptr,
    rnd_ptr,
    out_ptr,
    N,
    NC,
    NL: tl.constexpr,
    MC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: whole table resident in UB, once per program ---
    # ``f`` is the linear index over the [NL, MC] table; transposing it puts the
    # copy axis (axis 0) on the reduction axis. CANN 9.2 supports non-power-of-
    # two arange extents, so the table needs no padding here.
    f = tl.arange(0, NL * MC)
    p = tl.trans(tl.reshape(tl.load(prob_ptr + f), (NL, MC)))
    mt = tl.trans(tl.reshape(tl.load(map_ptr + f), (NL, MC)).to(tl.int32))
    # Probabilities are nonnegative, so "row sum > 0" is independent of the
    # summation order and a tree reduction is enough to detect a zero row.
    has_p = tl.sum(p, axis=0) > 0.0
    valid = (mt >= 0).to(tl.float32)
    # A zero-probability row falls back to uniform sampling over valid copies:
    # the scan source becomes 0/1 valid, so one cumsum covers both the
    # probability CDF and the valid-copy prefix count.
    src = tl.where(has_p[None, :], p, valid)
    cum = tl.cumsum(src, axis=0)
    # Use the last scanned column as the total to keep row_sum == cum_last,
    # avoiding a 1-ulp boundary flip between tl.sum and the cumsum order.
    tot = tl.max(cum, axis=0)
    cum_f = tl.reshape(cum, (NL * MC,))
    # ``tl.gather`` only accepts fp16/fp32/bf16/fp8/int8 data, so the physical
    # ids travel through float32. Ids are bounded by the physical expert count
    # (hundreds in production), far below the 2^24 exact-integer range.
    map_f = tl.reshape(mt, (NL * MC,)).to(tl.float32)

    # --- stage 1: per token block, gather + count only ---
    c = tl.arange(0, MC)
    o = tl.arange(0, BLOCK)
    nb = tl.cdiv(N, BLOCK)
    for i in tl.range(0, tl.cdiv(nb, NC)):
        offs = (pid + i * NC) * BLOCK + o
        msk = offs < N
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # The in-UB gather has no mask argument: clamp out-of-range ids into
        # [0, NL-1] so the index stays legal. Valid inputs never hit the clamp.
        lid = tl.minimum(tl.maximum(lid, 0), NL - 1)
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        thr = r0 * tl.gather(tot, lid, 0)
        # Gather the whole [MC, BLOCK] block at once, then count the columns
        # whose CDF value is <= threshold along axis 0.
        ig = tl.reshape(c[:, None] * NL + lid[None, :], (MC * BLOCK,))
        row = tl.reshape(tl.gather(cum_f, ig, 0), (MC, BLOCK))
        chosen = tl.sum((row <= thr[None, :]).to(tl.int32), axis=0)
        chosen = tl.minimum(chosen, MC - 1)
        res = tl.gather(map_f, chosen * NL + lid, 0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)


def dispatch_probability_cann92(topk_ids, log2phy_prob, log2phy_map, random_vals=None):
    """CANN 9.2 kernel; identical contract to ``dispatch_probability``."""
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32/int64")
    if log2phy_prob.dim() != 2 or log2phy_prob.dtype != torch.float32:
        raise TypeError("log2phy_prob must be a 2-D float32 tensor")
    if log2phy_map.shape != log2phy_prob.shape:
        raise ValueError(
            f"log2phy_map shape {tuple(log2phy_map.shape)} must match "
            f"log2phy_prob shape {tuple(log2phy_prob.shape)}"
        )
    if log2phy_map.dtype not in (torch.int32, torch.int64):
        raise TypeError("log2phy_map must be int32/int64")
    if not log2phy_prob.is_contiguous():
        raise ValueError("log2phy_prob must be contiguous")
    if not log2phy_map.is_contiguous():
        raise ValueError("log2phy_map must be contiguous")

    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous()
    num_logical, max_copies = log2phy_prob.shape
    if num_logical == 0 or max_copies == 0:
        raise ValueError("log2phy_prob must have at least one row and one column")

    n = flat_ids.numel()
    if n == 0:
        return torch.empty_like(flat_ids).reshape(original_shape)

    table_elements = num_logical * max_copies
    gather_elements = max_copies * _DISPATCH_BLOCK
    if table_elements > _MAX_TABLE_ELEMENTS or gather_elements > _MAX_TABLE_ELEMENTS:
        raise ValueError(
            f"dispatch tables ({table_elements} table and {gather_elements} "
            f"gather elements) exceed the UB budget of {_MAX_TABLE_ELEMENTS}"
        )

    if random_vals is None:
        random_vals = torch.rand(n, dtype=torch.float32, device=flat_ids.device)
    else:
        if random_vals.shape != (n,) or random_vals.dtype != torch.float32:
            raise TypeError(f"random_vals must be a float32 tensor of shape {(n,)}")
        random_vals = random_vals.contiguous()

    out = torch.empty_like(flat_ids)
    _, num_vectorcore = get_device_properties()
    grid = min(num_vectorcore, triton.cdiv(n, _DISPATCH_BLOCK))

    dispatch_probability_kernel_cann92[(grid,)](
        flat_ids,
        log2phy_prob,
        log2phy_map,
        random_vals,
        out,
        n,
        grid,
        num_logical,
        max_copies,
        _DISPATCH_BLOCK,
    )
    return out.reshape(original_shape)

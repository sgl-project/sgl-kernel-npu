"""Triton port of the DSV4 DSpark tree node projection (``project``).

One query row per tree node is projected once into its FP32 ``[kv | score]``
node row::

    kv    = x[i] @ wkv.T                       # FP32 accumulate
    score = x[i] @ wgate.T + ape[positions[i]]  # APE on the score half only
    nodes[i + 1] = [kv(c*d) | score(c*d)]

``c = 2`` for ``ratio == 4`` (previous/current window halves) and ``c = 1`` for
``ratio == 128``; the half order inside each group is ``half * head_dim``.  Row 0
of ``nodes`` is reserved and is never written.  There is no window aggregation,
RMSNorm or RoPE here: ``compressor_tree`` owns that epilogue.

Semantics source: vllm-ascend @ 8947f1f9f
  ``csrc/attention/compressor_tree_project/torch_binding.h`` (dtype/shape/stride
  contract, in-place ``nodes``) and
  ``csrc/attention/compressor_tree_project/README.md`` (one query per node,
  ``cu = arange(n + 1)``, row 0 reserved).  See ``docs/tree_ops_semantics.md`` §2
  and ``tree/README.md``.

Tiling (vendor parity)
----------------------
The template this op actually runs on 910B/910C is
``arch32/compressor_tree_project_kernel_perf.h`` together with
``arch32/compressor_tree_project_block_cube_perf.h::ComputeMm1``, tiled by
``op_host/arch32/compressor_tree_project_tiling.cpp::SetInnerSplitInfo``:
``dBaseSize = 64``, ``K_L1_BASE = 256``, ``K_L0_BASE = 128``,
``mBaseSize = 128`` (``coff == 2``) / ``256`` (``coff == 1``), and each core owns
one ``dIdx`` slice of ``headDim`` plus one M group (``dBasicBlockNum =
headDim / dBaseSize``, ``coreGroupNum = usedCoreNum / dBasicBlockNum``).

The load-bearing vendor decision is the **fused kv+gate N tile**:
``CopyWeightGmToL1`` stages ``dBaseSize`` rows of ``wkv`` *and* ``dBaseSize`` rows
of ``wgate`` for the same ``dIdx`` into one L1 block (``dstNzC0Stride =
2 * dBaseSize``), a single ``n = 2 * dBaseSize`` ``Mmad`` produces both halves, and
``CopyOutMm1Res`` splits them into the kv and score workspaces.  One x tile
therefore feeds both projections and the weight stream is touched once, not
twice.  The APE row is published only on the score half (``AddApeToScore`` runs
under ``if constexpr (IS_SCORE)``).

Chosen tiles::

    BLOCK_M = 64   node rows per program        (vendor: 128 / 256)
    BLOCK_D = 64   per-matrix columns per tile  (vendor: dBaseSize = 64)
    BLOCK_K = 256  reduction block              (vendor: K_L1_BASE = 256)

with a flat, D-major grid of ``cdiv(T, BLOCK_M) * cdiv(c * head_dim, BLOCK_D)``
programs: one program per *fused* kv+gate N tile, not one per half.

Deviations from the exact vendor shape, and why:

* ``BLOCK_M = 64`` instead of 128/256.  ``T`` is the node budget of one draft
  round (tens of rows, and only a multiple of it when batching), so the vendor's
  128 is a prefill-sized M.  For ``T <= 64`` both give one M tile and therefore
  identical weight traffic, while 64 halves the wasted Cube rows and the UB
  budget; 256 is impossible outright (two FP32 ``[256, 64]`` accumulators alone
  would be 128 KiB of UB).  The cost of 64 is that ``65 <= T <= 128`` needs two M
  tiles and re-reads the weights per tile; if measurement shows that case
  dominating, the vendor-shaped swap is ``BLOCK_M = 128, BLOCK_K = 128`` (the
  vendor ``mBaseSize`` and ``K_L0_BASE``), which stays inside UB (x 32 KiB,
  w 2 x 16 KiB, acc 2 x 32 KiB, ape 32 KiB = 160 KiB) at the price of twice the
  K-loop trips.
* ``BLOCK_K = 256`` -- the vendor ``K_L1`` stage only.  Triton exposes a single
  reduction block, so ``K_L0_BASE = 128`` stays the compiler's business inside
  ``tl.dot``.
* Two ``n = 64`` dots instead of one ``n = 128`` dot: Triton cannot fuse two
  distinct weight tensors into a single dot.  FLOP count and shared-x reuse are
  identical; only L0B/L0C blocking differs.
* No per-core ``hIdx`` stagger (the vendor rotates the K start by
  ``(coreIdx % dBasicBlockNum) * K_L1_BASE``).  Our flat D-major job order already
  gives concurrent programs distinct weight slices; only programs sharing one
  ``d`` (``num_m > 1``, i.e. ``T > 64``) would contend, so the rotation buys
  nothing here and would hide the affine K loop from the pipeliner.

Static UB budget (192 KiB per AIV on DAV_2201 / DAV_3510), BF16 in / FP32 out:

====================================  =======  ===========================
live tensor                           bytes    live during
====================================  =======  ===========================
x tile         ``[64, 256]``           32 768   K loop
wkv tile       ``[256, 64]``           32 768   K loop
wgate tile     ``[256, 64]``           32 768   K loop
acc_kv         ``[64, 64]`` FP32       16 384   epilogue (16 KiB in L0C too)
acc_gate       ``[64, 64]`` FP32       16 384   epilogue (16 KiB in L0C too)
ape tile       ``[64, 64]`` FP32       16 384   epilogue
====================================  =======  ===========================
peak, K loop                           98 304   50% of 192 KiB
worst case, assuming no buffer reuse  147 456   75% of 192 KiB
====================================  =======  ===========================

Worked examples at ``hidden = 7168``, ``T = 48`` (a single M tile):

* ``ratio = 4, head_dim = 512``: 16 programs (was 48 at ``BLOCK_M = 16`` with
  separate kv/score programs); weight traffic 29.4 MB once instead of 88.2 MB
  (three M tiles x two weight streams).
* ``ratio = 4, head_dim = 128``: 4 programs (was 12) -- a fused N tile is 64
  columns wide, so 256 projected columns can only occupy four AIC.
* ``ratio = 128, head_dim = 512``: 8 programs (was 24).

Only :mod:`torch`, :mod:`triton` and :mod:`triton.language` are imported.  The
independent CPU oracle in ``tests/python/sgl_kernel_npu/tree_reference.py`` is
never imported here (D5).
"""

import torch
import triton
import triton.language as tl

from . import _common

__all__ = ["compressor_tree_project"]

# Vendor-shaped, UB-legal tiles (see "Tiling" above).  ``BLOCK_D`` is the vendor
# ``dBaseSize``: one kv slice and the matching gate slice are produced together
# by the same program, sharing one x tile.
_BLOCK_M = 64
_BLOCK_D = 64
_BLOCK_K = 256
_NUM_WARPS = 1
_NUM_STAGES = 1


@triton.jit
def _compressor_tree_project_kernel(
    x_ptr,
    wkv_ptr,
    wgate_ptr,
    nodes_ptr,
    ape_ptr,
    positions_ptr,
    T,
    H,
    num_d,
    stride_xm,
    stride_wn,
    stride_ape,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused kv+gate projection of one ``[BLOCK_M, BLOCK_D]`` N tile.

    Grid is the flat product ``cdiv(T, BLOCK_M) * num_d`` with ``d`` on the fast
    axis, so neighbouring programs own different ``d`` slices (distinct weight
    streams) while their shared x tile stays hot.  ``num_d`` is
    ``cdiv(c * head_dim, BLOCK_D)`` and ``cols`` is the offset inside *one*
    projection matrix, i.e. inside ``[0, HALF)``.

    A single program produces the kv slice and the matching gate slice from the
    same x tile -- the Triton transcription of the vendor's single
    ``n = 2 * dBaseSize`` ``Mmad``.  The APE row is added to the gate half only
    and the two halves land in ``nodes[i + 1]`` at ``[0, HALF)`` and
    ``[HALF, 2 * HALF)``.
    """
    WIDTH: tl.constexpr = 2 * HALF

    pid = tl.program_id(0)
    d_tile = pid % num_d
    m_tile = pid // num_d

    rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    rows64 = rows.to(tl.int64)

    cols = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < HALF
    cols64 = cols.to(tl.int64)

    acc_kv = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for i_k in range(tl.cdiv(H, BLOCK_K)):
        k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k < H
        x = tl.load(
            x_ptr + rows64[:, None] * stride_xm + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # wkv/wgate are [HALF, H], so the reduction runs along the innermost
        # weight axis and each tile is the transposed [BLOCK_K, BLOCK_D] block.
        w_offs = cols64[None, :] * stride_wn + k[:, None]
        w_mask = k_mask[:, None] & col_mask[None, :]
        w_kv = tl.load(wkv_ptr + w_offs, mask=w_mask, other=0.0)
        w_gate = tl.load(wgate_ptr + w_offs, mask=w_mask, other=0.0)
        acc_kv += tl.dot(x, w_kv)
        acc_gate += tl.dot(x, w_gate)

    # APE is a per-residue bias of the score half only (vendor ``AddApeToScore``
    # under ``if constexpr (IS_SCORE)``); the kv half never sees it.
    position = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    ape_offs = position.to(tl.int64)[:, None] * stride_ape + cols64[None, :]
    ape = tl.load(
        ape_ptr + ape_offs,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    acc_gate += ape

    # Row 0 of ``nodes`` is reserved, so node i lands on row i + 1.
    out_row = (rows64 + 1)[:, None] * WIDTH
    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(nodes_ptr + out_row + cols64[None, :], acc_kv, mask=store_mask)
    tl.store(nodes_ptr + out_row + (HALF + cols64)[None, :], acc_gate, mask=store_mask)


def compressor_tree_project(
    x, wkv, wgate, nodes, ape, positions, cu, ratio=None, head_dim=None
) -> None:
    """Write ``nodes[1:] = [x @ wkv.T | x @ wgate.T + ape[positions]]`` in place.

    Args:
        x: BF16 ``[T, H]`` contiguous query rows, one per tree node.
        wkv: BF16 ``[c * head_dim, H]`` contiguous projection weight.
        wgate: BF16 ``[c * head_dim, H]`` contiguous gate weight.
        nodes: FP32 ``[T + 1, 1, 2 * c * head_dim]`` contiguous **in-place**
            output.  Row 0 is reserved and stays bit-identical.
        ape: FP32 ``[ratio, c * head_dim]`` contiguous per-residue bias.
        positions: int32 ``[T]`` contiguous APE residue in ``[0, ratio)``.
        cu: int32 ``[T + 1]`` contiguous, must equal ``arange(T + 1)`` (the
            one-query-per-node contract).
        ratio: optional compression ratio, ``4`` or ``128`` (``c = 2`` or
            ``1``).  When omitted it is derived from ``ape.shape[0]``, matching
            the native seven-argument caller.
        head_dim: optional projected half width, ``128`` or ``512``.  When
            omitted it is derived from ``ape.shape[1] // c``.

    Returns:
        ``None``; the result is written into ``nodes``.

    Raises:
        ValueError: (``TreeContractError``) if any host contract is violated.
            Nothing is launched in that case.
    """
    _common.check_tensor("x", x, dtype=torch.bfloat16, rank=2, contiguous=True)
    t, hidden = int(x.shape[0]), int(x.shape[1])
    if hidden <= 0:
        raise _common.TreeContractError(
            f"x: expected a positive hidden size, got {hidden}"
        )

    # The native caller passes only ``(x, wkv, wgate, nodes, ape, positions,
    # cu)``; ratio/head_dim are derived from APE exactly like torch_binding.h.
    # Keep the explicit 9-arg form for callers that already know them.
    _common.check_tensor("ape", ape, dtype=torch.float32, rank=2, contiguous=True)
    if ratio is None:
        ratio = int(ape.shape[0])
    else:
        ratio = _common.check_ratio(ratio)
    c = 2 if ratio == 4 else 1
    if head_dim is None:
        if int(ape.shape[1]) % c != 0:
            raise _common.TreeContractError(
                f"ape: width {ape.shape[1]} is not divisible by c={c}"
            )
        head_dim = int(ape.shape[1]) // c
    else:
        head_dim = _common.check_head_dim(head_dim)
    # Derived values need the same fast-fail contract as explicit ones;
    # otherwise an unsupported APE width silently mis-tiles the projection.
    ratio = _common.check_ratio(ratio)
    head_dim = _common.check_head_dim(head_dim)
    c = 2 if ratio == 4 else 1
    half = c * head_dim
    width = 2 * half
    if tuple(ape.shape) != (ratio, half):
        raise _common.TreeContractError(
            f"ape: expected shape {(ratio, half)}, got {tuple(ape.shape)}"
        )

    for name, weight in (("wkv", wkv), ("wgate", wgate)):
        _common.check_tensor(
            name, weight, dtype=torch.bfloat16, rank=2, contiguous=True
        )
        if tuple(weight.shape) != (half, hidden):
            raise _common.TreeContractError(
                f"{name}: expected shape {(half, hidden)}, got {tuple(weight.shape)}"
            )

    _common.check_tensor("nodes", nodes, dtype=torch.float32, contiguous=True)
    if nodes.numel() != (t + 1) * width:
        raise _common.TreeContractError(
            f"nodes: expected {(t + 1) * width} FP32 entries "
            f"({t + 1} rows of {width}), got {nodes.numel()}"
        )
    if nodes.dim() == 3 and tuple(nodes.shape) != (t + 1, 1, width):
        raise _common.TreeContractError(
            f"nodes: expected shape {(t + 1, 1, width)}, got {tuple(nodes.shape)}"
        )
    if nodes.dim() >= 2 and int(nodes.size(0)) != t + 1:
        raise _common.TreeContractError(
            f"nodes: expected {t + 1} rows, got {tuple(nodes.shape)}"
        )

    _common.check_tensor(
        "positions", positions, dtype=torch.int32, rank=1, contiguous=True
    )
    if positions.numel() != t:
        raise _common.TreeContractError(
            f"positions: expected {t} entries, got {positions.numel()}"
        )

    _common.check_cu_arange("cu", cu, t)
    _common.check_same_device(
        "compressor_tree_project", [x, wkv, wgate, nodes, ape, positions, cu]
    )

    if t > 0 and not bool(((positions >= 0) & (positions < ratio)).all()):
        raise _common.TreeContractError(
            f"positions: expected APE residues in [0, {ratio})"
        )

    if t == 0:
        return None

    # One program per fused kv+gate N tile (vendor ``dBaseSize`` slice), D-major.
    num_m = _common.ceil_div(t, _BLOCK_M)
    num_d = _common.ceil_div(half, _BLOCK_D)
    _compressor_tree_project_kernel[(num_m * num_d,)](
        x,
        wkv,
        wgate,
        nodes,
        ape,
        positions,
        t,
        hidden,
        num_d,
        x.stride(0),
        wkv.stride(0),
        ape.stride(0),
        HALF=half,
        BLOCK_M=_BLOCK_M,
        BLOCK_D=_BLOCK_D,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return None

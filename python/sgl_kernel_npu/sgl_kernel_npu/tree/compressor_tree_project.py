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

Only :mod:`torch`, :mod:`triton` and :mod:`triton.language` are imported.  The
independent CPU oracle in ``tests/python/sgl_kernel_npu/tree_reference.py`` is
never imported here (D5).
"""

import torch
import triton
import triton.language as tl

from . import _common

__all__ = ["compressor_tree_project"]

# Static tiling (D6): the node dimension is tiled by 16 (the Ascend Cube BF16
# base tile), the projected dimension by 128 and the reduction by 256.
_BLOCK_M = 16
_BLOCK_N = 128
_BLOCK_K = 256
_NUM_WARPS = 1
_NUM_STAGES = 1


@triton.jit
def _project_block(
    x_ptr,
    w_ptr,
    nodes_ptr,
    ape_ptr,
    positions_ptr,
    rows,
    row_mask,
    local_cols,
    col_mask,
    w_col,
    out_col,
    H,
    stride_xm,
    stride_wn,
    stride_ape,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_SCORE: tl.constexpr,
):
    """Project one ``[BLOCK_M, BLOCK_N]`` half-tile and store it.

    ``w_col`` is the column offset inside the weight matrix (``HALF`` wide) and
    ``out_col`` the column offset inside the node row (``2 * HALF`` wide).  The
    APE row is added only when ``IS_SCORE`` (compile-time specialization).
    """
    WIDTH: tl.constexpr = 2 * HALF
    rows64 = rows.to(tl.int64)
    half_cols = w_col + local_cols
    half_mask = half_cols < HALF
    half_cols64 = half_cols.to(tl.int64)
    out_cols = out_col + local_cols
    k_offs = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i_k in range(tl.cdiv(H, BLOCK_K)):
        k = i_k * BLOCK_K + k_offs
        k_mask = k < H
        x = tl.load(
            x_ptr + rows64[:, None] * stride_xm + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # wkv/wgate are [HALF, H]; the tile is the transposed [BLOCK_K, BLOCK_N]
        # block, i.e. the reduction runs along the innermost weight axis.
        w = tl.load(
            w_ptr + half_cols64[None, :] * stride_wn + k[:, None],
            mask=half_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(x, w)

    if IS_SCORE:
        position = tl.load(positions_ptr + rows, mask=row_mask, other=0)
        ape_offs = position.to(tl.int64)[:, None] * stride_ape + half_cols64[None, :]
        ape = tl.load(
            ape_ptr + ape_offs,
            mask=row_mask[:, None] & half_mask[None, :],
            other=0.0,
        )
        acc = acc + ape

    # Row 0 is reserved, so node i lands on row i + 1.
    out_offs = (rows64 + 1)[:, None] * WIDTH + out_cols[None, :]
    tl.store(nodes_ptr + out_offs, acc, mask=row_mask[:, None] & col_mask[None, :])


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
    stride_xm,
    stride_wn,
    stride_ape,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid ``(cdiv(T, BLOCK_M), cdiv(2 * HALF, BLOCK_N))`` over the node row."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    WIDTH: tl.constexpr = 2 * HALF

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    local_cols = tl.arange(0, BLOCK_N)
    out_col = pid_n * BLOCK_N
    col_mask = (out_col + local_cols) < WIDTH

    # BLOCK_N divides every supported HALF (128/256/512/1024), so a tile never
    # straddles the kv/score boundary and the choice is uniform per program.
    if out_col < HALF:
        _project_block(
            x_ptr,
            wkv_ptr,
            nodes_ptr,
            ape_ptr,
            positions_ptr,
            rows,
            row_mask,
            local_cols,
            col_mask,
            out_col,
            out_col,
            H,
            stride_xm,
            stride_wn,
            stride_ape,
            HALF=HALF,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            IS_SCORE=False,
        )
    else:
        _project_block(
            x_ptr,
            wgate_ptr,
            nodes_ptr,
            ape_ptr,
            positions_ptr,
            rows,
            row_mask,
            local_cols,
            col_mask,
            out_col - HALF,
            out_col,
            H,
            stride_xm,
            stride_wn,
            stride_ape,
            HALF=HALF,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            IS_SCORE=True,
        )


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

    # A tile must never straddle the kv/score boundary.  HALF=128 needs the
    # smaller tile; the wider 256 tile halves program count for larger heads.
    block_n = 128 if half == 128 else 256
    grid = (triton.cdiv(t, _BLOCK_M), triton.cdiv(width, block_n))
    _compressor_tree_project_kernel[grid](
        x,
        wkv,
        wgate,
        nodes,
        ape,
        positions,
        t,
        hidden,
        x.stride(0),
        wkv.stride(0),
        ape.stride(0),
        HALF=half,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return None

"""NPU QSA expansion for complete blocks with prefix-valid Top-K output."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1
_OUTPUT_TILE = 1024
_MAX_GRID = 65535
_NATIVE_MIN_ROWS = 128
_NATIVE_CHUNK_ELEMENTS = 1 << 25


@lru_cache(maxsize=None)
def _vector_cores(device):
    return triton.runtime.driver.active.utils.get_device_properties(device)[
        "num_vectorcore"
    ]


@triton.jit
def _expand_prefix(
    blocks,
    positions,
    lengths,
    output,
    ROWS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    WIDTH: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    PS: tl.constexpr,
    LS: tl.constexpr,
    TILE: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(WIDTH, TILE)
    # Independent output tiles share a bounded grid; task/row addresses stay int64.
    for task in range(tl.program_id(0).to(tl.int64), ROWS * TILES, tl.num_programs(0)):
        row = task // TILES
        col = (task % TILES).to(tl.int32) * TILE + tl.arange(0, TILE)
        ids = tl.arange(0, BLOCK_TOPK)
        selected = tl.load(blocks + row * BS0 + ids.to(tl.int64) * BS1).to(tl.int32)
        token_count = tl.sum((selected >= 0).to(tl.int32), 0) * RATIO
        visible = tl.load(positions + row * PS).to(tl.int32) + 1
        length = tl.load(lengths + row * LS).to(tl.int32)
        tail_start = visible // RATIO * RATIO
        tail_count = tl.minimum(
            visible - tail_start, tl.maximum(length - tail_start, 0)
        )
        # NPU gather accepts float payloads: bitcast, never round integer IDs.
        block = tl.gather(
            selected.to(tl.float32, bitcast=True),
            tl.minimum(col // RATIO, BLOCK_TOPK - 1),
            axis=0,
        ).to(tl.int32, bitcast=True)
        tail_offset = col - token_count
        value = tl.where(
            col < token_count,
            block * RATIO + col % RATIO,
            tl.where(
                (tail_offset >= 0) & (tail_offset < tail_count),
                tail_start + tail_offset,
                -1,
            ),
        )
        tl.store(output + row * WIDTH + col, value, mask=col < WIDTH)


def can_run_block_expansion(blocks, positions, lengths, ratio, topk):
    """Check host metadata only; device-value preconditions are listed below."""
    if type(ratio) is not int or type(topk) is not int:
        return False
    if ratio < 2 or topk <= 0 or topk % ratio or topk // ratio not in (512, 2048):
        return False
    # Leave room for padded tile columns; no overflow in int32 local arithmetic.
    width = topk + ratio - 1
    if width > _INT32_MAX - (_OUTPUT_TILE - 1):
        return False
    tensors = (blocks, positions, lengths)
    if not all(
        isinstance(t, torch.Tensor) and t.layout == torch.strided for t in tensors
    ):
        return False
    if blocks.ndim != 2 or blocks.shape[1] != topk // ratio:
        return False
    if positions.shape != lengths.shape or positions.shape != (blocks.shape[0],):
        return False
    if blocks.device.type != "npu":
        return False
    if not all(
        t.device == blocks.device and t.dtype in (torch.int32, torch.int64)
        for t in tensors
    ):
        return False
    if any(s < 0 for t in tensors for s in t.stride()):
        return False
    # Byte offsets and output allocation must remain representable in int64.
    limit = 2**63 - 1
    return blocks.shape[0] * width * 4 <= limit and all(
        (
            t.storage_offset()
            + sum(max(n - 1, 0) * s for n, s in zip(t.shape, t.stride()))
            + 1
        )
        * t.element_size()
        <= limit
        for t in tensors
    )


def expansion_branch(blocks, ratio, topk):
    """Metadata-only dispatch; see expand_blocks for shapes and preconditions."""
    rows = blocks.shape[0]
    return (
        "empty"
        if rows == 0
        else ("native_direct" if rows >= _NATIVE_MIN_ROWS else "direct_prefix")
    )


def _expand_native(blocks, positions, lengths, ratio, topk):
    rows = blocks.shape[0]
    output = torch.full(
        (rows, topk + ratio - 1), -1, dtype=torch.int32, device=blocks.device
    )
    offsets = torch.arange(ratio, dtype=torch.int32, device=blocks.device)
    values = blocks.int()[:, :, None]
    expanded = torch.where(values >= 0, values * ratio + offsets, -1)
    output[:, :topk].copy_(expanded.reshape(rows, topk))
    del values, expanded
    count = (blocks >= 0).sum(dim=1, dtype=torch.int32) * ratio
    visible = positions.int() + 1
    start = torch.div(visible, ratio, rounding_mode="floor") * ratio
    tail_offsets = torch.arange(ratio - 1, dtype=torch.int32, device=blocks.device)
    tail = start[:, None] + tail_offsets
    valid = (tail_offsets < (visible - start)[:, None]) & (
        tail < lengths.int()[:, None]
    )
    tail = torch.where(valid, tail, -1)
    # Each row writes distinct tail slots immediately after its full blocks.
    output.scatter_(1, (count[:, None] + tail_offsets).long(), tail)
    return output


def expand_blocks(blocks, positions, lengths, ratio, topk):
    """Expand selected compressed blocks into logical token indices.

    Inputs and output
    -----------------
        blocks      [R, K]   selected sequence-local compressed-block indices
        positions   [R]      zero-based logical position of each query token
        lengths     [R]      sequence length in tokens for each query row
        output      [R, W]   new contiguous int32 tensor on the same NPU

        R           total query rows, including any caller-padded rows
        K           block_topk: number of selected block slots, 512 or 2048
        ratio       tokens per compressed block
        topk        token_topk = K * ratio; NOT the upstream block_topk
        W           token_topk + ratio - 1

    Current model: ratio=4, token_topk=2048, block_topk=512, W=2051.
    TP does not divide these columns. R is dynamic; 128 is not a row limit.
    Decode/verify use query rows; prefill uses the current caller chunk.
    Neither mode guarantees R<128.

    Upstream input contract
    -----------------------
    The wrapper checks host metadata:
    - Same NPU; int32/int64 inputs, including mixed dtypes. Nonnegative
      strides, non-contiguous slices and broadcast inputs are supported.
    - Shapes as above; Python integer ratio>=2 and positive token_topk,
      divisible by ratio, with K in {512,2048}. Bool parameters are invalid.
    - W<=INT32_MAX-1023; input byte spans and output size must fit int64.
      Actual allocations still require enough device memory.

    The caller guarantees device contents; these are NOT validated:
    - Top-K returns nonnegative, complete block indices in a valid prefix,
      followed only by -1 padding. For every valid b,
      (b+1)*ratio <= min(position+1, length). The upstream metadata excludes
      incomplete blocks from Top-K; expansion appends their tail separately.
    - 0<=length<=INT32_MAX and -1<=position<INT32_MAX. Active positions are
      below length; an inert empty row may use position=0, length=0.
    - No holes between valid blocks, selected partial blocks, or integer
      wrapping. Violating these content premises has no correctness guarantee.

    Unsupported metadata raises ValueError; there is no reference fallback.
    No tensor values are read on the host and no validation scan is added.
    This shares the GPU fast path's prefix premise, not all reference behavior.

    Examples and two computation paths
    ----------------------------------
    With ratio=4 and token_topk=2048, positions and lengths each have shape [R]:

        R (query rows) | Path         | Example: blocks -> output
        ---------------+--------------+-----------------------------
        R == 0         | Empty        | [0, 512] -> [0, 2051]
        1 <= R < 128   | Triton       | [8, 512] -> [8, 2051]
        R >= 128       | Torch/native | [4096, 512] -> [4096, 2051]

    Empty inputs launch no computation. Triton avoids native setup overhead
    at small R; native performs better in measured large-R cases.
    Rows 127 and 128 have identical expansion semantics. The threshold is a
    coarse performance policy, not a correctness boundary or portable optimum.

    Block 10 expands to [40,41,42,43]. With position=46 and length=47,
    append tail [44,45,46] after the selected complete blocks, then pad with -1.
    All -1 blocks do not imply all -1 output: position=0, length=1 produces
    tail token 0. The caller still handles padding and inactive request rows.

    Tiles and row chunks
    --------------------
    Triton: one task writes 1024 consecutive output slots of one row.
    W=2051 needs three tiles; the last stores only three slots. Each tile
    loads the row's K block indices. Flatten (row, tile) into R*ceil(W/1024)
    tasks; launch min(tasks, 65535, vector_core_count) programs and loop over
    remaining tasks inside each program. Address arithmetic stays int64.

    Native: process at most max(1, 2**25 // W) rows per chunk. At W=2051,
    this is 16360 rows: R=4096 fits one chunk; R=16384 uses 16360+24.
    Chunks are resource management, not a third expansion algorithm. This
    element budget does not bound total memory for output, temporaries or graphs.

    How both paths compute the result
    ---------------------------------
    Count the n valid blocks. Since they form a complete prefix, their tokens
    occupy the first P=n*ratio output slots, with no prefix scan or compaction.
    For j<P, write blocks[j//ratio]*ratio + j%ratio.

    The tail starts at floor((position+1)/ratio)*ratio and ends just before
    min(position+1, length). Write it starting at output slot P, then fill -1.
    Preserve block entry order and duplicates; do not sort or deduplicate tokens.
    Expansion does not independently select the highest-scoring token_topk tokens.

    Triton gathers the selected blocks and writes each output tile directly.
    Its local gather uses FP32 bitcast payloads, never FP32 numeric conversion.
    Native broadcasts block indices with offsets, then scatter-writes only the
    tail at P. Neither path uses sorting keys or general-purpose hole compaction.

    Inputs are unchanged. Warm up before graph capture; replay may update all
    three inputs in place under the same contract, with shapes, strides, storage
    and ratio/token_topk fixed. Graph and output lifetimes remain with the caller.
    """
    if not can_run_block_expansion(blocks, positions, lengths, ratio, topk):
        raise ValueError("Unsupported NPU model block expansion metadata")
    rows = blocks.shape[0]
    width = topk + ratio - 1
    if expansion_branch(blocks, ratio, topk) == "native_direct":
        chunk_rows = max(1, _NATIVE_CHUNK_ELEMENTS // width)
        if rows <= chunk_rows:
            return _expand_native(blocks, positions, lengths, ratio, topk)
        output = torch.empty((rows, width), dtype=torch.int32, device=blocks.device)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            output[start:end].copy_(
                _expand_native(
                    blocks[start:end],
                    positions[start:end],
                    lengths[start:end],
                    ratio,
                    topk,
                )
            )
        return output
    output = torch.empty((rows, width), dtype=torch.int32, device=blocks.device)
    if rows:
        _expand_prefix[
            (
                min(
                    rows * triton.cdiv(width, _OUTPUT_TILE),
                    _MAX_GRID,
                    _vector_cores(blocks.device.index),
                ),
            )
        ](
            blocks,
            positions,
            lengths,
            output,
            rows,
            topk // ratio,
            ratio,
            width,
            *blocks.stride(),
            positions.stride(0),
            lengths.stride(0),
            _OUTPUT_TILE,
        )
    return output

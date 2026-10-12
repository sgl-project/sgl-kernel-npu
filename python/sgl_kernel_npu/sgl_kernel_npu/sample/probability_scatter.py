"""Scatter normalized, sorted probabilities without writing zero entries."""

import torch
import triton
import triton.language as tl

_MAX_GRID_PROGRAMS = 65535
_BLOCK_SIZE = 512
_MAX_SPLITS = 16


@triton.jit
def _scatter_nonzero(
    Values,
    Indices,
    RowEnds,
    Output,
    VALUE_ROW: tl.constexpr,
    VALUE_COL: tl.constexpr,
    INDEX_ROW: tl.constexpr,
    INDEX_COL: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    OUTPUT_COL: tl.constexpr,
    WIDTH: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    first_column = tl.program_id(1).to(tl.int64) * TILES * BLOCK_SIZE
    end = tl.load(RowEnds + row).to(tl.int64)
    live_width = tl.maximum(tl.minimum(end - first_column, TILES * BLOCK_SIZE), 0)
    active_tiles = tl.cdiv(live_width, BLOCK_SIZE)
    for tile in range(active_tiles):
        columns = first_column + tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            Values + row * VALUE_ROW + columns * VALUE_COL,
            columns < WIDTH,
            other=0,
        )
        live = (columns < WIDTH) & (values != 0)
        if tl.sum(live.to(tl.int32), 0) > 0:
            indices = tl.load(
                Indices + row * INDEX_ROW + columns * INDEX_COL,
                columns < WIDTH,
                other=0,
            ).to(tl.int64)
            # Each row's indices are a sort permutation: writes cannot collide.
            tl.store(Output + row * OUTPUT_ROW + indices * OUTPUT_COL, values, live)


def scatter_nonzero_sorted_probs(probs, sorted_probs, sorted_indices):
    """Preserve dense output and bound sparse writes entirely on device."""
    rows, width = sorted_probs.shape
    if width > 2**31 - 1:
        return torch.zeros_like(probs).scatter_(
            dim=-1, index=sorted_indices, src=sorted_probs
        )
    output = torch.zeros_like(probs)
    if rows == 0 or width == 0:
        return output
    columns = torch.arange(1, width + 1, device=probs.device, dtype=torch.int32).view(
        1, -1
    )
    # Last live position, not count(nonzero): retain holes and nonfinite values.
    # No host readback, scalar synchronization, or fixed top-k assumption.
    row_ends = torch.where(sorted_probs != 0, columns, 0).amax(dim=1)
    splits = min(_MAX_SPLITS, _MAX_GRID_PROGRAMS, triton.cdiv(width, _BLOCK_SIZE))
    tiles = triton.cdiv(width, splits * _BLOCK_SIZE)
    chunk_rows = max(1, _MAX_GRID_PROGRAMS // splits)
    for start in range(0, rows, chunk_rows):
        values = sorted_probs[start : start + chunk_rows]
        indices = sorted_indices[start : start + chunk_rows]
        ends = row_ends[start : start + chunk_rows]
        target = output[start : start + chunk_rows]
        _scatter_nonzero[(values.shape[0], splits)](
            values,
            indices,
            ends,
            target,
            values.stride(0),
            values.stride(1),
            indices.stride(0),
            indices.stride(1),
            target.stride(0),
            target.stride(1),
            width,
            tiles,
            _BLOCK_SIZE,
        )
    return output

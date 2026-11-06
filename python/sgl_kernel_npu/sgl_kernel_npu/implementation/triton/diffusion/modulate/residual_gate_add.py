"""Bit-exact ``residual + update * gate`` for Ascend NPUs.

The eager chain is ``mul`` followed by ``add``, each rounding its BF16 result.
This kernel reproduces both boundaries in one pass: the product is rounded to
BF16 before the add, and the store rounds the sum.
"""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties
from ..diffusion_utils import cast_rne_bf16_to_fp32

# Rows per tile, clamped by the FP32 element budget below. One tile holds
# residual, update, output and the rounded product in FP32.
DEFAULT_ROWS = 16
# FP32 elements per tile: four live buffers at 8192 elements are about 128 KB,
# which stays inside the 192 KB UB budget.
ELEMENTS_BUDGET = 8192
# The kernel indexes with 32-bit offsets; larger tensors need a different layout.
MAX_NUMEL = 2**31


@triton.jit(do_not_specialize=["num_tiles"])
def _residual_gate_add_kernel(
    out_ptr,
    residual_ptr,
    update_ptr,
    gate_ptr,
    num_tiles,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    GATE_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.arange(0, ROWS)[:, None]
    col = tl.arange(0, COLS)[None, :]
    for tile in range(pid, num_tiles, NUM_PROGS):
        offs = (tile * ROWS + row) * COLS + col
        residual = tl.load(residual_ptr + offs).to(tl.float32)
        update = tl.load(update_ptr + offs).to(tl.float32)
        if GATE_PER_ROW:
            gate = tl.load(gate_ptr + offs).to(tl.float32)
        else:
            gate = tl.load(gate_ptr + col).to(tl.float32)
        # eager rounds the product to BF16, then the add rounds once more
        product = cast_rne_bf16_to_fp32(update * gate)
        tl.store(out_ptr + offs, residual + product)


def residual_gate_add(residual, update, gate):
    """Compute ``residual + update * gate`` with the eager rounding boundaries.

    ``residual``, ``update`` and ``gate`` are BF16 on one NPU, with ``update``
    matching ``residual``. ``gate`` is either the same shape as ``residual`` or
    a single broadcast row (``[1, C]`` / ``[C]``). The last dimension must be a
    power of two so a tile is one contiguous DMA per row.
    """
    if residual.device.type != "npu" or update.device != residual.device:
        raise ValueError("residual and update must be on the same NPU")
    if gate.device != residual.device:
        raise ValueError("gate must be on the same NPU as residual")
    if residual.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("residual_gate_add expects BF16 inputs")
    if update.dtype != residual.dtype or update.shape != residual.shape:
        raise ValueError("update must match residual's shape and dtype")
    if residual.ndim < 2 or not (residual.is_contiguous() and update.is_contiguous()):
        raise ValueError("residual and update must be contiguous and 2-D or more")
    numel = residual.numel()
    if not 0 < numel < MAX_NUMEL:
        raise ValueError("residual_gate_add expects 0 < numel < 2**31")

    cols = residual.shape[-1]
    if cols & (cols - 1):
        raise ValueError("the last dimension must be a power of two")
    if cols > ELEMENTS_BUDGET:
        raise ValueError(
            f"the last dimension must fit one tile of {ELEMENTS_BUDGET} elements"
        )
    rows = numel // cols
    rows_per_tile = min(DEFAULT_ROWS, max(1, ELEMENTS_BUDGET // cols))
    while rows_per_tile > 1 and rows % rows_per_tile:
        rows_per_tile //= 2
    if rows % rows_per_tile:
        raise ValueError(
            f"rows must divide into tiles; got {rows} rows of {cols} columns"
        )

    if not gate.is_contiguous():
        raise ValueError("gate must be contiguous")
    gate_per_row = gate.numel() == numel
    if not gate_per_row and not (gate.numel() == cols and gate.stride(-1) == 1):
        raise ValueError("gate must match residual's shape or be one row of it")

    out = torch.empty_like(residual)
    _, num_vectorcore = get_device_properties()
    num_tiles = rows // rows_per_tile
    progs = min(num_tiles, num_vectorcore)
    _residual_gate_add_kernel[(progs,)](
        out,
        residual,
        update,
        gate,
        num_tiles,
        ROWS=rows_per_tile,
        COLS=cols,
        NUM_PROGS=progs,
        GATE_PER_ROW=gate_per_row,
    )
    return out

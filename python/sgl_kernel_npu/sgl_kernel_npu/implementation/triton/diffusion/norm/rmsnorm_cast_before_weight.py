"""Bit-exact RMSNorm that casts before multiplying the weight.

Reference semantics (``RMSNorm(cast_x_before_out_mul=True)``)::

    xf   = x.float()
    var  = xf.pow(2).mean(-1, keepdim=True)
    norm = xf * rsqrt(var + eps)
    out  = weight * norm.to(x.dtype)

The eager chain costs five aclnn calls (``pow``, ``mean``, ``rsqrt``, ``cast``,
``mul``). This kernel computes the same chain in one pass. The row reduction
uses ``tl.sum``, which lowers to the same tree aten's ``mean`` uses on the
validated NPU architecture; the caller still verifies the first call.
"""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties

# Rows per tile for a 128-wide row; wider rows clamp this down so one tile
# stays near 32 KB of FP32 intermediates inside the 192 KB UB budget.
DEFAULT_ROWS = 16
# FP32 elements per tile; one tile holds the input and the normalized value.
SQUARE_TILE = 8192
# triton_ascend 3.2.2 returns wrong values for tiles below this many rows.
MIN_ROWS = 16
# The kernel indexes with 32-bit offsets; larger tensors need a different layout.
MAX_NUMEL = 2**31


@triton.jit(do_not_specialize=["num_tiles"])
def _rmsnorm_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    num_tiles,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    pid = tl.program_id(0)
    col = tl.arange(0, D)
    row = tl.arange(0, ROWS)
    weight = tl.load(weight_ptr + col).to(tl.float32)
    scale = 1.0 / D
    for tile in range(pid, num_tiles, NUM_PROGS):
        rows = tile * ROWS + row
        offs = rows[:, None] * D + col[None, :]
        value = tl.load(x_ptr + offs).to(tl.float32)
        variance = tl.sum(value * value, 1) * scale
        # eager casts the normalized activation before multiplying the weight
        normalized = (value * tl.rsqrt(variance[:, None] + EPS)).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + offs, normalized.to(tl.float32) * weight[None, :])


def rmsnorm_cast_before_weight(x, weight, eps):
    """Compute RMSNorm with the eager ``cast_x_before_out_mul`` rounding.

    ``x`` is a contiguous BF16 or FP16 tensor whose last dimension ``D`` is a
    power of two; ``weight`` is a contiguous ``[D]`` tensor of the same dtype.
    Rows are reduced independently, so ``x`` may have any rank of two or more.
    """
    if x.device.type != "npu":
        raise ValueError("x must be on an NPU")
    if weight.device != x.device:
        raise ValueError("weight must be on the same NPU as x")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("rmsnorm_cast_before_weight expects BF16 or FP16 input")
    if weight.dtype != x.dtype:
        raise ValueError("weight must have the same dtype as x")
    if x.ndim < 2 or not x.is_contiguous():
        raise ValueError("x must be contiguous and 2-D or more")
    numel = x.numel()
    if not 0 < numel < MAX_NUMEL:
        raise ValueError("rmsnorm_cast_before_weight expects 0 < numel < 2**31")
    dim = x.shape[-1]
    if weight.shape != (dim,) or not weight.is_contiguous():
        raise ValueError("weight must be a contiguous vector of x's last dimension")
    if dim < 2 or dim & (dim - 1):
        raise ValueError("the last dimension must be a power of two")

    rows = numel // dim
    rows_per_tile = min(DEFAULT_ROWS, max(1, SQUARE_TILE // dim))
    while rows_per_tile > 1 and rows % rows_per_tile:
        rows_per_tile //= 2
    if rows % rows_per_tile or rows_per_tile < MIN_ROWS:
        raise ValueError(
            "rows must divide into tiles of at least "
            f"{MIN_ROWS} rows; got {rows} rows of {dim} elements"
        )

    out = torch.empty_like(x)
    _, num_vectorcore = get_device_properties()
    num_tiles = rows // rows_per_tile
    progs = min(num_tiles, num_vectorcore)
    _rmsnorm_kernel[(progs,)](
        out,
        x,
        weight,
        num_tiles,
        D=dim,
        EPS=eps,
        ROWS=rows_per_tile,
        NUM_PROGS=progs,
    )
    return out

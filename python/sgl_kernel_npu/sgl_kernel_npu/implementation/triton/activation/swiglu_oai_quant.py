"""Persistent SwiGLU-OAI kernel and per-row INT8 quantization for Ascend."""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from ..utils.triton_utils import get_device_properties

_INT8_MAX = 127
# Legacy Ascend quantization-output tile width; activation still uses a full row.
_COL_BLOCK_SIZE = 1536
# Ascend group-index loads are aligned to 64 bytes, for both INT32 and INT64.
_UB_ALIGN_BYTES = 64


@triton.jit
def _swiglu_oai_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    alpha,
    limit,
    total_rows,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
):
    # MoE: derive total_rows from group_list; dense: use the passed-in total_rows.
    if HAS_GROUP_LIST:
        if (
            GROUP_LIST_TYPE == 0
        ):  # cusum (group_list length = NUM_EXPERTS, last element = total)
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS - 1).to(tl.int32)
        else:
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(
                tl.int32
            )
            total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu_oai: gate * sigmoid(gate * alpha) * (up + 1)  with clamping
        # Preserve the legacy full-row activation and its UB/resource envelope.
        # The 1536-column output tiles below do not bound these intermediates.
        # This migration does not add adaptive tiling or extend width support.
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        gate = al.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        up = al.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )
        gate = tl.minimum(gate, limit)  # clamp(-inf, limit]
        up = tl.minimum(tl.maximum(up, -limit), limit)  # clamp[-limit, limit]
        out = gate * tl.sigmoid(gate * alpha) * (up + 1.0)

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            # store scale
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = al.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                # Define all-zero rows as INT8 zero with stored scale zero;
                # only the normalization divisor is replaced.
                tmp_out = (
                    tmp_out.to(tl.float32) / tl.where(scale == 0, 1.0, scale)
                ).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = (
                    row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                )
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(
                    out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask
                )
        else:
            # store out
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


def swiglu_oai_quant(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    """Launch with inputs checked by the public operator; group values stay on device."""
    half_cols = x.shape[-1] // 2
    if group_list is not None:
        group_list = group_list.contiguous()

    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous()
    rows, cols = x.shape
    out = torch.empty(
        (rows, half_cols), dtype=torch.int8 if need_quant else x.dtype, device=x.device
    )
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows == 0:
        return out.reshape(*orig_shape[:-1], half_cols), scale
    num_experts = 1 if group_list is None else group_list.numel()
    alignment = (
        1 if group_list is None else _UB_ALIGN_BYTES // group_list.element_size()
    )
    num_experts_align = (num_experts + alignment - 1) // alignment * alignment
    # Preserve the legacy launch context: callers select x.device.
    _, num_vectorcore = get_device_properties()
    _swiglu_oai_quant_kernel[(num_vectorcore,)](
        x,
        group_list,
        out,
        scale,
        float(alpha),
        float(limit),
        rows,
        TOTAL_COLS=cols,
        HALF_COLS=half_cols,
        COL_BLOCK_SIZE=_COL_BLOCK_SIZE,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALGIN=num_experts_align,
        GROUP_LIST_TYPE=0 if group_list is None else group_list_type,
        NUM_CORES=num_vectorcore,
        DTYPE_MAX=_INT8_MAX,
        SCALE=need_quant,
        HAS_GROUP_LIST=group_list is not None,
        multibuffer=True,
    )
    return out.reshape(*orig_shape[:-1], half_cols), scale

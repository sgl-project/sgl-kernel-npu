from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _situ_deepep_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALIGNED: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    NEED_QUANT: tl.constexpr,
):
    if GROUP_LIST_TYPE == 0:
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
    else:
        offsets = tl.arange(0, NUM_EXPERTS_ALIGNED)
        mask = offsets < NUM_EXPERTS
        counts = tl.load(group_list_ptr + offsets, mask=mask, other=0).to(tl.int32)
        total_rows = tl.sum(counts)

    rows_per_core = (total_rows - 1) // NUM_CORES + 1
    row_begin = tl.program_id(0) * rows_per_core
    if row_begin >= total_rows:
        return
    row_end = tl.minimum(row_begin + rows_per_core, total_rows)

    cols = tl.arange(0, HALF_COLS)
    for row in range(row_begin, row_end):
        row_offset = row.to(tl.int64) * TOTAL_COLS
        gate = tl.load(x_ptr + row_offset + cols).to(tl.float32)
        up = tl.load(x_ptr + row_offset + HALF_COLS + cols).to(tl.float32)
        gate = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
        if DO_LINEAR_BETA:
            up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
        value = gate * up

        if NEED_QUANT:
            scale = tl.maximum(tl.max(tl.abs(value)) / 127.0, 1e-30)
            tl.store(scale_ptr + row.to(tl.int64), scale.to(scale_ptr.dtype.element_ty))
            for col_begin in range(0, HALF_COLS, COL_BLOCK_SIZE):
                block = al.extract_slice(
                    value,
                    offsets=(col_begin,),
                    sizes=(COL_BLOCK_SIZE,),
                    strides=(1,),
                )
                block = tl.floor(block.to(tl.float32) / scale + 0.5)
                block = tl.clamp(block, -128, 127).to(tl.int8)
                block_cols = col_begin + tl.arange(0, COL_BLOCK_SIZE)
                tl.store(
                    out_ptr + row.to(tl.int64) * HALF_COLS + block_cols,
                    block.to(out_ptr.dtype.element_ty),
                    mask=block_cols < HALF_COLS,
                )
        else:
            tl.store(
                out_ptr + row.to(tl.int64) * HALF_COLS + cols,
                value.to(out_ptr.dtype.element_ty),
            )


def situ_deepep(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    *,
    need_quant: bool,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply Kimi-K3 SiTU to DeepEP-packed rows on Ascend."""
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] % 2:
        raise ValueError("DeepEP SiTU input must have shape [tokens, 2 * intermediate]")
    if group_list.dtype == torch.int64:
        num_experts_aligned = (group_list.numel() + 7) // 8 * 8
    elif group_list.dtype == torch.int32:
        num_experts_aligned = (group_list.numel() + 15) // 16 * 16
    else:
        raise ValueError("group_list must use int32 or int64")

    rows, total_cols = hidden_states.shape
    half_cols = total_cols // 2
    out = torch.empty(
        (rows, half_cols),
        dtype=torch.int8 if need_quant else hidden_states.dtype,
        device=hidden_states.device,
    )
    scale = torch.empty(rows, dtype=torch.float32, device=hidden_states.device)
    _, num_vector_cores = get_device_properties()
    linear_beta_value = linear_beta if linear_beta is not None else 1.0
    _situ_deepep_kernel[(num_vector_cores,)](
        hidden_states,
        group_list,
        out,
        scale,
        TOTAL_COLS=total_cols,
        HALF_COLS=half_cols,
        COL_BLOCK_SIZE=half_cols,
        NUM_EXPERTS=group_list.numel(),
        NUM_EXPERTS_ALIGNED=num_experts_aligned,
        GROUP_LIST_TYPE=group_list_type,
        NUM_CORES=num_vector_cores,
        BETA=float(beta),
        INV_BETA=1.0 / float(beta),
        DO_LINEAR_BETA=linear_beta is not None,
        LINEAR_BETA=float(linear_beta_value),
        INV_LINEAR_BETA=1.0 / float(linear_beta_value),
        NEED_QUANT=need_quant,
        multibuffer=True,
    )
    return out, scale if need_quant else None

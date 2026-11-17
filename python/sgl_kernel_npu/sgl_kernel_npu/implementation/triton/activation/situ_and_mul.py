"""Tiled SiTU activation, preserving the existing vector-core kernel."""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice

from ..utils.triton_utils import get_device_properties

# Group counts and row scheduling in the preserved kernel use signed int32.
_INT32_MAX = (1 << 31) - 1


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": b, "multibuffer": True})
        for b in (1024, 2048, 4096, 8192)
    ],
    key=["HALF_COLS", "HAS_GROUP_LIST"],
)
@triton.jit
def _situ_and_mul_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALIGNED: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    N_ROWS,
    NUM_CORES: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # total_rows: from group_list (routed MoE) or N_ROWS (dense / shared, no group_list).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:  # cumulative counts; the last entry is the total
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS - 1).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALIGNED)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(
                tl.int32
            )
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    # rows distributed across vector cores (manual split + early return for over-provision).
    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    # H-tile over the OUTPUT dim (HALF_COLS): out[i] only needs gate[i]=x[i] and
    # up[i]=x[d+i], so every [h:h+BLOCK_H] tile is self-contained -- no full-row resident,
    # which is what keeps large d (e.g. 33792) within UB. gate = first half, up = second half.
    h_offs = tl.arange(0, BLOCK_H)
    for row_idx in range(row_begin, row_end):
        # int64 row offset: row_idx * stride stays int32 by default on triton-ascend
        # (no auto-promote), which overflows when N*d is large (e.g. N=32768, d=33792).
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        gate_base = x_ptr + row_off
        up_base = x_ptr + row_off + HALF_COLS
        out_base = out_ptr + row_idx.to(tl.int64) * HALF_COLS
        for h_start in range(0, HALF_COLS, BLOCK_H):
            h_idx = h_start + h_offs
            mask = h_idx < HALF_COLS
            gate = tl.load(gate_base + h_idx, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(up_base + h_idx, mask=mask, other=0.0).to(tl.float32)
            situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
            if DO_LINEAR_BETA:
                up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
            out = situ_a * up
            tl.store(out_base + h_idx, out.to(out_ptr.dtype.element_ty), mask=mask)


def situ_and_mul(
    x: torch.Tensor,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: Optional[int] = None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> torch.Tensor:
    """Apply SiTU to gate/up halves, writing only the valid MoE token prefix."""
    # The public operator validates the shared legacy input contract.
    has_group_list = group_list is not None
    h = x.shape[-1]
    rows = x.numel() // h
    if rows == 0:
        return torch.empty((*x.shape[:-1], h // 2), dtype=x.dtype, device=x.device)
    _, num_vector_cores = get_device_properties()
    # Ceil-div partitioning may round the final row end up by cores - 1.
    # Reserve that headroom so counts AND scheduling intermediates fit int32.
    max_rows = _INT32_MAX - (num_vector_cores - 1)
    if rows > max_rows:
        raise ValueError(
            f"row count {rows} exceeds the Triton int32 scheduling limit {max_rows}"
        )
    output = torch.empty((*x.shape[:-1], h // 2), dtype=x.dtype, device=x.device)
    # reshape alone can retain padded row strides; the kernel uses packed offsets.
    x_2d = x.reshape(rows, h).contiguous()
    num_experts = group_list.numel() if has_group_list else 1
    linear_beta_value = linear_beta if linear_beta is not None else 1.0
    _situ_and_mul_kernel[(num_vector_cores,)](
        x_2d,
        group_list if has_group_list else x_2d,
        output,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALIGNED=triton.next_power_of_2(num_experts),
        GROUP_LIST_TYPE=group_list_type if has_group_list else 0,
        N_ROWS=rows,
        NUM_CORES=num_vector_cores,
        HAS_GROUP_LIST=has_group_list,
        BETA=beta,
        INV_BETA=1.0 / beta,
        DO_LINEAR_BETA=linear_beta is not None,
        LINEAR_BETA=linear_beta_value,
        INV_LINEAR_BETA=1.0 / linear_beta_value,
    )
    return output

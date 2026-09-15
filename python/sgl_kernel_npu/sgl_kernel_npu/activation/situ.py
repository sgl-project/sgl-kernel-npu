from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _situ_and_mul_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    N_ROWS,
    NUM_CORES: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    SCALE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    # total_rows: from group_list (routed MoE) or N_ROWS (dense / shared).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:  # cusum
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(
                tl.int32
            )
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    # full-row load (d<=6144 fits UB): situ computed once, single tl.max over the row.
    cols = tl.arange(0, HALF_COLS)
    for row_idx in range(row_begin, row_end):
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        gate = tl.load(x_ptr + row_off + cols).to(tl.float32)
        up = tl.load(x_ptr + row_off + HALF_COLS + cols).to(tl.float32)
        situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
        if DO_LINEAR_BETA:
            up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
        out = situ_a * up

        if SCALE:
            scale = tl.maximum(tl.max(tl.abs(out)) / DTYPE_MAX, 1e-30)
            tl.store(
                scale_ptr + row_idx.to(tl.int64), scale.to(scale_ptr.dtype.element_ty)
            )
            # quantize in COL_BLOCK_SIZE slices (a full-row rint overflows UB, cf. swiglu_quant).
            for cb in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp = al.extract_slice(
                    out, offsets=(cb,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                tmp = tmp.to(tl.float32) / scale
                tmp = tl.floor(tmp + 0.5)
                tmp = tl.clamp(tmp, -128, 127).to(tl.int8)
                c_idx = cb + tl.arange(0, COL_BLOCK_SIZE)
                mask = c_idx < HALF_COLS
                tl.store(
                    out_ptr + row_idx.to(tl.int64) * HALF_COLS + c_idx,
                    tmp.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )
        else:
            tl.store(
                out_ptr + row_idx.to(tl.int64) * HALF_COLS + cols,
                out.to(out_ptr.dtype.element_ty),
            )


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
    NUM_EXPERTS_ALGIN: tl.constexpr,
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
        if GROUP_LIST_TYPE == 0:  # cusum
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
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


def situ_and_mul_quant(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
    need_quant: bool = True,
    quant_type: int = 0,
):
    """SituAndMul activation + fused dynamic int8 quant (d<=6144); unquant fallback (d=33792).

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared path (all rows). Required for routed MoE.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta / linear_beta: SituAndMul soft-saturation bounds (``linear_beta=None`` leaves up).
        need_quant: True -> int8 out + per-token fp32 scale; False -> activation out (scale is
            uninitialised, caller must ignore).
        quant_type: 0 = int8 (default), 1 = fp8 (deferred -> NotImplementedError).

    Returns:
        ``(out, scale)``. For d<=6144 + quant: ``out`` int8, ``scale`` fp32. For d>6144 (e.g.
        33792) or need_quant=False: ``out`` is the BF16/FP32 activation (no quant), ``scale``
        uninitialised -- quant only supports d in {3072, 6144}.
    """
    if quant_type not in (0, 1):
        raise ValueError(
            f"quant_type must be 0 (int8) or 1 (fp8), but got {quant_type}"
        )
    if need_quant and quant_type == 1:
        raise NotImplementedError(
            "fp8 (quant_type=1) is deferred: A5-only, uses npu_dynamic_mx_quant (not fusible "
            "into Triton); MoE MXFP8 downstream still WIP in sglang. Use quant_type=0 (int8)."
        )

    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    half_cols = h // 2
    # quant only for small d (3072/6144); large d (33792) -> unquant fallback.
    do_quant = need_quant and (half_cols <= 6144)
    out_dtype = torch.int8 if do_quant else x.dtype
    out = torch.empty((s, half_cols), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    if do_quant:
        _situ_and_mul_quant_kernel[(num_vectorcore,)](
            x_2d,
            group_list_arg,
            out,
            scale,
            TOTAL_COLS=h,
            HALF_COLS=half_cols,
            COL_BLOCK_SIZE=half_cols,
            NUM_EXPERTS=num_experts_arg,
            NUM_EXPERTS_ALGIN=num_experts_algin_arg,
            GROUP_LIST_TYPE=gl_type_arg,
            N_ROWS=s,
            NUM_CORES=num_vectorcore,
            HAS_GROUP_LIST=has_group_list,
            BETA=beta,
            INV_BETA=1.0 / beta,
            DO_LINEAR_BETA=do_linear_beta,
            LINEAR_BETA=linear_beta_v,
            INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
            SCALE=need_quant,
            DTYPE_MAX=127,
            multibuffer=True,
        )
    else:
        raise NotImplementedError(
            "SituAndMul quantization is only implemented for d<=6144 (int8). "
        )
    return out.reshape(*x.shape[:-1], half_cols), scale


def situ_and_mul(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
):
    """SituAndMul activation with optional MoE group_list.

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared-expert path: process ALL rows of ``x`` (no dispatch
            padding). Required only for the routed-MoE path.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta: SituAndMul beta (soft-saturation bound on the gate path).
        linear_beta: optional soft-saturation bound on the up path; ``None`` leaves ``up``.

    Returns:
        ``[..., d]`` tensor. With ``group_list``: only the first ``sum(group_list)`` rows
        are written (rest is padding). Without: all rows written.
    """
    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    out = torch.empty((s, h // 2), dtype=x.dtype, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        # dense / shared: kernel skips the group_list block (HAS_GROUP_LIST=False),
        # so these are never read -- pass harmless dummies.
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    _situ_and_mul_kernel[(num_vectorcore,)](
        x_2d,
        group_list_arg,
        out,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        NUM_EXPERTS=num_experts_arg,
        NUM_EXPERTS_ALGIN=num_experts_algin_arg,
        GROUP_LIST_TYPE=gl_type_arg,
        N_ROWS=s,
        NUM_CORES=num_vectorcore,
        HAS_GROUP_LIST=has_group_list,
        BETA=beta,
        INV_BETA=1.0 / beta,
        DO_LINEAR_BETA=do_linear_beta,
        LINEAR_BETA=linear_beta_v,
        INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
    )
    return out.reshape(*x.shape[:-1], h // 2)


@triton.jit
def _situ_kernel(
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


def situ(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    *,
    need_quant: bool,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply grouped Kimi-K3 SiTU with optional INT8 requantization."""
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] % 2:
        raise ValueError("SiTU input must have shape [tokens, 2 * intermediate]")
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
    _situ_kernel[(num_vector_cores,)](
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

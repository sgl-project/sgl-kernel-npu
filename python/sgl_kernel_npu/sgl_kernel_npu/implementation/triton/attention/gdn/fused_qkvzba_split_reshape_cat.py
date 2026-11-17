"""Triton implementation of the QKVZ/BA split-reshape-cat operators."""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties

MAX_ROWS_PER_ITER = 64
# 85 is deployed because A3 ub-size is 192kb and taking double buffer and extra memory into consideration.
UB_SPECIFICATION = 85


@triton.jit(do_not_specialize=["total_rows", "rows_per_vec"])
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    total_rows,
    rows_per_vec,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    BA_OUT_ROW_STRIDE: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    vec_id = tl.program_id(0)

    V_HEADS_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    V_DIM_PER_QK: tl.constexpr = V_HEADS_PER_QK * HEAD_V
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_DIM_PER_QK * 2
    BA_DIM_T: tl.constexpr = V_HEADS_PER_QK * 2

    Q_TOTAL: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    K_TOTAL: tl.constexpr = NUM_HEADS_QK * HEAD_QK

    row_start = vec_id * rows_per_vec
    row_end = min(row_start + rows_per_vec, total_rows)

    row_offset = row_start

    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        for head_id in tl.static_range(NUM_HEADS_QK):
            src_head_offset = head_id * QKVZ_DIM_T

            q_range = tl.arange(0, HEAD_QK)
            q_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE
                + src_head_offset
                + q_range[None, :]
            )
            q_dst = (
                row_indices[:, None] * QKV_ROW_STRIDE
                + head_id * HEAD_QK
                + q_range[None, :]
            )
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            k_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE
                + src_head_offset
                + HEAD_QK
                + q_range[None, :]
            )
            k_dst = (
                row_indices[:, None] * QKV_ROW_STRIDE
                + Q_TOTAL
                + head_id * HEAD_QK
                + q_range[None, :]
            )
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            v_range = tl.arange(0, V_DIM_PER_QK)
            v_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE
                + src_head_offset
                + HEAD_QK * 2
                + v_range[None, :]
            )
            v_dst = (
                row_indices[:, None] * QKV_ROW_STRIDE
                + Q_TOTAL
                + K_TOTAL
                + head_id * V_DIM_PER_QK
                + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            z_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE
                + src_head_offset
                + HEAD_QK * 2
                + V_DIM_PER_QK
                + v_range[None, :]
            )
            z_dst = (
                row_indices[:, None] * Z_ROW_STRIDE
                + head_id * V_DIM_PER_QK
                + v_range[None, :]
            )
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            b_range = tl.arange(0, V_HEADS_PER_QK)
            ba_head_offset = head_id * BA_DIM_T
            b_src = (
                row_indices[:, None] * BA_ROW_STRIDE + ba_head_offset + b_range[None, :]
            )
            b_dst = (
                row_indices[:, None] * BA_OUT_ROW_STRIDE
                + head_id * V_HEADS_PER_QK
                + b_range[None, :]
            )
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            a_src = (
                row_indices[:, None] * BA_ROW_STRIDE
                + ba_head_offset
                + V_HEADS_PER_QK
                + b_range[None, :]
            )
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


def _check_inputs(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
    *,
    require_nonempty,
):
    """Reject inputs that would otherwise be read out of bounds or mis-strided."""
    assert (
        mixed_qkvz.dim() == 2
    ), f"mixed_qkvz must be 2-D, got shape={tuple(mixed_qkvz.shape)}"
    assert (
        mixed_ba.dim() == 2
    ), f"mixed_ba must be 2-D, got shape={tuple(mixed_ba.shape)}"
    assert mixed_qkvz.shape[0] == mixed_ba.shape[0], (
        f"row count mismatch: mixed_qkvz={tuple(mixed_qkvz.shape)}, "
        f"mixed_ba={tuple(mixed_ba.shape)}"
    )
    assert (
        mixed_qkvz.is_contiguous()
    ), f"mixed_qkvz must be contiguous, got strides={mixed_qkvz.stride()}"
    assert (
        mixed_ba.is_contiguous()
    ), f"mixed_ba must be contiguous, got strides={mixed_ba.stride()}"
    assert num_heads_v % num_heads_qk == 0, (
        f"num_heads_v ({num_heads_v}) must be a multiple of "
        f"num_heads_qk ({num_heads_qk})"
    )
    expected_qkvz_width = 2 * num_heads_qk * head_qk + 2 * num_heads_v * head_v
    assert mixed_qkvz.shape[1] == expected_qkvz_width, (
        f"mixed_qkvz row width {mixed_qkvz.shape[1]} != expected "
        f"{expected_qkvz_width} for num_heads_qk={num_heads_qk}, "
        f"num_heads_v={num_heads_v}, head_qk={head_qk}, head_v={head_v}"
    )
    assert (
        mixed_ba.shape[1] == 2 * num_heads_v
    ), f"mixed_ba row width {mixed_ba.shape[1]} != expected {2 * num_heads_v}"
    if require_nonempty:
        assert mixed_qkvz.shape[0] > 0, (
            f"fused_qkvzba_split_reshape_cat requires a non-empty batch, "
            f"got mixed_qkvz.shape={tuple(mixed_qkvz.shape)}"
        )


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """Split per-head-group interleaved QKVZ/BA projections."""
    _check_inputs(
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        require_nonempty=True,
    )
    batch, seq_len = mixed_qkvz.shape[0], 1
    total_rows = batch * seq_len

    v_heads_per_qk = num_heads_v // num_heads_qk
    v_dim_per_qk = v_heads_per_qk * head_v
    qkvz_dim_t = head_qk * 2 + v_dim_per_qk * 2
    ba_dim_t = v_heads_per_qk * 2

    qkvz_row_stride = num_heads_qk * qkvz_dim_t
    ba_row_stride = num_heads_qk * ba_dim_t
    qkv_row_stride = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    z_row_stride = num_heads_v * head_v
    ba_out_row_stride = num_heads_v

    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )

    num_vectorcore = get_device_properties()[1]

    grid_size = min(num_vectorcore, total_rows)
    grid_size = max(1, grid_size)

    rows_per_vec = triton.cdiv(total_rows, grid_size)

    ub_size = UB_SPECIFICATION * 1024 // mixed_qkvz.element_size()

    elements_per_row = (
        qkvz_row_stride  # mixed_qkvz
        + ba_row_stride  # mixed_ba
        + qkv_row_stride  # mixed_qkv
        + z_row_stride  # z
        + ba_out_row_stride * 2  # one for b, and the other for a
    )
    rows_per_iter = max(1, ub_size // elements_per_row)
    rows_per_iter = triton.next_power_of_2(rows_per_iter)
    rows_per_iter = min(rows_per_iter, rows_per_vec, MAX_ROWS_PER_ITER)

    grid = (grid_size, 1)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        total_rows,
        rows_per_vec,
        qkvz_row_stride,
        ba_row_stride,
        qkv_row_stride,
        z_row_stride,
        ba_out_row_stride,
        rows_per_iter,
    )
    return mixed_qkv, z, b, a


# UB pressure scales with the number of rows moved per iteration, and the
# affordable tile differs per generation. Tiles 8..32 correspond to roughly
# 0.2..0.4 of the 85 KB budget the wrapper used before. Let autotune pick one: a
# candidate that does not fit fails at compile time with a UB overflow, is
# dropped, and the rest are benchmarked. rows_per_vec keeps decode (one row per
# program) from inheriting the prefill choice.
@triton.autotune(
    configs=[triton.Config({"ROWS_PER_ITER": rows}) for rows in (8, 16, 32)],
    key=["NUM_HEADS_QK", "NUM_HEADS_V", "HEAD_QK", "HEAD_V", "rows_per_vec"],
)
@triton.jit
def fused_qkvzba_split_reshape_cat_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    total_rows,
    rows_per_vec,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    BA_OUT_ROW_STRIDE: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    vec_id = tl.program_id(0)

    TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V

    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)
    row_offset = row_start

    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        for col_start in tl.static_range(0, TOTAL_Q, BLOCK_SIZE_COL):
            col_range = tl.arange(0, BLOCK_SIZE_COL) + col_start
            col_mask = col_range < TOTAL_Q
            combined_mask = row_mask[:, None] & col_mask[None, :]

            q_src = row_indices[:, None] * QKVZ_ROW_STRIDE + col_range[None, :]
            q_dst = row_indices[:, None] * QKV_ROW_STRIDE + col_range[None, :]

            q_data = tl.load(mixed_qkvz + q_src, mask=combined_mask)
            tl.store(mixed_qkv + q_dst, q_data, mask=combined_mask)

        for col_start in tl.static_range(0, TOTAL_K, BLOCK_SIZE_COL):
            col_range = tl.arange(0, BLOCK_SIZE_COL) + col_start
            col_mask = col_range < TOTAL_K
            combined_mask = row_mask[:, None] & col_mask[None, :]

            k_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE + TOTAL_Q + col_range[None, :]
            )
            k_dst = row_indices[:, None] * QKV_ROW_STRIDE + TOTAL_Q + col_range[None, :]

            k_data = tl.load(mixed_qkvz + k_src, mask=combined_mask)
            tl.store(mixed_qkv + k_dst, k_data, mask=combined_mask)

        for col_start in tl.static_range(0, TOTAL_V, BLOCK_SIZE_COL):
            col_range = tl.arange(0, BLOCK_SIZE_COL) + col_start
            col_mask = col_range < TOTAL_V
            combined_mask = row_mask[:, None] & col_mask[None, :]

            v_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE
                + TOTAL_Q
                + TOTAL_K
                + col_range[None, :]
            )
            v_dst = (
                row_indices[:, None] * QKV_ROW_STRIDE
                + TOTAL_Q
                + TOTAL_K
                + col_range[None, :]
            )

            v_data = tl.load(mixed_qkvz + v_src, mask=combined_mask)
            tl.store(mixed_qkv + v_dst, v_data, mask=combined_mask)

        for col_start in tl.static_range(0, TOTAL_V, BLOCK_SIZE_COL):
            col_range = tl.arange(0, BLOCK_SIZE_COL) + col_start
            col_mask = col_range < TOTAL_V
            combined_mask = row_mask[:, None] & col_mask[None, :]

            z_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE
                + TOTAL_Q
                + TOTAL_K
                + TOTAL_V
                + col_range[None, :]
            )
            z_dst = row_indices[:, None] * Z_ROW_STRIDE + col_range[None, :]

            z_data = tl.load(mixed_qkvz + z_src, mask=combined_mask)
            tl.store(z + z_dst, z_data, mask=combined_mask)

        ba_range = tl.arange(0, NUM_HEADS_V)
        b_src = row_indices[:, None] * BA_ROW_STRIDE + ba_range[None, :]
        b_dst = row_indices[:, None] * BA_OUT_ROW_STRIDE + ba_range[None, :]

        b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
        tl.store(b + b_dst, b_data, mask=row_mask[:, None])

        a_src = row_indices[:, None] * BA_ROW_STRIDE + NUM_HEADS_V + ba_range[None, :]
        a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
        tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
    block_size_col=1024,
):
    """Split contiguous [q|k|v|z] / [b|a] projections."""
    _check_inputs(
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        require_nonempty=False,
    )
    batch, seq_len = mixed_qkvz.shape[0], 1
    total_rows = batch * seq_len

    total_q = num_heads_qk * head_qk
    total_k = num_heads_qk * head_qk
    total_v = num_heads_v * head_v

    qkvz_row_stride = total_q + total_k + total_v + total_v
    ba_row_stride = num_heads_v * 2
    qkv_row_stride = total_q + total_k + total_v
    z_row_stride = num_heads_v * head_v
    ba_out_row_stride = num_heads_v

    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_row_stride],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)

    num_vectorcore = get_device_properties()[1]
    grid_size = min(num_vectorcore, total_rows)
    grid_size = max(1, grid_size)

    rows_per_vec = triton.cdiv(total_rows, grid_size)

    grid = (grid_size,)
    fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        total_rows,
        rows_per_vec,
        qkvz_row_stride,
        ba_row_stride,
        qkv_row_stride,
        z_row_stride,
        ba_out_row_stride,
        BLOCK_SIZE_COL=block_size_col,
    )

    return mixed_qkv, z, b, a

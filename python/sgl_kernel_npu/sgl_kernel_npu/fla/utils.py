# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Copied from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/utils.py
# -*- coding: utf-8 -*-

import contextlib
import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

is_gather_supported = hasattr(triton.language, "gather")
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None

else:
    gather = tl.gather


def custom_device_ctx(index: int):
    return torch.npu.device(index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    cu_seqlens_i64 = cu_seqlens.to(torch.int64)
    return cu_seqlens_i64[1:] - cu_seqlens_i64[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in prepare_lens(cu_seqlens).unbind()
        ]
    )


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


def fused_qkvzba_split_reshape_cat_torch(
    mixed_qkvz: torch.Tensor,  # [B, 3072]
    mixed_ba: torch.Tensor,  # [B, 16]
    num_heads_qk: int = 4,
    num_heads_v: int = 8,
    head_qk: int = 128,
    head_v: int = 128,
):
    B = mixed_qkvz.shape[0]
    v_group_size = num_heads_v // num_heads_qk  # = 2

    # Step 1: Reshape to [B, num_heads_qk, per_head_dim]
    per_head_dim = 2 * head_qk + 2 * v_group_size * head_v  # 768
    x = mixed_qkvz.view(B, num_heads_qk, per_head_dim)

    # Extract components per head
    q = x[:, :, :head_qk]  # [B, 4, 128]
    k = x[:, :, head_qk : 2 * head_qk]  # [B, 4, 128]
    v_groups = x[:, :, 2 * head_qk : 2 * head_qk + v_group_size * head_v]  # [B, 4, 256]
    z_groups = x[:, :, 2 * head_qk + v_group_size * head_v :]  # [B, 4, 256]

    # Reshape V and Z to [B, num_heads_v, head_v]
    v = v_groups.reshape(B, num_heads_v, head_v)  # [B, 8, 128]
    z = z_groups.reshape(B, num_heads_v, head_v)  # [B, 8, 128]

    # Build mixed_qkv = [Q_flat, K_flat, V_flat]
    # Q_flat: concatenate all q heads → [B, 4*128]
    q_flat = q.reshape(B, -1)
    k_flat = k.reshape(B, -1)
    v_flat = v.reshape(B, -1)
    mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=1)  # [B, 2048]

    # Process mixed_ba: [B, 16] → view as [B, 4, 4] → split b/a
    ba = mixed_ba.view(B, num_heads_qk, 2 * v_group_size)  # [B, 4, 4]
    b = ba[:, :, :v_group_size].reshape(B, num_heads_v)  # [B, 8]
    a = ba[:, :, v_group_size:].reshape(B, num_heads_v)  # [B, 8]

    return mixed_qkv, z, b, a


from sgl_kernel_npu.utils.triton_utils import get_device_properties

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


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
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

"""Split QKV + optional RMSNorm + RoPE using a position-indexed cos/sin cache (packed layout).

Provides ``split_qkv_rmsnorm_rope_pos_cache_half_npu``. Use this when you have a global
cos/sin cache and per-row position indices; use ``split_qkv_rmsnorm_rope`` in the same
package when you already have sin/cos tensors of shape [B, rope_dim].
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def fp32_to_bf16_rne(x):
    """Deterministic FP32 -> BF16 (round-to-nearest-even). Keep ONLY for RMSNorm cast."""
    u = tl.cast(x, tl.uint32, bitcast=True)
    lsb = (u >> 16) & 1
    u = (u + (0x7FFF + lsb)) & 0xFFFF0000
    return tl.cast(u, tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def split_qkv_rmsnorm_rope_half_pos_cache_kernel(
    input_ptr,
    pos_ptr,  # [B]
    cos_sin_cache_ptr,  # [max_seq, ROPE_DIM] layout: [cos_half, sin_half]
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    q_block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    q_block_n: tl.constexpr,  # q_block_size // head_dim
    k_block_n: tl.constexpr,  # kv_block_size // head_dim
    bias: tl.constexpr,
    norms: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    cos_sin_stride0: tl.constexpr,
    cast_norm_to_bf16: tl.constexpr,  # ONLY this cast uses fp32_to_bf16_rne
    max_seq: tl.constexpr,  # cos_sin_cache rows; clamp position to [0, max_seq-1] (graph-safe)
):
    """Triton kernel: split QKV from concatenated input, optional RMSNorm on Q/K, RoPE (half cache), copy V."""
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)

    q_ty = q_ptr.dtype.element_ty
    k_ty = k_ptr.dtype.element_ty
    v_ty = v_ptr.dtype.element_ty

    if norms:
        q_w = tl.load(q_weight_ptr + tl.arange(0, head_dim)).to(tl.float32)
        k_w = tl.load(k_weight_ptr + tl.arange(0, head_dim)).to(tl.float32)
    if bias:
        q_b = tl.load(q_bias_ptr + tl.arange(0, head_dim)).to(tl.float32)
        k_b = tl.load(k_bias_ptr + tl.arange(0, head_dim)).to(tl.float32)

    # Single loop over batch: for each row process Q, K, V (avoids duplication, same cos/sin per row)
    for row_idx in tl.range(row_pid, batch_size, row_step):
        # ---- load cos/sin once per row (shared by Q and K RoPE) ----
        p = tl.load(pos_ptr + row_idx).to(tl.int32)
        # Clamp on-device: avoids OOB cos/sin loads without Python .any()/.item() sync (breaks NPU graph capture).
        p = tl.maximum(p, 0)
        p = tl.minimum(p, max_seq - 1)
        base = p * cos_sin_stride0
        offs = tl.arange(0, half_rope_dim)
        cos_f = (
            tl.load(cos_sin_cache_ptr + base + offs)
            .to(tl.float32)
            .reshape(1, half_rope_dim)
        )
        sin_f = (
            tl.load(cos_sin_cache_ptr + base + half_rope_dim + offs)
            .to(tl.float32)
            .reshape(1, half_rope_dim)
        )

        # --- Q ---
        in_off_q = row_idx * total_hidden_size
        outq_off = row_idx * q_hidden_size
        col = col_pid * q_block_size + tl.arange(0, q_block_size)
        mask = col < q_hidden_size

        x = tl.load(input_ptr + in_off_q + col, mask=mask, other=0.0).to(tl.float32)
        x = x.reshape(q_block_n, head_dim)

        if norms:
            var = tl.sum(x * x, axis=1) / (1.0 * head_dim)
            inv_std = tl.rsqrt(var + eps).reshape(q_block_n, 1)
            y = x * inv_std
            if bias:
                y = y * q_w + q_b
            else:
                y = y * q_w
        else:
            y = x

        if cast_norm_to_bf16:
            y_base = fp32_to_bf16_rne(y)
        else:
            y_base = y

        y_rot = al.extract_slice(
            y_base, offsets=(0, 0), sizes=(q_block_n, rope_dim), strides=(1, 1)
        )
        y_rot_f = y_rot.to(tl.float32)
        x1 = al.extract_slice(
            y_rot_f, offsets=(0, 0), sizes=(q_block_n, half_rope_dim), strides=(1, 1)
        )
        x2 = al.extract_slice(
            y_rot_f,
            offsets=(0, half_rope_dim),
            sizes=(q_block_n, half_rope_dim),
            strides=(1, 1),
        )
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x2 * cos_f + x1 * sin_f
        ro1 = o1.to(tl.bfloat16)
        ro2 = o2.to(tl.bfloat16)
        roped = tl.zeros((q_block_n, rope_dim), dtype=tl.bfloat16)
        roped = al.insert_slice(
            roped, ro1, offsets=(0, 0), sizes=(q_block_n, half_rope_dim), strides=(1, 1)
        )
        roped = al.insert_slice(
            roped,
            ro2,
            offsets=(0, half_rope_dim),
            sizes=(q_block_n, half_rope_dim),
            strides=(1, 1),
        )
        if cast_norm_to_bf16:
            y_out = y_base
        else:
            y_out = y_base.to(tl.bfloat16)
        y_out = al.insert_slice(
            y_out, roped, offsets=(0, 0), sizes=(q_block_n, rope_dim), strides=(1, 1)
        )
        tl.store(
            q_ptr + outq_off + col, y_out.reshape(q_block_size).to(q_ty), mask=mask
        )

        # --- K ---
        in_off_k = row_idx * total_hidden_size + q_hidden_size
        outk_off = row_idx * kv_hidden_size
        col = col_pid * kv_block_size + tl.arange(0, kv_block_size)
        mask = col < kv_hidden_size

        x = tl.load(input_ptr + in_off_k + col, mask=mask, other=0.0).to(tl.float32)
        x = x.reshape(k_block_n, head_dim)

        if norms:
            var = tl.sum(x * x, axis=1) / (1.0 * head_dim)
            inv_std = tl.rsqrt(var + eps).reshape(k_block_n, 1)
            y = x * inv_std
            if bias:
                y = y * k_w + k_b
            else:
                y = y * k_w
        else:
            y = x

        if cast_norm_to_bf16:
            y_base = fp32_to_bf16_rne(y)
        else:
            y_base = y

        y_rot = al.extract_slice(
            y_base, offsets=(0, 0), sizes=(k_block_n, rope_dim), strides=(1, 1)
        )
        y_rot_f = y_rot.to(tl.float32)
        x1 = al.extract_slice(
            y_rot_f, offsets=(0, 0), sizes=(k_block_n, half_rope_dim), strides=(1, 1)
        )
        x2 = al.extract_slice(
            y_rot_f,
            offsets=(0, half_rope_dim),
            sizes=(k_block_n, half_rope_dim),
            strides=(1, 1),
        )
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x2 * cos_f + x1 * sin_f
        ro1 = o1.to(tl.bfloat16)
        ro2 = o2.to(tl.bfloat16)
        roped = tl.zeros((k_block_n, rope_dim), dtype=tl.bfloat16)
        roped = al.insert_slice(
            roped, ro1, offsets=(0, 0), sizes=(k_block_n, half_rope_dim), strides=(1, 1)
        )
        roped = al.insert_slice(
            roped,
            ro2,
            offsets=(0, half_rope_dim),
            sizes=(k_block_n, half_rope_dim),
            strides=(1, 1),
        )
        if cast_norm_to_bf16:
            y_out = y_base
        else:
            y_out = y_base.to(tl.bfloat16)
        y_out = al.insert_slice(
            y_out, roped, offsets=(0, 0), sizes=(k_block_n, rope_dim), strides=(1, 1)
        )
        tl.store(
            k_ptr + outk_off + col, y_out.reshape(kv_block_size).to(k_ty), mask=mask
        )

        # --- V ---
        in_off_v = row_idx * total_hidden_size + q_hidden_size + kv_hidden_size
        outv_off = row_idx * kv_hidden_size
        col = col_pid * kv_block_size + tl.arange(0, kv_block_size)
        mask = col < kv_hidden_size
        v = tl.load(input_ptr + in_off_v + col, mask=mask, other=0.0)
        tl.store(v_ptr + outv_off + col, v.to(v_ty), mask=mask)


# Largest q_hidden_size whose single-row tile still fits the Unified Buffer on the wide
# grid (measured on 910B3: 8192 compiles, 12288 raises MLIRCompilationError).
_WIDE_GRID_MAX_Q_HIDDEN = 8192

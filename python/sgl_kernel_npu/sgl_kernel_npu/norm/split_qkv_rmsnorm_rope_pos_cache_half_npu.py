"""Split QKV + optional RMSNorm + RoPE using a position-indexed cos/sin cache (packed layout).

Provides ``split_qkv_rmsnorm_rope_pos_cache_half_npu``. Use this when you have a global
cos/sin cache and per-row position indices; use ``split_qkv_rmsnorm_rope`` in the same
package when you already have sin/cos tensors of shape [B, rope_dim].
"""

import torch
import triton
import triton.language as tl
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

        y_rot = tl.extract_slice(
            y_base, offsets=(0, 0), sizes=(q_block_n, rope_dim), strides=(1, 1)
        )
        y_rot_f = y_rot.to(tl.float32)
        x1 = tl.extract_slice(
            y_rot_f, offsets=(0, 0), sizes=(q_block_n, half_rope_dim), strides=(1, 1)
        )
        x2 = tl.extract_slice(
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
        roped = tl.insert_slice(
            roped, ro1, offsets=(0, 0), sizes=(q_block_n, half_rope_dim), strides=(1, 1)
        )
        roped = tl.insert_slice(
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
        y_out = tl.insert_slice(
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

        y_rot = tl.extract_slice(
            y_base, offsets=(0, 0), sizes=(k_block_n, rope_dim), strides=(1, 1)
        )
        y_rot_f = y_rot.to(tl.float32)
        x1 = tl.extract_slice(
            y_rot_f, offsets=(0, 0), sizes=(k_block_n, half_rope_dim), strides=(1, 1)
        )
        x2 = tl.extract_slice(
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
        roped = tl.insert_slice(
            roped, ro1, offsets=(0, 0), sizes=(k_block_n, half_rope_dim), strides=(1, 1)
        )
        roped = tl.insert_slice(
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
        y_out = tl.insert_slice(
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


def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor: torch.Tensor,  # [B, q_hidden + 2*kv_hidden]
    positions: torch.Tensor,  # [B]
    cos_sin_cache: torch.Tensor,  # [max_seq, rope_dim] layout [cos_block, sin_block]
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float = None,
    q_weight: torch.Tensor = None,
    k_weight: torch.Tensor = None,
    q_bias: torch.Tensor = None,
    k_bias: torch.Tensor = None,
    rope_dim: int = None,
    cast_norm_to_bf16: bool = True,  # cast norm result to bf16 before RoPE
):
    """Split QKV from concatenated input, optional RMSNorm on Q/K, RoPE via position-indexed cache, copy V.

    Input shape is [B, q_hidden_size + 2*kv_hidden_size] (Q|K|V concatenated). Outputs are
    separate Q, K, V tensors with optional RMSNorm and rotary position embedding applied to Q/K.
    RoPE cos/sin are read from ``cos_sin_cache`` at indices ``positions[b]`` for each row b.
    Cache layout is packed: first ``rope_dim // 2`` columns are cos(θ), next ``rope_dim // 2`` are sin(θ).

    See also:
        split_qkv_rmsnorm_rope: variant that takes pre-indexed sin/cos tensors [B, rope_dim].

    Args:
        input_tensor: Concatenated QKV hidden states, shape [B, q_hidden_size + 2*kv_hidden_size].
        positions: Position index per batch item, shape [B], dtype int32 or int64. Must be in [0, max_seq).
        cos_sin_cache: RoPE cos/sin cache, shape [max_seq, rope_dim]. Layout: [0 : rope_dim//2] = cos,
            [rope_dim//2 : rope_dim] = sin.
        q_hidden_size: Query hidden size.
        kv_hidden_size: Key/Value hidden size (each).
        head_dim: Head dimension (must be power of 2).
        eps: RMSNorm epsilon; if None, norm is skipped.
        q_weight: Optional Q RMSNorm weight (length >= head_dim when eps is not None).
        k_weight: Optional K RMSNorm weight (length >= head_dim when eps is not None).
        q_bias: Optional Q RMSNorm bias (length >= head_dim when provided).
        k_bias: Optional K RMSNorm bias (length >= head_dim when provided).
        rope_dim: RoPE dimension (default head_dim). Must be even and <= head_dim.
        cast_norm_to_bf16: If True, cast norm output to bf16 with RNE before RoPE; else keep in fp32 then cast.

    Returns:
        Tuple of (q_out, k_out, v_out), shapes [B, q_hidden_size], [B, kv_hidden_size], [B, kv_hidden_size].
    """
    _, num_vectorcore = get_device_properties()
    assert input_tensor.dim() == 2
    B, total_hidden = input_tensor.shape

    if rope_dim is None:
        rope_dim = head_dim
    assert rope_dim % 2 == 0 and rope_dim <= head_dim

    expected_total = q_hidden_size + 2 * kv_hidden_size
    assert total_hidden == expected_total

    pos = positions
    assert pos.numel() == B, f"positions must be [B], got numel={pos.numel()} B={B}"
    if pos.dtype not in (torch.int32, torch.int64):
        pos = pos.to(torch.int32)
    pos = pos.contiguous()

    cache = cos_sin_cache.contiguous()
    max_seq = cache.shape[0]
    # Bounds check: prevent out-of-bounds read in kernel (security)
    if (pos < 0).any() or (pos >= max_seq).any():
        bad = (pos < 0) | (pos >= max_seq)
        idx = bad.nonzero(as_tuple=True)[0]
        raise ValueError(
            f"Position indices must be in [0, {max_seq}); "
            f"got invalid at batch indices {idx.tolist()[:10]}{'...' if bad.sum().item() > 10 else ''} "
            f"(min={pos.min().item()}, max={pos.max().item()})."
        )
    stride0 = cache.stride(0)

    kv_block_size = triton.next_power_of_2(head_dim)
    assert kv_block_size == head_dim, "this kernel assumes head_dim is power-of-2"
    assert q_hidden_size % kv_hidden_size == 0
    q_block_size = (q_hidden_size // kv_hidden_size) * head_dim

    q_block_n = q_block_size // head_dim
    k_block_n = kv_block_size // head_dim  # usually 1

    q_out = torch.empty(
        (B, q_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )
    k_out = torch.empty(
        (B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )
    v_out = torch.empty(
        (B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )

    n_cols = kv_hidden_size // kv_block_size
    n_rows = (num_vectorcore + n_cols - 1) // n_cols

    bias = q_bias is not None
    norms = eps is not None

    # Bounds check: kernel loads weight/bias with head_dim elements (security)
    if norms:
        if q_weight is None or q_weight.numel() < head_dim:
            raise ValueError(
                f"When using RMSNorm (eps is not None), q_weight must have at least head_dim={head_dim} elements, "
                f"got {q_weight.numel() if q_weight is not None else 0}."
            )
        if k_weight is None or k_weight.numel() < head_dim:
            raise ValueError(
                f"When using RMSNorm (eps is not None), k_weight must have at least head_dim={head_dim} elements, "
                f"got {k_weight.numel() if k_weight is not None else 0}."
            )
    if bias:
        if q_bias is None or q_bias.numel() < head_dim:
            raise ValueError(
                f"When using bias (q_bias provided), q_bias must have at least head_dim={head_dim} elements, "
                f"got {q_bias.numel() if q_bias is not None else 0}."
            )
        if k_bias is None or k_bias.numel() < head_dim:
            raise ValueError(
                f"When using bias (k_bias provided), k_bias must have at least head_dim={head_dim} elements, "
                f"got {k_bias.numel() if k_bias is not None else 0}."
            )

    split_qkv_rmsnorm_rope_half_pos_cache_kernel[(n_rows, n_cols, 1)](
        input_tensor,
        pos,
        cache,
        q_out,
        k_out,
        v_out,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        B,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        total_hidden_size=expected_total,
        eps=eps if eps is not None else 0.0,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_block_n=q_block_n,
        k_block_n=k_block_n,
        bias=bias,
        norms=norms,
        head_dim=head_dim,
        rope_dim=rope_dim,
        half_rope_dim=rope_dim // 2,
        cos_sin_stride0=stride0,
        cast_norm_to_bf16=cast_norm_to_bf16,
    )

    return q_out, k_out, v_out

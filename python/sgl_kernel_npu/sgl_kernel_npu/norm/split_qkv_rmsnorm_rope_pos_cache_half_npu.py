import triton
import triton.language as tl
import torch
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

    # =========================
    # Q
    # =========================
    if norms:
        q_w = tl.load(q_weight_ptr + tl.arange(0, head_dim)).to(tl.float32)
    if bias:
        q_b = tl.load(q_bias_ptr + tl.arange(0, head_dim)).to(tl.float32)

    in_off = row_pid * total_hidden_size
    outq_off = row_pid * q_hidden_size
    in_off_step = row_step * total_hidden_size
    outq_off_step = row_step * q_hidden_size

    for row_idx in tl.range(row_pid, batch_size, row_step):
        col = col_pid * q_block_size + tl.arange(0, q_block_size)
        mask = col < q_hidden_size

        x = tl.load(input_ptr + in_off + col, mask=mask, other=0.0).to(tl.float32)
        x = x.reshape(q_block_n, head_dim)

        # RMSNorm (fp32)
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

        # y_base: fp32 or bf16
        if cast_norm_to_bf16:
            # keep strict RNE here ONLY
            y_base = fp32_to_bf16_rne(y)  # bf16
        else:
            y_base = y  # fp32

        # ---- load cos/sin half (fp32) ----
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

        # ---- RoPE half on first rope_dim (fp32 math; casts use .to(bf16)) ----
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

        # output base bf16 (tail passthrough)
        if cast_norm_to_bf16:
            y_out = y_base  # already bf16
        else:
            y_out = y_base.to(tl.bfloat16)  # fast cast

        y_out = tl.insert_slice(
            y_out, roped, offsets=(0, 0), sizes=(q_block_n, rope_dim), strides=(1, 1)
        )
        tl.store(
            q_ptr + outq_off + col, y_out.reshape(q_block_size).to(q_ty), mask=mask
        )

        in_off += in_off_step
        outq_off += outq_off_step

    # =========================
    # K
    # =========================
    if norms:
        k_w = tl.load(k_weight_ptr + tl.arange(0, head_dim)).to(tl.float32)
    if bias:
        k_b = tl.load(k_bias_ptr + tl.arange(0, head_dim)).to(tl.float32)

    in_off = row_pid * total_hidden_size + q_hidden_size
    outk_off = row_pid * kv_hidden_size
    outk_off_step = row_step * kv_hidden_size

    for row_idx in tl.range(row_pid, batch_size, row_step):
        col = col_pid * kv_block_size + tl.arange(0, kv_block_size)
        mask = col < kv_hidden_size

        x = tl.load(input_ptr + in_off + col, mask=mask, other=0.0).to(tl.float32)
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
            y_base = fp32_to_bf16_rne(y)  # strict here only
        else:
            y_base = y

        # cos/sin half
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

        # rope
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

        in_off += in_off_step
        outk_off += outk_off_step

    # =========================
    # V
    # =========================
    in_off = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    outv_off = row_pid * kv_hidden_size
    outv_off_step = row_step * kv_hidden_size

    for _ in tl.range(row_pid, batch_size, row_step):
        col = col_pid * kv_block_size + tl.arange(0, kv_block_size)
        mask = col < kv_hidden_size
        v = tl.load(input_ptr + in_off + col, mask=mask, other=0.0)
        tl.store(v_ptr + outv_off + col, v.to(v_ty), mask=mask)
        in_off += in_off_step
        outv_off += outv_off_step


def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor: torch.Tensor,  # [B, q_hidden + 2*kv_hidden]
    positions: torch.Tensor,  # [B]
    cos_sin_cache: torch.Tensor,  # [max_seq, rope_dim] layout [cos_half, sin_half]
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float = None,
    q_weight: torch.Tensor = None,
    k_weight: torch.Tensor = None,
    q_bias: torch.Tensor = None,
    k_bias: torch.Tensor = None,
    rope_dim: int = None,
    cast_norm_to_bf16: bool = True, # cast norm result to bf16 before RoPE
):
    """Split concatenated QKV input, optionally apply RMSNorm, then RoPE using position cache (half layout).

    The input is [B, q_hidden + 2*kv_hidden] (Q|K|V concatenated). Outputs are separate
    Q, K, V tensors with optional RMSNorm and rotary position embedding applied.
    cos_sin_cache uses half-angle layout [cos_half, sin_half] per dimension.

    Args:
        input_tensor: Concatenated QKV hidden states, shape [B, q_hidden_size + 2*kv_hidden_size].
        positions: Position indices per batch item, shape [B].
        cos_sin_cache: RoPE cos/sin cache, shape [max_seq, rope_dim], layout [cos_half, sin_half].
        q_hidden_size: Query hidden size.
        kv_hidden_size: Key/Value hidden size (each).
        head_dim: Head dimension (must be power of 2).
        eps: RMSNorm epsilon; if None, norm is skipped.
        q_weight: Optional Q RMSNorm weight.
        k_weight: Optional K RMSNorm weight.
        q_bias: Optional Q RMSNorm bias.
        k_bias: Optional K RMSNorm bias.
        rope_dim: RoPE dimension (default head_dim).
        cast_norm_to_bf16: If True, cast norm output to bf16 before RoPE.

    Returns:
        Tuple of (q_out, k_out, v_out), each shape [B, *] with respective hidden sizes.
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
    stride0 = cache.stride(0)

    kv_block_size = triton.next_power_of_2(head_dim)
    assert kv_block_size == head_dim, "this kernel assumes head_dim is power-of-2"
    assert q_hidden_size % kv_hidden_size == 0
    q_block_size = (q_hidden_size // kv_hidden_size) * head_dim

    q_block_n = q_block_size // head_dim
    k_block_n = kv_block_size // head_dim  # usually 1

    q_out = torch.empty((B, q_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype)
    k_out = torch.empty((B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype)
    v_out = torch.empty((B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype)

    n_cols = kv_hidden_size // kv_block_size
    n_rows = (num_vectorcore + n_cols - 1) // n_cols

    bias = q_bias is not None
    norms = eps is not None

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


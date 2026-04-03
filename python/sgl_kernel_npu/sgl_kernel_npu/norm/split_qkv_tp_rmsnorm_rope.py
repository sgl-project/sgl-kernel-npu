import torch
import torch.distributed as dist
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _split_qkv_and_compute_local_qk_var_kernel(
    input_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    qk_var_ptr,
    batch_size,
    q_cols: tl.constexpr,
    k_cols: tl.constexpr,
    PAD_Q: tl.constexpr,
    PAD_K: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    input_row_stride = q_cols + 2 * k_cols

    for idx in tl.range(pid, batch_size, num_programs):
        input_base = input_ptr + idx * input_row_stride

        q_in_base = input_base
        q_out_base = q_out_ptr + idx * q_cols
        q_sum = tl.zeros((), dtype=tl.float32)
        q_comp = tl.zeros((), dtype=tl.float32)
        for q_off in tl.static_range(0, PAD_Q, Q_BLOCK_SIZE):
            q_offsets = q_off + tl.arange(0, Q_BLOCK_SIZE)
            q_mask = q_offsets < q_cols
            q_vals = tl.load(q_in_base + q_offsets, mask=q_mask, other=0.0)
            q_vals_f32 = q_vals.to(tl.float32)
            q_chunk = tl.sum(q_vals_f32 * q_vals_f32, axis=0)
            y = q_chunk - q_comp
            t = q_sum + y
            q_comp = (t - q_sum) - y
            q_sum = t
            tl.store(q_out_base + q_offsets, q_vals, mask=q_mask)
        q_var = q_sum / q_cols

        k_in_base = input_base + q_cols
        k_out_base = k_out_ptr + idx * k_cols
        k_sum = tl.zeros((), dtype=tl.float32)
        k_comp = tl.zeros((), dtype=tl.float32)
        for k_off in tl.static_range(0, PAD_K, KV_BLOCK_SIZE):
            k_offsets = k_off + tl.arange(0, KV_BLOCK_SIZE)
            k_mask = k_offsets < k_cols
            k_vals = tl.load(k_in_base + k_offsets, mask=k_mask, other=0.0)
            k_vals_f32 = k_vals.to(tl.float32)
            k_chunk = tl.sum(k_vals_f32 * k_vals_f32, axis=0)
            y = k_chunk - k_comp
            t = k_sum + y
            k_comp = (t - k_sum) - y
            k_sum = t
            tl.store(k_out_base + k_offsets, k_vals, mask=k_mask)
        k_var = k_sum / k_cols

        v_in_base = input_base + q_cols + k_cols
        v_out_base = v_out_ptr + idx * k_cols
        for v_off in tl.static_range(0, PAD_K, KV_BLOCK_SIZE):
            v_offsets = v_off + tl.arange(0, KV_BLOCK_SIZE)
            v_mask = v_offsets < k_cols
            v_vals = tl.load(v_in_base + v_offsets, mask=v_mask, other=0.0)
            tl.store(v_out_base + v_offsets, v_vals, mask=v_mask)

        tl.store(qk_var_ptr + idx * 2, q_var)
        tl.store(qk_var_ptr + idx * 2 + 1, k_var)


@triton.jit
def _apply_global_rmsnorm_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    rotary_dim: tl.constexpr,
    q_weight_ptr,
    k_weight_ptr,
    qk_global_var_ptr,
    eps: tl.constexpr,
    inv_tp_world: tl.constexpr,
    batch_size,
    q_cols: tl.constexpr,
    k_cols: tl.constexpr,
    q_num_heads,
    k_num_heads,
    head_dim: tl.constexpr,
    PAD_Q: tl.constexpr,
    PAD_K: tl.constexpr,
    PAD_QH: tl.constexpr,
    PAD_KH: tl.constexpr,
    PAD_HALF: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)

    for idx in tl.range(pid, batch_size, num_programs):
        q_gv = tl.load(qk_global_var_ptr + idx * 2).to(tl.float32) * inv_tp_world
        k_gv = tl.load(qk_global_var_ptr + idx * 2 + 1).to(tl.float32) * inv_tp_world
        q_scale = 1.0 / tl.sqrt(q_gv + eps)
        k_scale = 1.0 / tl.sqrt(k_gv + eps)

        q_base = q_ptr + idx * q_cols
        for q_off in tl.static_range(0, PAD_Q, Q_BLOCK_SIZE):
            q_offsets = q_off + tl.arange(0, Q_BLOCK_SIZE)
            q_mask = q_offsets < q_cols
            q_vals = tl.load(q_base + q_offsets, mask=q_mask, other=0.0)
            q_weight = tl.load(q_weight_ptr + q_offsets, mask=q_mask, other=1.0).to(
                tl.float32
            )
            q_vals = (q_vals.to(tl.float32) * q_scale * q_weight).to(q_vals.dtype)
            tl.store(q_base + q_offsets, q_vals, mask=q_mask)

        k_base = k_ptr + idx * k_cols
        for k_off in tl.static_range(0, PAD_K, KV_BLOCK_SIZE):
            k_offsets = k_off + tl.arange(0, KV_BLOCK_SIZE)
            k_mask = k_offsets < k_cols
            k_vals = tl.load(k_base + k_offsets, mask=k_mask, other=0.0)
            k_weight = tl.load(k_weight_ptr + k_offsets, mask=k_mask, other=1.0).to(
                tl.float32
            )
            k_vals = (k_vals.to(tl.float32) * k_scale * k_weight).to(k_vals.dtype)
            tl.store(k_base + k_offsets, k_vals, mask=k_mask)

        half = rotary_dim // 2
        half_offsets = tl.arange(0, PAD_HALF)
        half_mask = half_offsets < half
        cos_row = tl.load(
            cos_ptr + idx * rotary_dim + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        sin_row = tl.load(
            sin_ptr + idx * rotary_dim + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)

        qh_offsets = tl.arange(0, PAD_QH)[:, None] * head_dim + half_offsets[None, :]
        qh_mask = (tl.arange(0, PAD_QH)[:, None] < q_num_heads) & half_mask[None, :]
        qh_offsets_2 = qh_offsets + half
        q1_raw = tl.load(q_base + qh_offsets, mask=qh_mask, other=0.0)
        q2_raw = tl.load(q_base + qh_offsets_2, mask=qh_mask, other=0.0)
        q1 = q1_raw.to(tl.float32)
        q2 = q2_raw.to(tl.float32)
        qn1 = q1 * cos_row[None, :] - q2 * sin_row[None, :]
        qn2 = q2 * cos_row[None, :] + q1 * sin_row[None, :]
        tl.store(q_base + qh_offsets, qn1.to(q1_raw.dtype), mask=qh_mask)
        tl.store(q_base + qh_offsets_2, qn2.to(q2_raw.dtype), mask=qh_mask)

        kh_offsets = tl.arange(0, PAD_KH)[:, None] * head_dim + half_offsets[None, :]
        kh_mask = (tl.arange(0, PAD_KH)[:, None] < k_num_heads) & half_mask[None, :]
        kh_offsets_2 = kh_offsets + half
        k1_raw = tl.load(k_base + kh_offsets, mask=kh_mask, other=0.0)
        k2_raw = tl.load(k_base + kh_offsets_2, mask=kh_mask, other=0.0)
        k1 = k1_raw.to(tl.float32)
        k2 = k2_raw.to(tl.float32)
        kn1 = k1 * cos_row[None, :] - k2 * sin_row[None, :]
        kn2 = k2 * cos_row[None, :] + k1 * sin_row[None, :]
        tl.store(k_base + kh_offsets, kn1.to(k1_raw.dtype), mask=kh_mask)
        tl.store(k_base + kh_offsets_2, kn2.to(k2_raw.dtype), mask=kh_mask)


def split_qkv_tp_rmsnorm_rope(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_dim: int,
    tp_world: int,
    tp_group: torch.distributed.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused kernel for splitting QKV, applying Tensor Parallel RMSNorm, and RoPE.

    This kernel performs the following operations in a fused manner:
    1. Split the concatenated QKV input into separate Q, K, V tensors
    2. Compute local variance for Q and K (for distributed RMSNorm)
    3. All-reduce the variance across TP ranks
    4. Apply global RMSNorm to Q and K using the synchronized variance
    5. Apply Neox-style Rotary Position Embedding (RoPE) to Q and K

    The input tensor layout is: [Q_hidden_size, K_hidden_size, V_hidden_size]
    where K and V share the same hidden size (kv_hidden_size).

    Args:
        input: Input tensor of shape (batch_size, q_hidden_size + 2 * kv_hidden_size)
            containing concatenated Q, K, V projections.
        cos: Cosine values for RoPE, shape (batch_size, rotary_dim).
        sin: Sine values for RoPE, shape (batch_size, rotary_dim).
        q_hidden_size: Hidden size for Q projection.
        kv_hidden_size: Hidden size for K and V projections.
        head_dim: Dimension of each attention head. Must be a power of 2.
        eps: Epsilon value for RMSNorm numerical stability.
        q_weight: Weight tensor for Q RMSNorm, shape (q_hidden_size,).
        k_weight: Weight tensor for K RMSNorm, shape (kv_hidden_size,).
        rotary_dim: Dimension for rotary position embedding. Usually equals head_dim.
        tp_world: Total number of tensor parallel ranks.
        tp_group: Process group for tensor parallel all-reduce communication.

    Returns:
        A tuple of (q, k, v) tensors:
            - q: Query tensor of shape (batch_size, q_hidden_size)
            - k: Key tensor of shape (batch_size, kv_hidden_size)
            - v: Value tensor of shape (batch_size, kv_hidden_size)

    Note:
        - This kernel is designed for Tensor Parallelism where each rank processes
          a portion of the hidden dimension.
        - The variance computation uses Kahan summation for numerical stability.
        - V is not normalized or rotated, only split from the input.
    """
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert KV_BLOCK_SIZE == head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    q = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    if batch_size == 0:
        return q, k, v
    _, num_vectorcore = get_device_properties()
    grid = (min(batch_size, num_vectorcore),)
    q_cols = q_hidden_size
    k_cols = kv_hidden_size
    q_num_heads = q_hidden_size // head_dim
    k_num_heads = kv_hidden_size // head_dim

    qk_var = torch.empty(batch_size, 2, dtype=torch.float32, device=q.device)
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input,
        q,
        k,
        v,
        qk_var,
        batch_size,
        q_cols,
        k_cols,
        triton.next_power_of_2(q_cols),
        triton.next_power_of_2(k_cols),
        KV_BLOCK_SIZE,
        Q_BLOCK_SIZE,
    )
    if tp_world > 1:
        dist.all_reduce(qk_var, group=tp_group)

    _apply_global_rmsnorm_kernel[grid](
        q,
        k,
        cos,
        sin,
        rotary_dim,
        q_weight,
        k_weight,
        qk_var,
        eps,
        1.0 / tp_world,
        batch_size,
        q_cols,
        k_cols,
        q_num_heads,
        k_num_heads,
        head_dim,
        triton.next_power_of_2(q_cols),
        triton.next_power_of_2(k_cols),
        triton.next_power_of_2(q_num_heads),
        triton.next_power_of_2(k_num_heads),
        triton.next_power_of_2(rotary_dim // 2),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )

    return q, k, v

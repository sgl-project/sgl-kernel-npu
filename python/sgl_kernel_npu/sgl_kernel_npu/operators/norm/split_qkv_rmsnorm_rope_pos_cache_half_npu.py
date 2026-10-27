"""Public split_qkv_rmsnorm_rope_pos_cache_half_npu with implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", "auto")
@register_impl("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", "triton")
def _triton_impl(
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
    wide_grid_min_tokens: int = 1024,  # batch at/above which the wide grid is used
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
        positions: Position index per batch item, shape [B], dtype int32 or int64. Values are clamped to
            [0, max_seq) inside the kernel (no host sync); keep inputs in range for correct RoPE.
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
        wide_grid_min_tokens: Batch size at/above which the single-column ("wide") launch grid is used
            instead of the per-head grid. Below it the per-head grid is used with unchanged launch
            parameters. See the grid selection comment below for the measured crossover.

    Returns:
        Tuple of (q_out, k_out, v_out), shapes [B, q_hidden_size], [B, kv_hidden_size], [B, kv_hidden_size].
    """
    import triton

    from ...implementation.triton.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
        _WIDE_GRID_MAX_Q_HIDDEN,
        split_qkv_rmsnorm_rope_half_pos_cache_kernel,
    )
    from ...utils.triton_utils import get_device_properties

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
    assert max_seq >= 1, "cos_sin_cache must be non-empty"
    # Do not use (pos < 0).any() / pos.min().item() here: they synchronize the device and break
    # torch.npu.graph capture ("Not allow to synchronize captured-stream"). Out-of-range indices
    # are clamped inside the Triton kernel (see max_seq).
    stride0 = cache.stride(0)

    assert (
        triton.next_power_of_2(head_dim) == head_dim
    ), "this kernel assumes head_dim is power-of-2"
    assert q_hidden_size % kv_hidden_size == 0

    # Launch grid. Each program walks its rows serially (batch // n_rows iterations), so the
    # per-row scalar/address overhead is paid once per column program. Collapsing to a single
    # column program (full Q/KV width per row, n_rows = num_vectorcore) cuts that serial
    # iteration count by n_cols and is up to 2.7x faster at large batch. Device-time measured
    # on 910B3 (40 vector cores, q_hidden=2048/kv_hidden=512/head_dim=128, bf16): the wide grid
    # stays near a ~106 us fixed-cost floor while the per-head grid starts climbing above it at
    # ~1150 tokens (1.13x), reaching 1.9x at 2048 and 2.7x at 4096. Under ~1024 tokens both sit
    # on that floor, so the default errs low: being early costs nothing measurable, being late
    # forfeits up to 1.9x.
    #
    # The wide grid materializes a q_hidden_size-wide tile per row instead of a
    # (q_hidden_size // kv_hidden_size) * head_dim one, so it is only taken while that tile
    # still fits the Unified Buffer. Measured on 910B3: q_hidden_size up to 8192 compiles,
    # 12288 fails with MLIRCompilationError. Wider models keep the per-head grid, which is
    # what they use today, rather than failing to compile once the batch grows.
    if B >= wide_grid_min_tokens and q_hidden_size <= _WIDE_GRID_MAX_Q_HIDDEN:
        q_block_size = q_hidden_size
        kv_block_size = kv_hidden_size
        n_cols = 1
        n_rows = num_vectorcore
    else:
        kv_block_size = head_dim
        q_block_size = (q_hidden_size // kv_hidden_size) * head_dim
        n_cols = kv_hidden_size // kv_block_size
        n_rows = (num_vectorcore + n_cols - 1) // n_cols

    q_block_n = q_block_size // head_dim
    k_block_n = kv_block_size // head_dim  # 1 on the per-head grid

    q_out = torch.empty(
        (B, q_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )
    k_out = torch.empty(
        (B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )
    v_out = torch.empty(
        (B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )

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
        max_seq=max_seq,
    )

    return q_out, k_out, v_out


@register_impl("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", "torch_native")
def _torch_native_impl(
    input_tensor: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float = None,
    q_weight: torch.Tensor = None,
    k_weight: torch.Tensor = None,
    q_bias: torch.Tensor = None,
    k_bias: torch.Tensor = None,
    rope_dim: int = None,
    cast_norm_to_bf16: bool = True,
    wide_grid_min_tokens: int = 1024,
):
    from ...implementation.torch_native.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
        split_qkv_rmsnorm_rope_pos_cache_half_npu,
    )
    from ._qkv_inputs import prepare_position_cache, reference_weights, split_qkv_inputs

    pos, cache, rope_dim = prepare_position_cache(
        input_tensor,
        positions,
        cos_sin_cache,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
    )
    # Native loads int64 positions as int32, then clamps on device.
    pos = pos.reshape(-1).to(torch.int32).long().clamp(0, cache.shape[0] - 1)
    selected = cache.index_select(0, pos)[:, :rope_dim]
    cos, sin = selected.chunk(2, dim=-1)
    sin = sin.reshape(input_tensor.shape[0], 1, rope_dim // 2)
    cos = cos.reshape(input_tensor.shape[0], 1, rope_dim // 2)
    q, k, v = split_qkv_inputs(input_tensor, q_hidden_size, kv_hidden_size, head_dim)
    weights = reference_weights(head_dim, eps, q_weight, k_weight, q_bias, k_bias)
    return split_qkv_rmsnorm_rope_pos_cache_half_npu(
        q, k, v, sin, cos, eps, *weights, cast_norm_to_bf16, input_tensor.dtype
    )


@register_operator("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu")
def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float = None,
    q_weight: torch.Tensor = None,
    k_weight: torch.Tensor = None,
    q_bias: torch.Tensor = None,
    k_bias: torch.Tensor = None,
    rope_dim: int = None,
    cast_norm_to_bf16: bool = True,
    wide_grid_min_tokens: int = 1024,
    *,
    impl="auto",
):
    """Use the original Triton implementation by default, or the PyTorch reference."""
    return get_impl_by_name("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", impl)(
        input_tensor,
        positions,
        cos_sin_cache,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        rope_dim,
        cast_norm_to_bf16,
        wide_grid_min_tokens,
    )

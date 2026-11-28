"""Public Ascend 950 (arch35) 3-stage fused GDN prefill operator."""

from ...utils.generation import Generation
from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


def build_gdn_prefill_chunk_meta(seq_lens_cpu, device, chunk_size=64):
    """Build (chunk_indices [NT, 2], chunk_offsets [B+1], NT, total_valid_tokens)."""
    from ....implementation.triton.attention.gdn.fused_gdn_prefill import (
        build_gdn_prefill_chunk_meta as _build_meta,
    )

    return _build_meta(seq_lens_cpu, device, chunk_size=chunk_size)


@register_impl(
    "attention.gdn.fused_gdn_prefill", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "attention.gdn.fused_gdn_prefill",
    "triton",
    supported_generations={Generation.ARCH35},
)
def _triton_impl(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_k_heads,
    num_v_heads,
    total_valid_tokens,
    ckpt_chunk_idx,
    ckpt_dst_slots,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_prefill import (
            fused_gdn_prefill,
        )

        _triton_fn = fused_gdn_prefill
    return _triton_fn(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_k_heads,
        num_v_heads,
        total_valid_tokens,
        ckpt_chunk_idx,
        ckpt_dst_slots,
    )


@register_impl("attention.gdn.fused_gdn_prefill", "torch_native")
def _torch_native_impl(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_k_heads,
    num_v_heads,
    total_valid_tokens,
    ckpt_chunk_idx,
    ckpt_dst_slots,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_prefill import (
            fused_gdn_prefill,
        )

        _torch_native_fn = fused_gdn_prefill
    return _torch_native_fn(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_k_heads,
        num_v_heads,
        total_valid_tokens,
        ckpt_chunk_idx,
        ckpt_dst_slots,
    )


@register_operator("attention.gdn.fused_gdn_prefill")
def fused_gdn_prefill(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_k_heads,
    num_v_heads,
    total_valid_tokens=None,
    ckpt_chunk_idx=None,
    ckpt_dst_slots=None,
    *,
    impl="auto",
):
    """Run Ascend 950 (arch35) 3-stage fused GDN prefill with in-place state & checkpointing.

    auto selects Triton on Generation.ARCH35. Fuses Q/K L2Norm + GDN gating (zero V copy),
    GQA-interleaved Prepare (K@K^T, Q@K^T, cumsum, exact 2-level 8x8 Euler solve_tril in FP16,
    WY W/U), and slim 4-dot ChunkFwdH + ChunkFwdO with in-place ssm_states update and
    optional single-point Radix Cache checkpointing.

    Args:
        mixed_qkv (torch.Tensor): Contiguous bfloat16 ``[T, (2*NK + NV)*128]``.
        a (torch.Tensor): Contiguous ``[T, NV]`` pre-softplus decay input.
        b (torch.Tensor): Contiguous ``[T, NV]`` pre-sigmoid beta input.
        A_log (torch.Tensor): ``[NV]`` float32 log-A parameter.
        dt_bias (torch.Tensor): ``[NV]`` decay bias parameter.
        ssm_states (torch.Tensor): Contiguous writable ``[num_slots, NV, 128, 128]`` pool.
        cache_indices (torch.Tensor): ``[B]`` int32 active slot indices per sequence.
        cu_seqlens (torch.Tensor): ``[B+1]`` int32 cumulative sequence lengths.
        chunk_indices (torch.Tensor): ``[NT, 2]`` int32 ``(seq_id, chunk_id)`` table.
        chunk_offsets (torch.Tensor): ``[B+1]`` int32 cumulative chunk counts.
        num_k_heads (int): Number of Q/K heads (power of 2, divides ``num_v_heads``).
        num_v_heads (int): Number of V heads.
        total_valid_tokens (int | None): Valid token count; trailing padded rows are zeroed.
        ckpt_chunk_idx (torch.Tensor | None): Optional ``[B]`` int32 chunk index to snapshot.
        ckpt_dst_slots (torch.Tensor | None): Optional ``[B]`` int32 destination slots (<0 skips).
        impl (str): ``"auto"``, ``"triton"``, or ``"torch_native"``. Defaults to ``"auto"``.

    Returns:
        torch.Tensor: Contiguous bfloat16 ``[T, NV, 128]`` prefill output.
    """
    return get_impl_by_name("attention.gdn.fused_gdn_prefill", impl)(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_k_heads,
        num_v_heads,
        total_valid_tokens,
        ckpt_chunk_idx,
        ckpt_dst_slots,
    )

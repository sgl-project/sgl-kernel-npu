from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.chunk_fwd_o", "auto")
@register_impl("attention.gdn.chunk_fwd_o", "triton")
def _triton_impl(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.chunk_o import chunk_fwd_o_npu

        _triton_fn = chunk_fwd_o_npu
    return _triton_fn(q, k, v, h, g, scale, cu_seqlens, chunk_size)


@register_impl("attention.gdn.chunk_fwd_o", "torch_native")
def _torch_native_impl(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.chunk_o import chunk_fwd_o

        _torch_native_fn = chunk_fwd_o
    return _torch_native_fn(q, k, v, h, g, scale, cu_seqlens, chunk_size)


@register_operator("attention.gdn.chunk_fwd_o")
def chunk_fwd_o(
    q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64, *, impl="auto"
):
    """Combine entering chunk states and causal local attention for GDN outputs.

    Args:
        q: Contiguous queries [B,T,Hg,K], FP16/BF16/FP32 on the NPU.
        k: Contiguous keys with q's shape/dtype/device.
        v: Contiguous transformed values [B,T,H,V]; H must be a multiple of Hg.
        h: Contiguous entering states [B,ceil(T/C),H,K,V], or [1,total_chunks,H,K,V]
            for packed sequences. Use q's dtype/device.
        g: Chunk-local cumulative log decay [B,T,H], usually FP32. The default
            None is retained in the signature, but both implementations require a tensor.
        scale: Output multiplier; defaults to K**-0.5.
        cu_seqlens: Optional int32/int64 boundaries [N+1] for packed B=1 input.
            Start at 0, end at T; each sequence restarts its chunk numbering.
        chunk_size: Chunk width C, default 64; Triton uses a power of two.
        impl: auto (default, selects Triton), triton or torch_native.

    Returns:
        Newly allocated tensor with v's shape/dtype/device. Inputs are unchanged.
    """
    return get_impl_by_name("attention.gdn.chunk_fwd_o", impl)(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

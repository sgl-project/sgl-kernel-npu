"""Public chunked scaled key-key product."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.chunk_scaled_dot_kkt", "auto")
@register_impl("attention.chunk_scaled_dot_kkt", "triton")
def _triton(*args, **kwargs):
    from ...implementation.triton.attention.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd_npu,
    )

    return chunk_scaled_dot_kkt_fwd_npu(*args, **kwargs)


@register_impl("attention.chunk_scaled_dot_kkt", "torch_native")
def _native(*args, **kwargs):
    from ...implementation.torch_native.attention.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt,
    )

    return chunk_scaled_dot_kkt(*args, **kwargs)


@register_operator("attention.chunk_scaled_dot_kkt")
def chunk_scaled_dot_kkt(
    k,
    beta,
    g_cumsum=None,
    cu_seqlens=None,
    chunk_size=64,
    output_dtype=torch.float32,
    *,
    impl="auto",
):
    """Return strict-lower chunked beta * exp(g_i-g_j) * (k_i @ k_j).

    k is [B,T,Hg,K], beta and g_cumsum are [B,T,H], and the output is
    [B,T,H,chunk_size]. The current Triton wrapper requires g_cumsum.
    """
    if g_cumsum is None:
        raise ValueError("g_cumsum is required by the current Triton wrapper")
    return get_impl_by_name("attention.chunk_scaled_dot_kkt", impl)(
        k, beta, g_cumsum, cu_seqlens, chunk_size, output_dtype
    )

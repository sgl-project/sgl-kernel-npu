"""Public scalar chunk-local cumulative sum."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.chunk_local_cumsum", "auto")
@register_impl("attention.chunk_local_cumsum", "triton")
def _triton_impl(*args, **kwargs):
    from ...implementation.triton.attention.chunk_local_cumsum import (
        chunk_local_cumsum_scalar_npu,
    )

    return chunk_local_cumsum_scalar_npu(*args, **kwargs)


@register_impl("attention.chunk_local_cumsum", "torch_native")
def _native_impl(*args, **kwargs):
    from ...implementation.torch_native.attention.chunk_local_cumsum import (
        chunk_local_cumsum,
    )

    return chunk_local_cumsum(*args, **kwargs)


@register_operator("attention.chunk_local_cumsum")
def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float32,
    *,
    impl="auto",
):
    """Sum each sequence chunk independently; auto selects Triton.

    g has shape [B,T,H] or [B,H,T] when head_first. cu_seqlens selects
    variable-length sequences in a flattened batch of size one.
    """
    if g.ndim != 3:
        raise ValueError("scalar cumulative sum requires a 3-D input")
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a positive power of two")
    if cu_seqlens is not None and g.shape[0] != 1:
        raise ValueError("variable-length input requires batch size 1")
    return get_impl_by_name("attention.chunk_local_cumsum", impl)(
        g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype
    )

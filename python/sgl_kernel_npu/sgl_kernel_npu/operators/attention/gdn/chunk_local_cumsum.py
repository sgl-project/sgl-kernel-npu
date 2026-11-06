"""Public scalar chunk-local cumulative sum."""

import torch

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.chunk_local_cumsum", "auto")
@register_impl("attention.gdn.chunk_local_cumsum", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.chunk_local_cumsum import (
            chunk_local_cumsum_scalar_npu,
        )

        _triton_fn = chunk_local_cumsum_scalar_npu
    return _triton_fn(*args, **kwargs)


@register_impl("attention.gdn.chunk_local_cumsum", "torch_native")
def _native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.chunk_local_cumsum import (
            chunk_local_cumsum,
        )

        _torch_native_fn = chunk_local_cumsum
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.gdn.chunk_local_cumsum")
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
    Triton accumulates in FP32 and casts to g.dtype before writing. Its output
    dtype must equal g.dtype: pass output_dtype=None or g.dtype for FP16/BF16
    inputs, since the default is torch.float32.
    Triton rejects head_first with cu_seqlens and head counts yielding BLOCK_T
    smaller than chunk_size. head_first requires T divisible by BLOCK_T.
    BLOCK_T is the next power of two of floor(2**18 / (H * chunk_size)).
    """
    if g.ndim != 3:
        raise ValueError("scalar cumulative sum requires a 3-D input")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
        or chunk_size & (chunk_size - 1)
    ):
        raise ValueError("chunk_size must be a positive power of two")
    if any(dim <= 0 for dim in g.shape):
        raise ValueError("chunk_local_cumsum requires nonempty dimensions")
    if impl in ("auto", "triton"):
        if head_first and cu_seqlens is not None:
            raise ValueError("Triton does not support head_first=True with cu_seqlens")
        H, T = (g.shape[1], g.shape[2]) if head_first else (g.shape[2], g.shape[1])
        tile_budget = (2**18) // (H * chunk_size)
        block_t = 1 << (max(tile_budget, 1) - 1).bit_length()
        if tile_budget == 0 or block_t < chunk_size:
            raise ValueError(
                "Triton head count and chunk_size require BLOCK_T >= chunk_size"
            )
        if head_first and T % block_t:
            raise ValueError(
                f"Triton head_first requires T divisible by BLOCK_T={block_t}"
            )
    if cu_seqlens is not None and g.shape[0] != 1:
        raise ValueError("variable-length input requires batch size 1")
    if (
        impl in ("auto", "triton")
        and output_dtype is not None
        and output_dtype != g.dtype
    ):
        raise ValueError(
            "Triton output_dtype must match g.dtype; use None or g.dtype "
            f"(got output_dtype={output_dtype}, g.dtype={g.dtype})"
        )
    return get_impl_by_name("attention.gdn.chunk_local_cumsum", impl)(
        g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype
    )

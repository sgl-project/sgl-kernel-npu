"""Public mrope operator."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "rope.mrope"
_torch_native_fn = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH22})
@register_impl(_OP_NAME, "ops-transformer", supported_generations={Generation.ARCH22})
def _torch_npu_impl(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode="half",
    cache_mode="default"
):
    kwargs = {"mrope_section": mrope_section, "rotary_mode": rotary_mode}
    # Older TorchNPU versions implement the default cache layout implicitly.
    if cache_mode != "default":
        kwargs["cache_mode"] = cache_mode
    return torch_npu.npu_mrope(
        positions, query, key, cos_sin_cache, head_size, **kwargs
    )


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode="half",
    cache_mode="default"
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.rope.mrope import mrope

        _torch_native_fn = mrope
    return _torch_native_fn(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        mrope_section=mrope_section,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )


@register_operator(_OP_NAME)
def mrope(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode="half",
    cache_mode="default",
    impl="auto"
):
    """Rotate packed query and key heads using a cosine/sine cache.

    positions is [T] for standard RoPE or [axes, T] for multimodal
    RoPE. query and key are [T, num_heads * head_size], and the cache
    is [max_seq_len, rotary_dim]. mrope_section configures multimodal
    frequency segments. rotary_mode controls pairing, and cache_mode
    selects the segment layout. torch_native covers the default cache
    layout. auto selects ops-transformer on arch22.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        mrope_section=mrope_section,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )

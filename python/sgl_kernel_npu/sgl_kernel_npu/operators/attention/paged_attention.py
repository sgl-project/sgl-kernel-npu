"""TorchNPU prefill and decode operators for paged KV caches."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_ATB_GENERATIONS = {Generation.ARCH22, Generation.ARCH35}
_native_flash_attention_qlens = None
_native_paged_attention = None
_native_paged_attention_mla = None


@register_impl(
    "attention.flash_attention_qlens", "auto", supported_generations=_ATB_GENERATIONS
)
@register_impl(
    "attention.flash_attention_qlens",
    "ascend-transformer-boost",
    supported_generations=_ATB_GENERATIONS,
)
def _flash_attention_qlens_impl(**kwargs):
    return torch_npu._npu_flash_attention_qlens(**kwargs)


@register_impl("attention.flash_attention_qlens", "torch_native")
def _torch_native_flash_attention_qlens(*args, **kwargs):
    global _native_flash_attention_qlens
    if _native_flash_attention_qlens is None:
        from ...implementation.torch_native.attention.paged_attention import (
            flash_attention_qlens,
        )

        _native_flash_attention_qlens = flash_attention_qlens
    return _native_flash_attention_qlens(*args, **kwargs)


@register_operator("attention.flash_attention_qlens")
def flash_attention_qlens(
    query,
    key_cache,
    value_cache,
    mask,
    block_table,
    seq_len,
    context_lens,
    scale_value,
    num_heads,
    num_kv_heads,
    out,
    *,
    impl="auto",
    **kwargs,
):
    """Run paged prefill using query lengths and write into the caller's out.

    Query is packed by token (for example [T, num_heads * head_dim]).
    The block table maps requests to KV pages. seq_len and context_lens give
    each request's query and context lengths. auto selects ascend-transformer-boost.
    """
    return get_impl_by_name("attention.flash_attention_qlens", impl)(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        mask=mask,
        block_table=block_table,
        seq_len=seq_len,
        context_lens=context_lens,
        scale_value=scale_value,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        out=out,
        **kwargs,
    )


@register_impl(
    "attention.paged_attention",
    "auto",
    supported_generations=_ATB_GENERATIONS,
)
@register_impl(
    "attention.paged_attention",
    "ascend-transformer-boost",
    supported_generations=_ATB_GENERATIONS,
)
def _paged_attention_impl(**kwargs):
    return torch_npu._npu_paged_attention(**kwargs)


@register_impl("attention.paged_attention", "torch_native")
def _torch_native_paged_attention(*args, **kwargs):
    global _native_paged_attention
    if _native_paged_attention is None:
        from ...implementation.torch_native.attention.paged_attention import (
            paged_attention,
        )

        _native_paged_attention = paged_attention
    return _native_paged_attention(*args, **kwargs)


@register_operator("attention.paged_attention")
def paged_attention(
    query,
    key_cache,
    value_cache,
    num_heads,
    num_kv_heads,
    scale_value,
    block_table,
    context_lens,
    out,
    *,
    impl="auto",
    **kwargs,
):
    """Decode against paged KV and write into the caller's out.

    Query is shaped [B, num_heads, head_dim] on SGLang's decode path.
    block_table gives the page IDs and context_lens the valid KV lengths.
    auto selects ascend-transformer-boost and passes through SDK options such as workspace.
    """
    return get_impl_by_name("attention.paged_attention", impl)(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale_value=scale_value,
        block_table=block_table,
        context_lens=context_lens,
        out=out,
        **kwargs,
    )


@register_impl(
    "attention.paged_attention_mla", "auto", supported_generations=_ATB_GENERATIONS
)
@register_impl(
    "attention.paged_attention_mla",
    "ascend-transformer-boost",
    supported_generations=_ATB_GENERATIONS,
)
def _paged_attention_mla_impl(**kwargs):
    return torch_npu._npu_paged_attention_mla(**kwargs)


@register_impl("attention.paged_attention_mla", "torch_native")
def _torch_native_paged_attention_mla(*args, **kwargs):
    global _native_paged_attention_mla
    if _native_paged_attention_mla is None:
        from ...implementation.torch_native.attention.paged_attention import (
            paged_attention_mla,
        )

        _native_paged_attention_mla = paged_attention_mla
    return _native_paged_attention_mla(*args, **kwargs)


@register_operator("attention.paged_attention_mla")
def paged_attention_mla(
    query,
    key_cache,
    num_kv_heads,
    num_heads,
    scale_value,
    block_table,
    context_lens,
    mla_vheadsize,
    out,
    *,
    impl="auto",
    **kwargs,
):
    """Decode MLA against a shared paged latent/RoPE cache.

    mla_vheadsize is the latent value width. out is the preallocated
    [B, num_heads, mla_vheadsize] output. auto selects ascend-transformer-boost.
    """
    return get_impl_by_name("attention.paged_attention_mla", impl)(
        query=query,
        key_cache=key_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale_value,
        block_table=block_table,
        context_lens=context_lens,
        mla_vheadsize=mla_vheadsize,
        out=out,
        **kwargs,
    )

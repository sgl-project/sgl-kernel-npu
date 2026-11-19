"""TorchNPU prefill and decode operators for paged KV caches."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_npu = None


def _get_torch_npu():
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


@register_impl("attention.flash_attention_qlens", "auto")
@register_impl("attention.flash_attention_qlens", "torch_npu")
def _flash_attention_qlens_impl(**kwargs):
    return _get_torch_npu()._npu_flash_attention_qlens(**kwargs)


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
    each request's query and context lengths. auto selects torch_npu.
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


@register_impl("attention.paged_attention", "auto")
@register_impl("attention.paged_attention", "torch_npu")
def _paged_attention_impl(**kwargs):
    return _get_torch_npu()._npu_paged_attention(**kwargs)


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
    auto selects torch_npu and passes through SDK options such as workspace.
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


@register_impl("attention.paged_attention_mla", "auto")
@register_impl("attention.paged_attention_mla", "torch_npu")
def _paged_attention_mla_impl(**kwargs):
    return _get_torch_npu()._npu_paged_attention_mla(**kwargs)


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
    [B, num_heads, mla_vheadsize] output. auto selects torch_npu.
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

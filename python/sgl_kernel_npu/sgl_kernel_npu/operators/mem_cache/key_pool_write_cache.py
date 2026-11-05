"""Public KeyPool cache commit, including speculative rollback projection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("mem_cache.key_pool_write_cache", "auto")
@register_impl("mem_cache.key_pool_write_cache", "triton")
def _triton_impl(*args, **kwargs):
    from ...implementation.triton.mem_cache.key_pool_write_cache import (
        key_pool_write_cache,
    )

    return key_pool_write_cache(*args, **kwargs)


@register_operator("mem_cache.key_pool_write_cache")
def key_pool_write_cache(
    pooled_key,
    state_cache,
    pooled_slot_mapping,
    tail_write_indices,
    tail_state_indices,
    tail_token_indices,
    index_cache,
    tail_k,
    tail_gate,
    *,
    hidden_states=None,
    wk=None,
    gate_weight=None,
    norm_weight=None,
    norm_bias=None,
    mode="normal",
    cmp_ratio=4,
    norm_eps=1e-6,
    impl="auto",
):
    """Update caches in place and return None; -1 mappings never read/write.

    Persistent tails are BF16; CANN state and MTP normalization remain FP32.
    MTP projects the retained hidden rows inside this entry point. All mapping
    buffers are explicit inputs, with no runtime Python objects or global scratch.
    """
    return get_impl_by_name("mem_cache.key_pool_write_cache", impl)(
        pooled_key,
        state_cache,
        pooled_slot_mapping,
        tail_write_indices,
        tail_state_indices,
        tail_token_indices,
        index_cache,
        tail_k,
        tail_gate,
        hidden_states=hidden_states,
        wk=wk,
        gate_weight=gate_weight,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        mode=mode,
        cmp_ratio=cmp_ratio,
        norm_eps=norm_eps,
    )

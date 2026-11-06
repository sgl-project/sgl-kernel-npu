"""Fused projection, normalization and pooling through the attentions wheel."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_aclnn_fn = None


@register_impl("mem_cache.key_pool", "auto")
@register_impl("mem_cache.key_pool", "aclnn")
def _aclnn_impl(*args, **kwargs):
    global _aclnn_fn
    if _aclnn_fn is None:
        import attentions  # noqa: F401 -- loads the native op registration

        _aclnn_fn = torch.ops.attentions.key_pool.default
    return _aclnn_fn(*args, **kwargs)


@register_operator("mem_cache.key_pool")
def key_pool(
    hidden_states,
    wk,
    gate_weight,
    ape,
    state_cache,
    cache_block_table,
    start_pos,
    *,
    norm_weight=None,
    norm_bias=None,
    cos=None,
    sin=None,
    cu_seqlens=None,
    seqused=None,
    cmp_ratio=4,
    norm_eps=1e-6,
    rotary_mode=1,
    impl="auto",
):
    """Return compact pooled keys and mutate FP32 ``state_cache`` in place.

    TH inputs require device int32 ``cu_seqlens`` including the leading zero.
    Only the first sum((start+Q)//r-start//r) output rows are initialized.
    RoPE and seqused are reserved and must remain None. Requires the separate
    attentions wheel and CANN/custom ops exporting aclnnKeyPool.
    """
    return get_impl_by_name("mem_cache.key_pool", impl)(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cos=cos,
        sin=sin,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        cmp_ratio=cmp_ratio,
        norm_eps=norm_eps,
        rotary_mode=rotary_mode,
    )

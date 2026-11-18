"""LongCat2 sparse index selection with explicit implementation selection."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "indexer.mlp_lightning_indexer", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "indexer.mlp_lightning_indexer",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(*args):
    return torch.ops.npu.mlp_lightning_indexer(*args)


@register_impl(
    "indexer.mlp_lightning_indexer",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.mlp_lightning_indexer import (
            mlp_lightning_indexer,
        )

        _torch_native_fn = mlp_lightning_indexer
    return _torch_native_fn(*args)


@register_operator("indexer.mlp_lightning_indexer")
def mlp_lightning_indexer(
    query,
    key,
    weights,
    cur_seq_lengths_query=None,
    cur_seq_lengths_key=None,
    block_table=None,
    init_tensor=None,
    local_tensor=None,
    layout_query="BSND",
    layout_key="PA_BSND",
    sparse_count=2048,
    kv_block_len=1,
    q_block_len=1,
    init_num=0,
    local_num=0,
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    return_value=False,
    *,
    impl="auto",
):
    """Select LongCat2 block indices; auto selects Ascend C on arch22.

    Parameters and defaults mirror the native schema. The LongCat2 path uses
    TND queries, PA_BSND keys, fp32 weights and int64 B+1 length prefixes.
    Logical zero-length queries occupy one physical token slot. Returns an
    (int32 indices, query-dtype values) tuple; values are undefined when
    return_value=False. The reference uses the kernel's B+1 prefix convention
    for both TND and BSND queries. No automatic fallback is performed.
    """
    impl_fn = get_impl_by_name("indexer.mlp_lightning_indexer", impl)
    return impl_fn(
        query,
        key,
        weights,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        block_table,
        init_tensor,
        local_tensor,
        layout_query,
        layout_key,
        sparse_count,
        kv_block_len,
        q_block_len,
        init_num,
        local_num,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
    )

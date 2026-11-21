"""Generic block sparse attention and its scheduling metadata for A3."""

from functools import lru_cache

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@lru_cache(maxsize=1)
def _bindings():
    version = detect_cann_version()
    if version == "9.2.0":
        from ..._third_party.sgl_ops_transformer import ops
    elif version == "9.1.0":
        from ..._third_party.sgl_ops_transformer_gbsa import ops
    else:
        raise RuntimeError(f"No A3 GBSA profile for CANN {version}")
    return ops


@register_impl(
    "attention.generic_block_sparse_attention",
    "auto",
    supported_generations={Generation.ARCH22},
)
def _gbsa(*args, **kwargs):
    return _bindings().generic_block_sparse_attention(*args, **kwargs)


@register_impl(
    "attention.generic_block_sparse_attention_metadata",
    "auto",
    supported_generations={Generation.ARCH22},
)
def _gbsa_metadata(*args, **kwargs):
    return _bindings().generic_block_sparse_attention_metadata(*args, **kwargs)


@register_operator("attention.generic_block_sparse_attention")
def generic_block_sparse_attention(
    q, k, v, sparse_block_idx, sparse_block_count, block_shape, *, impl="auto", **kwargs
):
    """Attend to selected KV blocks using the bundled ops-transformer operator."""
    return get_impl_by_name("attention.generic_block_sparse_attention", impl)(
        q, k, v, sparse_block_idx, sparse_block_count, block_shape, **kwargs
    )


@register_operator("attention.generic_block_sparse_attention_metadata")
def generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_shape,
    *,
    impl="auto",
    **kwargs,
):
    """Build scheduling metadata for the corresponding GBSA call."""
    return get_impl_by_name("attention.generic_block_sparse_attention_metadata", impl)(
        sparse_block_idx,
        sparse_block_count,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_shape,
        **kwargs,
    )

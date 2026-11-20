"""TorchNPU Lightning Indexer adapter using the original SDK schema."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "indexer.lightning_indexer"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-transformer")
def _torch_npu_impl(query, key, weights, **kwargs):
    return torch_npu.npu_lightning_indexer(query, key, weights, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.indexer.lightning_indexer import (
            lightning_indexer,
        )

        _native_impl = lightning_indexer
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def lightning_indexer(query, key, weights, *, impl="auto", **kwargs):
    """Select top-k index positions using TorchNPU's weighted Q/K scores.

    Query is [T, H, D] for TND or [B, S, H, D] for BSND. weights has one
    scalar per query head per token, and key may be a paged tensor. Pass
    SDK keywords such as sparse_count, block_table, actual sequence lengths
    and layouts directly. Returns the TorchNPU indices and optional scores.
    auto selects ops-transformer.
    """
    return get_impl_by_name(_OP_NAME, impl)(query, key, weights, **kwargs)

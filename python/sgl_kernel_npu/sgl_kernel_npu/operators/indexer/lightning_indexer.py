"""TorchNPU Lightning Indexer adapter using the original SDK schema."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_npu_fn = None
_OP_NAME = "indexer.lightning_indexer"


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(query, key, weights, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_lightning_indexer

        _torch_npu_fn = npu_lightning_indexer
    return _torch_npu_fn(query, key, weights, **kwargs)


@register_operator(_OP_NAME)
def lightning_indexer(query, key, weights, *, impl="auto", **kwargs):
    """Select top-k index positions using TorchNPU's weighted Q/K scores.

    Query is [T, H, D] for TND or [B, S, H, D] for BSND. weights has one
    scalar per query head per token, and key may be a paged tensor. Pass
    SDK keywords such as sparse_count, block_table, actual sequence lengths
    and layouts directly. Returns the TorchNPU indices and optional scores.
    auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(query, key, weights, **kwargs)

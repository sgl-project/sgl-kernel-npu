"""Public quant_lightning_indexer operator with registered implementations."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError, indexer_family
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._indexer_compat import legacy_indexer_kwargs


@register_impl("attention.quant_lightning_indexer", "auto")
@register_impl("attention.quant_lightning_indexer", "cann_ops_transformer")
def _cann_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    family = indexer_family()
    if not family.legacy:
        return family.ops[1](
            q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
        )
    if quant_mode not in (1, 2):
        raise OperatorUnavailableError(
            "Legacy indexer supports the migrated FP8/INT8 modes only"
        )
    return family.ops[1](
        query=q,
        key=k,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale,
        sparse_count=topk,
        **legacy_indexer_kwargs(kwargs),
    )


@register_impl("attention.quant_lightning_indexer", "torch_native")
def _torch_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.attention.quant_lightning_indexer import (
        quant_lightning_indexer,
    )

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_operator("attention.quant_lightning_indexer")
def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl="auto", **kwargs
):
    """Return logical compressed-KV top-k indices and optional values.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    indices, values = get_impl_by_name("attention.quant_lightning_indexer", impl)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
    return indices.to(torch.int32), (
        values
        if kwargs.get("return_value", 0)
        else torch.empty(0, dtype=torch.float32, device=q.device)
    )

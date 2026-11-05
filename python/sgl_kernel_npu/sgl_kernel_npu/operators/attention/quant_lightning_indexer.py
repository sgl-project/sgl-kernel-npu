"""Public quant_lightning_indexer operator with registered implementations."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_op,
)
from .attention_utils import _cann_9_1_indexer_kwargs


@register_impl("attention.quant_lightning_indexer", "auto")
def _auto_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    backend = cann_backend()
    if backend == "cann_9_1":
        backend = "ops_transformer_cann910"
    return get_impl_by_name("attention.quant_lightning_indexer", backend)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_impl("attention.quant_lightning_indexer", "cann_ops_transformer_cann920")
def _cann_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    op = require_op(
        "cann_ops_transformer",
        "quant_lightning_indexer",
        backend="cann_ops_transformer_cann920",
    )
    return op(q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs)


@register_impl("attention.quant_lightning_indexer", "ops_transformer_cann910")
def _cann_9_1_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    op = require_op("custom", "npu_quant_lightning_indexer", backend="cann_9_1")
    if quant_mode not in (1, 2):
        raise OperatorUnavailableError(
            "cann_9_1 indexer supports the migrated FP8/INT8 modes only"
        )
    return op(
        query=q,
        key=k,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale,
        sparse_count=topk,
        **_cann_9_1_indexer_kwargs(kwargs),
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

    impl: auto (SDK-selected), ops_transformer_cann910, cann_ops_transformer_cann920, or torch_native.
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

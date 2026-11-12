"""Public quant_lightning_indexer_metadata operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from .indexer_utils import _cann_9_1_indexer_kwargs


def get_quant_lightning_indexer_metadata_size():
    """Return this operator's int32 metadata length without launching a kernel."""
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        return 1024
    if cann_version.startswith("9.2."):
        return 1024
    raise RuntimeError(
        f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
    )


@register_impl("indexer.quant_lightning_indexer_metadata", "auto")
def _auto_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "ops_transformers_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata", backend)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_impl(
    "indexer.quant_lightning_indexer_metadata", "cann_ops_transformers_cann920"
)
def _cann_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    return ops.quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_impl("indexer.quant_lightning_indexer_metadata", "ops_transformers_cann910")
def _cann_9_1_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(
        torch.ops.sgl_cann_recipes_infer, "npu_quant_lightning_indexer_metadata", None
    )
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.npu_quant_lightning_indexer_metadata in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    if quant_mode not in (1, 2):
        raise RuntimeError(
            "cann_9_1 indexer metadata supports the migrated FP8/INT8 modes only"
        )
    args = _cann_9_1_indexer_kwargs(kwargs)
    args.update(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        sparse_count=topk,
        device=str(args["actual_seq_lengths_query"].device),
    )
    return op(**args)


@register_impl("indexer.quant_lightning_indexer_metadata", "torch_native")
def _torch_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.indexer.quant_lightning_indexer_metadata import (
        quant_lightning_indexer_metadata,
    )

    return quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_operator("indexer.quant_lightning_indexer_metadata")
def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, *, impl="auto", **kwargs
):
    """Build quantized C4 indexer metadata.

    impl: auto (SDK-selected), ops_transformers_cann910, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata", impl)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )

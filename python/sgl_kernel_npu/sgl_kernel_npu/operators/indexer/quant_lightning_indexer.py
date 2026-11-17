"""Public quant_lightning_indexer operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from .indexer_utils import _cann_9_1_indexer_kwargs


@register_impl("indexer.quant_lightning_indexer", "auto")
def _auto_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
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
    return get_impl_by_name("indexer.quant_lightning_indexer", backend)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_impl("indexer.quant_lightning_indexer", "cann_ops_transformers_cann920")
def _cann_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    if "candidate_mode" in kwargs:
        from .indexer_utils import validate_quant_indexer_candidates

        validate_quant_indexer_candidates(
            q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
        )
        if torch_npu_soc_family() != 250:
            raise RuntimeError("INT8 candidate indexing requires A3 + CANN 9.2")
    result = ops.quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
    return result if "candidate_mode" in kwargs else result[:2]


@register_impl("indexer.quant_lightning_indexer", "ops_transformers_cann910")
def _cann_9_1_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(torch.ops.sgl_cann_recipes_infer, "npu_quant_lightning_indexer", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.npu_quant_lightning_indexer in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    if "candidate_mode" in kwargs:
        raise RuntimeError("INT8 candidate indexing requires A3 + CANN 9.2")
    if quant_mode not in (1, 2):
        raise RuntimeError("cann_9_1 indexer supports the migrated FP8/INT8 modes only")
    return op(
        query=q,
        key=k,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale,
        sparse_count=topk,
        **_cann_9_1_indexer_kwargs(kwargs),
    )


@register_impl("indexer.quant_lightning_indexer", "torch_native")
def _torch_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.indexer.quant_lightning_indexer import (
        quant_lightning_indexer,
    )

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_operator("indexer.quant_lightning_indexer")
def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl="auto", **kwargs
):
    """Return logical compressed-KV top-k indices and optional values.

    With candidate_mode omitted, return (indices, values), as before.
    A3 + CANN 9.2 also accepts candidate_mode=1 (source), 2 (consumer), or
    3 (ordinary), returning (indices, values, candidate_blocks). Consumer
    passes INT32 candidate_topk_index; only source produces candidate_blocks.
    Candidate counts are multiples of 64 in [64, 2048], with block size 8.

    impl: auto (SDK-selected), ops_transformers_cann910, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    result = get_impl_by_name("indexer.quant_lightning_indexer", impl)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
    indices, values = result[:2]
    if "candidate_mode" in kwargs:
        return indices.to(torch.int32), values, result[2]
    return indices.to(torch.int32), (
        values
        if kwargs.get("return_value", 0)
        else torch.empty(0, dtype=torch.float32, device=q.device)
    )


def torch_npu_soc_family():
    import torch_npu

    return torch_npu._C._npu_get_soc_version() // 10 * 10

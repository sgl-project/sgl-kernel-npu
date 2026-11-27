"""Public pooled-key index selection backed by CANN ops-transformer."""

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_bundled_cann_fn = None
_system_cann_fn = None
_torch_native_fn = None


@register_impl(
    "indexer.pool_key_indexer", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "indexer.pool_key_indexer",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH22},
)
def _bundled_cann_impl(*args, **kwargs):
    global _bundled_cann_fn
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("pool_key_indexer requires CANN 9.2.0")
    if _bundled_cann_fn is None:
        from ..._third_party.sgl_ops_transformer.ops import (
            pool_key_indexer as cann_pool_key_indexer,
        )

        _bundled_cann_fn = cann_pool_key_indexer

    return _bundled_cann_fn(*args, **kwargs)


@register_impl(
    "indexer.pool_key_indexer", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "indexer.pool_key_indexer",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _system_cann_impl(*args, **kwargs):
    global _system_cann_fn
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("pool_key_indexer requires CANN 9.2.0")
    if _system_cann_fn is None:
        from cann_ops_transformer.ops import pool_key_indexer as cann_pool_key_indexer

        _system_cann_fn = cann_pool_key_indexer

    return _system_cann_fn(*args, **kwargs)


@register_impl("indexer.pool_key_indexer", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.pool_key_indexer import (
            pool_key_indexer,
        )

        _torch_native_fn = pool_key_indexer
    return _torch_native_fn(*args, **kwargs)


@register_operator("indexer.pool_key_indexer")
def pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    topk=2048,
    pool_size=16,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
    q_descale=None,
    k_descale=None,
    *,
    impl="auto",
):
    """Select pooled top-k token indices and append the incomplete tail.

    ``auto`` and ``cann_ops_transformer`` load the bundled ops-transformer binding
    on A3 and the system CANN binding on A5. ``torch_native`` is an independent reference for
    BSND/BSND and TND/{TND, PA_BBND} layouts.
    ``pool_tail_k`` requires an int64 Tensor on CPU or NPU; ``actual_seq_*``
    accept the same Tensor inputs or None when optional for the layout.
    ``actual_seq_k`` contains cumulative ends for TND or per-request pool
    counts for PA_BBND. Both ``q_descale`` and ``k_descale`` are required for
    FP8 (``quant_mode=0``, FP32 scales) and INT8 (``quant_mode=2``, FP16
    scales). CANN validates their dtypes and shapes.
    Arguments and the (indices, values) result are forwarded unchanged; CANN
    validates its supported modes, dtypes and layouts. For graph replay, use metadata
    on the query NPU and update its buffers in place. All-CPU metadata is captured
    as constants and requires recapture when its values change.
    CANN derives the key stride from ``pool_key``. Indices are int32 with width
    topk + pool_size - 1. Values are float32 with width topk // pool_size when
    return_value=True, or an empty Tensor otherwise. Inputs remain read-only.
    """
    impl_fn = get_impl_by_name("indexer.pool_key_indexer", impl)
    return impl_fn(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        topk=topk,
        pool_size=pool_size,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
        q_descale=q_descale,
        k_descale=k_descale,
    )

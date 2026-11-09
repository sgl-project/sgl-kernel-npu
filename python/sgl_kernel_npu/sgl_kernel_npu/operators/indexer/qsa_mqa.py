"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_qsa_mqa_packed_triton_fn = None
_qsa_mqa_packed_torch_fn = None


@register_impl(
    "indexer.qsa_mqa_packed", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "indexer.qsa_mqa_packed", "triton", supported_generations={Generation.ARCH22}
)
def _qsa_mqa_packed_triton(q, k, starts, ends):
    global _qsa_mqa_packed_triton_fn
    if _qsa_mqa_packed_triton_fn is None:
        from ...implementation.triton.indexer.qsa_mqa import packed

        _qsa_mqa_packed_triton_fn = packed
    return _qsa_mqa_packed_triton_fn(q, k, starts, ends)


@register_impl(
    "indexer.qsa_mqa_packed", "torch_native", supported_generations={Generation.ARCH22}
)
def _qsa_mqa_packed_torch(q, k, starts, ends):
    global _qsa_mqa_packed_torch_fn
    if _qsa_mqa_packed_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_mqa import packed

        _qsa_mqa_packed_torch_fn = packed
    return _qsa_mqa_packed_torch_fn(q, k, starts, ends)


@register_operator("indexer.qsa_mqa_packed")
def qsa_mqa_packed(q, k, starts, ends, *, impl="auto"):
    """Packed QSA scores: BF16 Q[R,4,128], K[M,1,128], int32 starts/ends[R]. Returns FP32[R,M], sum(ReLU(QK))/sqrt(128), with -inf outside each interval. auto selects Triton.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("indexer.qsa_mqa_packed", impl)(q, k, starts, ends)


_qsa_mqa_paged_triton_fn = None
_qsa_mqa_paged_torch_fn = None


@register_impl(
    "indexer.qsa_mqa_paged", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "indexer.qsa_mqa_paged", "triton", supported_generations={Generation.ARCH22}
)
def _qsa_mqa_paged_triton(q, cache, table, lengths, width):
    global _qsa_mqa_paged_triton_fn
    if _qsa_mqa_paged_triton_fn is None:
        from ...implementation.triton.indexer.qsa_mqa import paged

        _qsa_mqa_paged_triton_fn = paged
    return _qsa_mqa_paged_triton_fn(q, cache, table, lengths, width)


@register_impl(
    "indexer.qsa_mqa_paged", "torch_native", supported_generations={Generation.ARCH22}
)
def _qsa_mqa_paged_torch(q, cache, table, lengths, width):
    global _qsa_mqa_paged_torch_fn
    if _qsa_mqa_paged_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_mqa import paged

        _qsa_mqa_paged_torch_fn = paged
    return _qsa_mqa_paged_torch_fn(q, cache, table, lengths, width)


@register_operator("indexer.qsa_mqa_paged")
def qsa_mqa_paged(q, cache, table, lengths, width, *, impl="auto"):
    """Paged QSA scores: BF16 Q[R,4,128], cache[P,16,1,128], int32 table[R,L] and lengths[R]; width=16*L. Returns FP32[R,width] with -inf padding. auto selects Triton.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("indexer.qsa_mqa_paged", impl)(
        q, cache, table, lengths, width
    )

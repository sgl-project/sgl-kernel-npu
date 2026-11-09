"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_qsa_topk_triton_fn = None
_qsa_topk_torch_fn = None


@register_impl("indexer.qsa_topk", "auto", supported_generations={Generation.ARCH22})
@register_impl("indexer.qsa_topk", "triton", supported_generations={Generation.ARCH22})
def _qsa_topk_triton(score, lengths, topk, row_starts=None):
    global _qsa_topk_triton_fn
    if _qsa_topk_triton_fn is None:
        from ...implementation.triton.indexer.qsa_topk import fast_topk

        _qsa_topk_triton_fn = fast_topk
    return _qsa_topk_triton_fn(score, lengths, topk, row_starts)


@register_impl(
    "indexer.qsa_topk", "torch_native", supported_generations={Generation.ARCH22}
)
def _qsa_topk_torch(score, lengths, topk, row_starts=None):
    global _qsa_topk_torch_fn
    if _qsa_topk_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_topk import fast_topk

        _qsa_topk_torch_fn = fast_topk
    return _qsa_topk_torch_fn(score, lengths, topk, row_starts)


@register_operator("indexer.qsa_topk")
def qsa_topk(score, lengths, topk, row_starts=None, *, impl="auto"):
    """Select sequence-relative QSA blocks from FP32 score[R,M]. topk=512/2048; contiguous int32 bounds. Returns contiguous int32[R,topk], ordered short rows and -1 suffixes. Ties may select any valid equal-valued indices. auto selects the Triton/native hybrid.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("indexer.qsa_topk", impl)(score, lengths, topk, row_starts)

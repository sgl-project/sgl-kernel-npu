"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_qsa_topk_triton_fn = None
_qsa_topk_torch_fn = None


@register_impl(
    "indexer.qsa_topk",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "indexer.qsa_topk",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_topk_triton(score, lengths, topk, row_starts=None):
    global _qsa_topk_triton_fn
    if _qsa_topk_triton_fn is None:
        from ...implementation.triton.indexer.qsa_topk import fast_topk

        _qsa_topk_triton_fn = fast_topk
    return _qsa_topk_triton_fn(score, lengths, topk, row_starts)


@register_impl(
    "indexer.qsa_topk",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_topk_torch(score, lengths, topk, row_starts=None):
    global _qsa_topk_torch_fn
    if _qsa_topk_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_topk import fast_topk

        _qsa_topk_torch_fn = fast_topk
    return _qsa_topk_torch_fn(score, lengths, topk, row_starts)


@register_operator("indexer.qsa_topk")
def qsa_topk(score, lengths, topk, row_starts=None, *, impl="auto"):
    """Select sequence-relative block indices from QSA scores.

    score is FP32 [R,M] on NPU with contiguous columns; row gaps are allowed.
    lengths and optional row_starts are contiguous int32 [R] on the same NPU.
    row_starts=None means zero starts. Integer topk is 512 or 2048.
    Return contiguous int32 [R,topk]. Rows shorter than topk keep input order
    and end in -1 padding; longer rows select the highest scores, allowing
    any valid tie choice. Inputs are unchanged. impl defaults to auto,
    selecting the Triton/native hybrid.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("indexer.qsa_topk", impl)(score, lengths, topk, row_starts)

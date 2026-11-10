"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_qsa_mqa_packed_triton_fn = None
_qsa_mqa_packed_torch_fn = None


@register_impl(
    "indexer.qsa_mqa_packed",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "indexer.qsa_mqa_packed",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_mqa_packed_triton(q, k, starts, ends):
    global _qsa_mqa_packed_triton_fn
    if _qsa_mqa_packed_triton_fn is None:
        from ...implementation.triton.indexer.qsa_mqa import packed

        _qsa_mqa_packed_triton_fn = packed
    return _qsa_mqa_packed_triton_fn(q, k, starts, ends)


@register_impl(
    "indexer.qsa_mqa_packed",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_mqa_packed_torch(q, k, starts, ends):
    global _qsa_mqa_packed_torch_fn
    if _qsa_mqa_packed_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_mqa import packed

        _qsa_mqa_packed_torch_fn = packed
    return _qsa_mqa_packed_torch_fn(q, k, starts, ends)


@register_operator("indexer.qsa_mqa_packed")
def qsa_mqa_packed(q, k, starts, ends, *, impl="auto"):
    """Compute QSA scores for keys in a packed sequence interval.

    q is contiguous BF16 [R,4,128] and k is contiguous BF16 [M,1,128].
    starts and ends are contiguous int32 [R], defining [start, end) in k.
    All tensors are on the same NPU. Return FP32 [R,M], summing
    ReLU(q @ k) / sqrt(128) over heads, with -inf outside each interval.
    impl defaults to auto, selecting Triton. Inputs are unchanged.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("indexer.qsa_mqa_packed", impl)(q, k, starts, ends)


_qsa_mqa_paged_triton_fn = None
_qsa_mqa_paged_torch_fn = None


@register_impl(
    "indexer.qsa_mqa_paged",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "indexer.qsa_mqa_paged",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_mqa_paged_triton(q, cache, table, lengths, width):
    global _qsa_mqa_paged_triton_fn
    if _qsa_mqa_paged_triton_fn is None:
        from ...implementation.triton.indexer.qsa_mqa import paged

        _qsa_mqa_paged_triton_fn = paged
    return _qsa_mqa_paged_triton_fn(q, cache, table, lengths, width)


@register_impl(
    "indexer.qsa_mqa_paged",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_mqa_paged_torch(q, cache, table, lengths, width):
    global _qsa_mqa_paged_torch_fn
    if _qsa_mqa_paged_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_mqa import paged

        _qsa_mqa_paged_torch_fn = paged
    return _qsa_mqa_paged_torch_fn(q, cache, table, lengths, width)


@register_operator("indexer.qsa_mqa_paged")
def qsa_mqa_paged(q, cache, table, lengths, width, *, impl="auto"):
    """Compute QSA scores using a physical key-cache page table.

    q is BF16 [R,4,128], cache is BF16 [P,16,1,128], table is int32 [R,L],
    and lengths is int32 [R]. All tensors are contiguous on the same NPU.
    Integer width equals 16 * L; each length is between zero and width.
    Return FP32 [R,width], summing ReLU(q @ key) / sqrt(128) over heads,
    with -inf after each sequence. Only active page-table entries are read.
    impl defaults to auto, selecting Triton. Inputs are unchanged.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("indexer.qsa_mqa_paged", impl)(
        q, cache, table, lengths, width
    )

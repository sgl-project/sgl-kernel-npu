"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_expand_qsa_blocks_triton_fn = None
_expand_qsa_blocks_torch_fn = None


@register_impl(
    "indexer.expand_qsa_blocks", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "indexer.expand_qsa_blocks", "triton", supported_generations={Generation.ARCH22}
)
def _expand_qsa_blocks_triton(blocks, positions, lengths, ratio, topk):
    global _expand_qsa_blocks_triton_fn
    if _expand_qsa_blocks_triton_fn is None:
        from ...implementation.triton.indexer.qsa_expansion import expand_blocks

        _expand_qsa_blocks_triton_fn = expand_blocks
    return _expand_qsa_blocks_triton_fn(blocks, positions, lengths, ratio, topk)


@register_impl(
    "indexer.expand_qsa_blocks",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _expand_qsa_blocks_torch(blocks, positions, lengths, ratio, topk):
    global _expand_qsa_blocks_torch_fn
    if _expand_qsa_blocks_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_expansion import expand_blocks

        _expand_qsa_blocks_torch_fn = expand_blocks
    return _expand_qsa_blocks_torch_fn(blocks, positions, lengths, ratio, topk)


@register_operator("indexer.expand_qsa_blocks")
def expand_qsa_blocks(blocks, positions, lengths, ratio, topk, *, impl="auto"):
    """Expand int32/int64 blocks[R,K] and positions/lengths[R]. topk=K*ratio, K=512/2048. Returns contiguous int32[R,topk+ratio-1], valid blocks then tail then exact -1 padding. auto selects the Triton/native hybrid.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("indexer.expand_qsa_blocks", impl)(
        blocks, positions, lengths, ratio, topk
    )

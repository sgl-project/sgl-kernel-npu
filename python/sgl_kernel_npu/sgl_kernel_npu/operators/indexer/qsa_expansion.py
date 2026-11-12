"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_expand_qsa_blocks_triton_fn = None
_expand_qsa_blocks_torch_fn = None


@register_impl(
    "indexer.expand_qsa_blocks",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "indexer.expand_qsa_blocks",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
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
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _expand_qsa_blocks_torch(blocks, positions, lengths, ratio, topk):
    global _expand_qsa_blocks_torch_fn
    if _expand_qsa_blocks_torch_fn is None:
        from ...implementation.torch_native.indexer.qsa_expansion import expand_blocks

        _expand_qsa_blocks_torch_fn = expand_blocks
    return _expand_qsa_blocks_torch_fn(blocks, positions, lengths, ratio, topk)


@register_operator("indexer.expand_qsa_blocks")
def expand_qsa_blocks(blocks, positions, lengths, ratio, topk, *, impl="auto"):
    """Expand selected complete blocks and append the current partial block.

    blocks is int32/int64 [R,K]; positions and lengths are int32/int64 [R]
    on the same NPU. Strided views are allowed. K is 512 or 2048; ratio is
    an integer >= 2, and integer topk equals K * ratio. Valid block IDs
    form a prefix followed by -1, and each block must end by both the query
    position and sequence length. Return contiguous int32 [R,topk+ratio-1]
    with valid block tokens, tail tokens, then -1 padding. Inputs are unchanged.
    impl defaults to auto, selecting the Triton/native hybrid.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("indexer.expand_qsa_blocks", impl)(
        blocks, positions, lengths, ratio, topk
    )

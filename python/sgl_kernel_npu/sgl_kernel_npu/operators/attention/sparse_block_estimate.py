"""Legacy-compatible sparse block estimation with explicit backend selection."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "attention.sparse_block_estimate", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.sparse_block_estimate",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(*args):
    return torch.ops.npu.sparse_block_estimate(*args)


@register_impl("attention.sparse_block_estimate", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.sparse_block_estimate import (
            sparse_block_estimate,
        )

        _torch_native_fn = sparse_block_estimate
    return _torch_native_fn(*args)


@register_operator("attention.sparse_block_estimate")
def sparse_block_estimate(
    query,
    key,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    input_layout="BNSD",
    stride=8,
    sparse_size=128,
    num_heads=1,
    num_key_value_heads=1,
    scale_value=1,
    threshold=1,
    causal=True,
    keep_sink=True,
    keep_recent=True,
    row_sparse=1,
    *,
    impl="auto",
):
    """Return (int8 sparse_mask, int32 sparse_count_table).

    auto selects ascendc. The positional parameters and defaults match
    torch.ops.npu.sparse_block_estimate, including causal=True.
    """
    return get_impl_by_name("attention.sparse_block_estimate", impl)(
        query,
        key,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        input_layout,
        stride,
        sparse_size,
        num_heads,
        num_key_value_heads,
        scale_value,
        threshold,
        causal,
        keep_sink,
        keep_recent,
        row_sparse,
    )

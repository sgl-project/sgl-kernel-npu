"""Public in-place LongCat token-table update operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "embedding.update_oe_token_table", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "embedding.update_oe_token_table",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(*args, **kwargs):
    return torch.ops.npu.update_oe_token_table(*args, **kwargs)


@register_impl(
    "embedding.update_oe_token_table",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.embedding.update_oe_token_table import (
            update_oe_token_table,
        )

        _torch_native_fn = update_oe_token_table
    return _torch_native_fn(*args, **kwargs)


@register_operator("embedding.update_oe_token_table")
def update_oe_token_table(
    tokens,
    req_lens,
    row_indices,
    column_starts,
    ignore_tokens,
    batch_size,
    max_context_len,
    oe_token_table,
    *,
    impl="auto",
):
    """Write packed tokens in place; auto selects Ascend C on arch22.

    All tensors are contiguous int32 on one NPU, except int64 row_indices.
    req_lens, row_indices and column_starts are [batch_size]; tokens is
    [sum(req_lens)], ignore_tokens is [I] (possibly empty), and oe_token_table
    is [R, max_context_len]. Each request writes its packed token slice to
    table[row, column_start:column_start + req_len]. Ignored tokens become -1
    in the written slices only; all other table entries retain their values.

    Returns the mutated oe_token_table with the same storage. impl is auto,
    ascendc or torch_native; torch_native provides an independent reference.
    """
    impl_fn = get_impl_by_name("embedding.update_oe_token_table", impl)
    return impl_fn(
        tokens,
        req_lens,
        row_indices,
        column_starts,
        ignore_tokens,
        batch_size,
        max_context_len,
        oe_token_table,
    )

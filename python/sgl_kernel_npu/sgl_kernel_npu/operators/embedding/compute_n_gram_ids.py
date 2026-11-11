"""Public LongCat n-gram embedding ID operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "embedding.compute_n_gram_ids", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "embedding.compute_n_gram_ids", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(*args, **kwargs):
    return torch.ops.npu.compute_n_gram_ids(*args, **kwargs)


@register_impl(
    "embedding.compute_n_gram_ids",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.embedding.compute_n_gram_ids import (
            compute_n_gram_ids,
        )

        _torch_native_fn = compute_n_gram_ids
    return _torch_native_fn(*args, **kwargs)


@register_operator("embedding.compute_n_gram_ids")
def compute_n_gram_ids(
    oe_weights,
    oe_mods,
    exclusive_oe_embeder_size_sums,
    tokens,
    exclusive_req_len_sums,
    oe_token_table,
    row_indices,
    column_starts,
    batch_size,
    oe_n,
    oe_k,
    max_context_len,
    *,
    impl="auto",
):
    """Compute IDs from the updated token table; auto selects Ascend C on arch22.

    All tensors are contiguous int32 on one NPU, except int64 row_indices.
    oe_weights has shape [oe_n - 1, oe_k, oe_n], oe_mods [oe_n - 1, oe_k],
    and exclusive_oe_embeder_size_sums [(oe_n - 1) * oe_k + 1].
    tokens has shape [T] and supplies the output length; token values are read
    from oe_token_table [R, max_context_len]. Per-request row_indices,
    column_starts and exclusive_req_len_sums have shape [batch_size]; the
    latter contains inclusive cumulative token counts ending at T.

    History stops at the row boundary or first negative token. The result is
    int32 [T, (oe_n - 1) * oe_k], with the embedding-table offset added to each
    modular n-gram hash. Inputs are not mutated. impl is auto, ascendc or
    torch_native; torch_native is an independent comparison implementation.
    """
    impl_fn = get_impl_by_name("embedding.compute_n_gram_ids", impl)
    return impl_fn(
        oe_weights,
        oe_mods,
        exclusive_oe_embeder_size_sums,
        tokens,
        exclusive_req_len_sums,
        oe_token_table,
        row_indices,
        column_starts,
        batch_size,
        oe_n,
        oe_k,
        max_context_len,
    )

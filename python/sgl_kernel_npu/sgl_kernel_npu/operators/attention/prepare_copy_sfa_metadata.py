"""Public metadata preparation operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.prepare_copy_sfa_metadata", "auto")
@register_impl("attention.prepare_copy_sfa_metadata", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.prepare_copy_sfa_metadata import (
            prepare_copy_sfa_metadata,
        )

        _triton_fn = prepare_copy_sfa_metadata
    return _triton_fn(**kwargs)


@register_impl("attention.prepare_copy_sfa_metadata", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.prepare_copy_sfa_metadata import (
            prepare_copy_sfa_metadata,
        )

        _torch_native_fn = prepare_copy_sfa_metadata
    return _torch_native_fn(**kwargs)


@register_operator("attention.prepare_copy_sfa_metadata")
def prepare_copy_sfa_metadata(
    *,
    source_ids: torch.Tensor,
    valid_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    request_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    capacity: int,
    query_width: int = 1,
    impl="auto",
) -> None:
    """Prepare Copy-SFA metadata in place; auto selects Triton.

    source_ids is int32 [B * query_width, K]; invalid rows are set to -1.
    valid_rows and sparse_rows are bool [B] flags; request_slots and seq_lens
    are int32 [B] inputs. actual_seq_lengths_kv is an int32 [B] output and
    hbm_block_table is an int32 [B, H] output. capacity is the cache size,
    query_width is the number of queries per row, and impl selects auto,
    triton or torch_native. Returns None.
    """
    return get_impl_by_name("attention.prepare_copy_sfa_metadata", impl)(
        source_ids=source_ids,
        valid_rows=valid_rows,
        sparse_rows=sparse_rows,
        request_slots=request_slots,
        seq_lens=seq_lens,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        hbm_block_table=hbm_block_table,
        capacity=capacity,
        query_width=query_width,
    )

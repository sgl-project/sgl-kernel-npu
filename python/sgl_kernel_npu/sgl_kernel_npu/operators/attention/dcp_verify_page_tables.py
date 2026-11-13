"""Public dispatch for the DCP Verify page tables operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_ARCH22 = {Generation.ARCH22}
_torch_native_fn = None


@register_impl(
    "attention.dcp_verify_page_tables", "auto", supported_generations=_ARCH22
)
@register_impl(
    "attention.dcp_verify_page_tables", "ascendc", supported_generations=_ARCH22
)
def _ascendc_impl(
    slots, requests, ordinary, local, active_cols, query_len, page_size, dcp_size
):
    return torch.ops.npu.sgl_dcp_verify_page_tables(
        slots, requests, ordinary, local, active_cols, query_len, page_size, dcp_size
    )


@register_impl("attention.dcp_verify_page_tables", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.dcp_verify_page_tables import (
            dcp_verify_page_tables as torch_native_dcp_verify_page_tables,
        )

        _torch_native_fn = torch_native_dcp_verify_page_tables
    return _torch_native_fn(*args)


@register_operator("attention.dcp_verify_page_tables")
def dcp_verify_page_tables(
    slots: torch.Tensor,
    requests: torch.Tensor,
    ordinary: torch.Tensor,
    local: torch.Tensor,
    active_cols: int,
    query_len: int,
    page_size: int,
    dcp_size: int,
    *,
    impl: str = "auto",
) -> None:
    """Build ordinary and rank-local page tables in place for Target Verify.

    Args:
        slots: Contiguous int32 tensor [pool_rows, slot_cols] of physical KV slots.
        requests: Contiguous int32 or int64 tensor [rows] of request-pool row IDs.
        ordinary: Contiguous int32 output tensor [rows, ordinary_cols].
        local: Contiguous int32 output tensor [rows * query_len, local_cols].
        active_cols: Number of populated ordinary page columns.
        query_len: Positive number of query tokens per request.
        page_size: Number of tokens per page (power of two up to 65536).
        dcp_size: Number of ranks in the DCP group (2 or 4).
        impl: Backend name. The default "auto" selects "ascendc" on arch22.

    All tensors are on the same device. Outputs must not overlap any other tensor.
    Unused columns and rows with an invalid request-pool ID are zero-filled.
    """
    return get_impl_by_name("attention.dcp_verify_page_tables", impl)(
        slots, requests, ordinary, local, active_cols, query_len, page_size, dcp_size
    )

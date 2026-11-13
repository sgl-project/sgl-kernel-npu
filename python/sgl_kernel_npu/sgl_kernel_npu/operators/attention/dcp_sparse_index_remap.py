"""Public dispatch for the DCP sparse index remap operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_ARCH22 = {Generation.ARCH22}
_torch_native_fn = None


@register_impl(
    "attention.dcp_sparse_index_remap", "auto", supported_generations=_ARCH22
)
@register_impl(
    "attention.dcp_sparse_index_remap", "ascendc", supported_generations=_ARCH22
)
def _ascendc_impl(indices, dcp_size, dcp_rank, page_size):
    return torch.ops.npu.sgl_dcp_sparse_index_remap(
        indices, dcp_size, dcp_rank, page_size
    )


@register_impl("attention.dcp_sparse_index_remap", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.dcp_sparse_index_remap import (
            dcp_sparse_index_remap as torch_native_dcp_sparse_index_remap,
        )

        _torch_native_fn = torch_native_dcp_sparse_index_remap
    return _torch_native_fn(*args)


@register_operator("attention.dcp_sparse_index_remap")
def dcp_sparse_index_remap(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Compact global sparse indices into a DCP rank's local KV positions.

    Args:
        indices: Contiguous int32 tensor [rows, 2048]. Negative entries are padding.
        dcp_size: Number of ranks in the DCP group (2 or 4).
        dcp_rank: Zero-based rank in the DCP group.
        page_size: Number of tokens per page (power of two up to 65536).
        impl: Backend name. The default "auto" selects "ascendc" on arch22.

    Returns:
        An int32 tensor with the input shape, stable within each row and padded with -1.
    """
    return get_impl_by_name("attention.dcp_sparse_index_remap", impl)(
        indices, dcp_size, dcp_rank, page_size
    )

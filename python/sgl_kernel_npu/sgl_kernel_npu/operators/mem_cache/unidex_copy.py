"""Public sparse KV indexed row copy."""

from math import prod

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


def _copy_geometry(
    src, dst, src_index, dst_index, valid_mask, src_address_ndims, dst_address_ndims
):
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )
    return src_rows, dst_rows, src_block_bytes


@register_impl(
    "mem_cache.unidex_copy_inplace", "auto", supported_generations=[Generation.ARCH22]
)
@register_impl(
    "mem_cache.unidex_copy_inplace",
    "ascendc",
    supported_generations=[Generation.ARCH22],
)
def _ascendc_impl(
    src,
    dst,
    src_index,
    dst_index,
    valid_mask,
    src_address_ndims,
    dst_address_ndims,
    block_dim,
    src_ptr,
    dst_ptr,
):
    src_rows, dst_rows, block_bytes = _copy_geometry(
        src, dst, src_index, dst_index, valid_mask, src_address_ndims, dst_address_ndims
    )
    torch.ops.npu.unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst


@register_impl(
    "mem_cache.unidex_copy_inplace",
    "torch_native",
    supported_generations=[Generation.ARCH22],
)
def _torch_native_impl(
    src,
    dst,
    src_index,
    dst_index,
    valid_mask,
    src_address_ndims,
    dst_address_ndims,
    block_dim,
    src_ptr,
    dst_ptr,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.unidex_copy import (
            unidex_copy_inplace,
        )

        _torch_native_fn = unidex_copy_inplace
    src_rows, dst_rows, block_bytes = _copy_geometry(
        src, dst, src_index, dst_index, valid_mask, src_address_ndims, dst_address_ndims
    )
    return _torch_native_fn(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        block_bytes,
        block_dim,
        src_ptr,
        dst_ptr,
    )


@register_operator("mem_cache.unidex_copy_inplace")
def unidex_copy_inplace(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: int | None = None,
    dst_ptr: int | None = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Copy indexed logical rows of a KV buffer into dst in place.

    Entries masked out or with out-of-range row indices are skipped.

    Args:
        src (Tensor): Contiguous source buffer with the same dtype as dst.
        dst (Tensor): Contiguous writable destination buffer.
        src_index (Tensor): INT64 [mappings] flat source row indices on NPU.
        dst_index (Tensor): INT64 [mappings] flat destination row indices on NPU.
        valid_mask (Tensor): Bool or UINT8 [mappings] flags on NPU.
        src_address_ndims (int): Number of leading axes that form source rows.
        dst_address_ndims (int): Number of leading axes that form destination rows.
        block_dim (int): Number of AI Vector cores to launch. Defaults to 8.
        src_ptr (int, optional): Device-visible source address override.
        dst_ptr (int, optional): Device-visible destination address override.
        impl (str): Defaults to "auto", which selects "ascendc". Use AscendC
            for address overrides and registered host memory.

    Returns:
        Tensor: The original dst object.
    """
    return get_impl_by_name("mem_cache.unidex_copy_inplace", impl)(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_address_ndims,
        dst_address_ndims,
        block_dim,
        src_ptr,
        dst_ptr,
    )

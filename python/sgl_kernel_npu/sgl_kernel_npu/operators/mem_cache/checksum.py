"""Public Adler-32 checksum over indexed, strided NPU buffers."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.adler32_strided_checksum", "auto")
@register_impl("mem_cache.adler32_strided_checksum", "triton")
def _triton_impl(data_ptrs, strides, indices):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.checksum import adler32_strided_checksum

        _triton_fn = adler32_strided_checksum
    if data_ptrs and isinstance(data_ptrs[0], torch.Tensor):
        for tensor, idx in zip(data_ptrs, indices):
            if not tensor.is_contiguous() or tensor.device != idx.device:
                raise ValueError(
                    "Tensor buffers must be contiguous and on the index device"
                )
        data_ptrs = [tensor.data_ptr() for tensor in data_ptrs]
    return _triton_fn(data_ptrs, strides, indices)


@register_impl("mem_cache.adler32_strided_checksum", "torch_native")
def _torch_native_impl(data_ptrs, strides, indices):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.checksum import (
            adler32_strided_checksum,
        )

        _torch_native_fn = adler32_strided_checksum
    return _torch_native_fn(data_ptrs, strides, indices)


@register_operator("mem_cache.adler32_strided_checksum")
def adler32_strided_checksum(
    data_ptrs: list[int] | list[torch.Tensor],
    strides: list[int],
    indices: list[torch.Tensor],
    *,
    impl="auto",
) -> int:
    """Compute ordered raw-byte Adler-32; auto selects Triton.

    Args:
        data_ptrs: Contiguous NPU Tensor buffers, one per descriptor.
            auto/triton also accept integer NPU base addresses.
        strides: Item sizes in bytes; one nonnegative integer per descriptor.
        indices: Contiguous 1D int32/int64 item indices on the same NPU.
        impl: auto, triton, or torch_native; no automatic fallback.
            torch_native requires Tensor buffers and uses CPU zlib as a reference.

    Returns:
        A Python integer, synchronizing the device result with the host.
        Empty byte streams return 1. Inputs are not modified.

    Callers retain the buffers, supply in-bounds indices, and order writes
    before this call. Bytes are ordered by descriptor, then supplied index.
    """
    impl_fn = get_impl_by_name("mem_cache.adler32_strided_checksum", impl)
    return impl_fn(data_ptrs, strides, indices)

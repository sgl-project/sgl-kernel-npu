"""Public Adler-32 checksum over indexed, strided NPU buffers."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("mem_cache.adler32_strided_checksum", "auto")
@register_impl("mem_cache.adler32_strided_checksum", "triton")
def _triton_impl(data_ptrs, strides, indices):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.checksum import adler32_strided_checksum

        _triton_fn = adler32_strided_checksum
    return _triton_fn(data_ptrs, strides, indices)


@register_operator("mem_cache.adler32_strided_checksum")
def adler32_strided_checksum(
    data_ptrs: list[int],
    strides: list[int],
    indices: list[torch.Tensor],
    *,
    impl="auto",
) -> int:
    """Compute ordered raw-byte Adler-32; auto selects Triton.

    Args:
        data_ptrs: NPU base addresses; one integer per descriptor.
        strides: Item sizes in bytes; one nonnegative integer per descriptor.
        indices: Contiguous 1D int32/int64 item indices on the same NPU.
        impl: Implementation name: auto or triton; no automatic fallback.

    Returns:
        A Python integer, synchronizing the device result with the host.
        Empty byte streams return 1. Inputs are not modified.

    Callers retain the buffers, supply in-bounds indices, and order writes
    before this call. Bytes are ordered by descriptor, then supplied index.
    """
    impl_fn = get_impl_by_name("mem_cache.adler32_strided_checksum", impl)
    return impl_fn(data_ptrs, strides, indices)

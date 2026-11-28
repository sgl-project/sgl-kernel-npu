"""Public in-place Mamba/GDN pool slot zeroing operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.clear_mamba_slots", "auto")
@register_impl("mem_cache.clear_mamba_slots", "triton")
def _triton_impl(buf, indices):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.clear_mamba_slots import (
            clear_mamba_slots,
        )

        _triton_fn = clear_mamba_slots
    return _triton_fn(buf, indices)


@register_impl("mem_cache.clear_mamba_slots", "torch_native")
def _torch_native_impl(buf, indices):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.clear_mamba_slots import (
            clear_mamba_slots,
        )

        _torch_native_fn = clear_mamba_slots
    return _torch_native_fn(buf, indices)


@register_operator("mem_cache.clear_mamba_slots")
def clear_mamba_slots(buf, indices, *, impl="auto"):
    """Zero selected dim-1 slots in place across all layers.

    auto selects Triton, which uses persistent 1D AIV blocks with 64-bit element
    offset arithmetic to avoid int32 wraparound on multi-GB SSM state pools.

    Args:
        buf (torch.Tensor): Writable NPU state pool tensor of shape
            ``[num_layers, pool_size, ...]`` with contiguous per-slot slices.
        indices (torch.Tensor): Contiguous 1D int32/int64 tensor of slot indices
            in ``[0, pool_size)`` to zero out.
        impl (str): Implementation name: ``"auto"``, ``"triton"``, or
            ``"torch_native"``. Defaults to ``"auto"``.
    """
    return get_impl_by_name("mem_cache.clear_mamba_slots", impl)(buf, indices)

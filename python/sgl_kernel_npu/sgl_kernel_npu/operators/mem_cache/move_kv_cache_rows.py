"""Public move_kv_cache_rows operator and implementation registration."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.move_kv_cache_rows", "auto")
@register_impl("mem_cache.move_kv_cache_rows", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.move_kv_cache_rows import (
            move_kv_cache_rows,
        )

        _triton_fn = move_kv_cache_rows
    return _triton_fn(*args, **kwargs)


@register_impl("mem_cache.move_kv_cache_rows", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.move_kv_cache_rows import (
            move_kv_cache_rows,
        )

        _torch_native_fn = move_kv_cache_rows
    return _torch_native_fn(*args, **kwargs)


@register_operator("mem_cache.move_kv_cache_rows")
def move_kv_cache_rows(
    keys: torch.Tensor,
    values: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    *,
    impl: str = "auto",
) -> None:
    """Move MHA KV rows in place, preserving overlapping source rows bitwise.

    ``keys`` and ``values`` are FP16/BF16 [layers, pages, slots, heads, dim]
    tensors with contiguous payloads and uniform row strides, including views
    of a shared allocation. ``tgt_loc`` and ``src_loc`` are equally sized
    one-dimensional int32/int64 tensors of flattened page/slot positions, with
    unique non-identity targets. Repeated self-copy padding is allowed.
    ``impl`` selects a backend; ``auto`` selects Triton.
    """
    return get_impl_by_name("mem_cache.move_kv_cache_rows", impl)(
        keys, values, tgt_loc, src_loc
    )

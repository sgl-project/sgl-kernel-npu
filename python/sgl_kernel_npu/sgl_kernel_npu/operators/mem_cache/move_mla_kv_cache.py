"""Public move_mla_kv_cache operator and implementation registration."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.move_mla_kv_cache", "auto")
@register_impl("mem_cache.move_mla_kv_cache", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.move_mla_kv_cache import (
            move_mla_kv_cache,
        )

        _triton_fn = move_mla_kv_cache
    return _triton_fn(*args, **kwargs)


@register_impl("mem_cache.move_mla_kv_cache", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.move_mla_kv_cache import (
            move_mla_kv_cache,
        )

        _torch_native_fn = move_mla_kv_cache
    return _torch_native_fn(*args, **kwargs)


@register_operator("mem_cache.move_mla_kv_cache")
def move_mla_kv_cache(
    k_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    page_size: int,
    use_fia_nz: bool = False,
    *,
    impl: str = "auto",
) -> None:
    """Move MLA latent and RoPE rows in place from a source snapshot.

    ``k_cache`` and ``rope_cache`` are FP16/BF16 [layers, pages, slots, 1, dim].
    ``tgt_loc`` and ``src_loc`` are matching int32/int64 slot lists, with unique
    non-identity targets and optional repeated self-copy padding. ``page_size``
    is the slots per page. With ``use_fia_nz`` the buffers are contiguous packed
    [layers, pages, dim/16, slots, 16] storage; the reserved padding slot must
    not be a target of a real move in that case. Otherwise the buffers are ND.
    ``impl`` selects a backend; ``auto`` selects Triton.
    """
    return get_impl_by_name("mem_cache.move_mla_kv_cache", impl)(
        k_cache, rope_cache, tgt_loc, src_loc, page_size, use_fia_nz
    )

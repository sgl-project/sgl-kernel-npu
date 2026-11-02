"""Public GLM image/video patch layout operator."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("multimodal.flatten_patches", "auto")
@register_impl("multimodal.flatten_patches", "torch_native")
def _torch_native_impl(patches, patch_size, merge_size, temporal_patch_size):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.multimodal.flatten_patches import (
            flatten_patches,
        )

        _torch_native_fn = flatten_patches
    return _torch_native_fn(patches, patch_size, merge_size, temporal_patch_size)


@register_operator("multimodal.flatten_patches")
def flatten_patches(
    patches: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    *,
    impl="auto",
) -> tuple[torch.Tensor, int, int, int]:
    """Flatten [B, T, C, H, W] GLM patches using at most 8-D layouts.

    ``auto`` selects Torch native. Positive patch/merge sizes must divide the
    spatial dimensions. Repeat the last frame to complete each temporal patch.
    Return [B, grid_t*grid_h*grid_w, C*temporal_patch_size*patch_size**2] and
    integer grid_t, grid_h, grid_w. Inputs are read-only; strides are supported.
    """
    impl_fn = get_impl_by_name("multimodal.flatten_patches", impl)
    return impl_fn(patches, patch_size, merge_size, temporal_patch_size)

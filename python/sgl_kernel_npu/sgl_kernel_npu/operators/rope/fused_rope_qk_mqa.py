"""Public Q/K rotary embedding operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("rope.fused_rope_qk_mqa", "auto")
@register_impl("rope.fused_rope_qk_mqa", "triton")
def _triton_impl(query, key, cos_sin, rotary_dim, is_neox_style):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.rope.fused_rope_qk_mqa import fused_rope_qk_mqa

        _triton_fn = fused_rope_qk_mqa
    return _triton_fn(query, key, cos_sin, rotary_dim, is_neox_style)


@register_impl("rope.fused_rope_qk_mqa", "torch_native")
def _torch_native_impl(query, key, cos_sin, rotary_dim, is_neox_style):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.rope.fused_rope_qk_mqa import (
            fused_rope_qk_mqa,
        )

        _torch_native_fn = fused_rope_qk_mqa
    return _torch_native_fn(query, key, cos_sin, rotary_dim, is_neox_style)


@register_operator("rope.fused_rope_qk_mqa")
def fused_rope_qk_mqa(
    query,
    key,
    cos_sin,
    rotary_dim,
    is_neox_style,
    *,
    impl="auto",
):
    """Apply Q/K RoPE; the default ``impl="auto"`` selects Triton.

    Args:
        query (torch.Tensor): Query of shape [T, Hq, D].
        key (torch.Tensor): Key of shape [T, Hk, D], matching query dtype/device.
        cos_sin (torch.Tensor): Preselected rows of shape [at least T, rotary_dim],
            with cosine in the first half and sine in the second half.
        rotary_dim (int): Positive even number of rotated features, at most D.
        is_neox_style (bool): True pairs the two halves; False pairs adjacent
            features (GPT-J layout).
        impl (str): "auto", "triton" or "torch_native".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: New query and key tensors, preserving
        their shapes, dtypes and devices. Features beyond rotary_dim are copied.
        Inputs are not modified; implementations validate their own constraints.
    """
    impl_fn = get_impl_by_name("rope.fused_rope_qk_mqa", impl)
    return impl_fn(query, key, cos_sin, rotary_dim, is_neox_style)

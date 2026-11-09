"""Public QKVZ/BA split operator for contiguous [q|k|v|z] inputs."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_qkvzba_split_reshape_cat_contiguous", "auto")
@register_impl("attention.gdn.fused_qkvzba_split_reshape_cat_contiguous", "triton")
def _triton_impl(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_qkvzba_split_reshape_cat import (
            fused_qkvzba_split_reshape_cat_contiguous,
        )

        _triton_fn = fused_qkvzba_split_reshape_cat_contiguous
    return _triton_fn(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)


@register_impl(
    "attention.gdn.fused_qkvzba_split_reshape_cat_contiguous", "torch_native"
)
def _torch_native_impl(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_qkvzba_split_reshape_cat import (
            fused_qkvzba_split_reshape_cat_contiguous,
        )

        _torch_native_fn = fused_qkvzba_split_reshape_cat_contiguous
    return _torch_native_fn(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )


@register_operator("attention.gdn.fused_qkvzba_split_reshape_cat_contiguous")
def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
    *,
    impl="auto",
):
    """Split contiguous [q|k|v|z] QKVZ and [b|a] BA projections (Qwen3.5 GDN).

    ``auto`` selects Triton; ``torch_native`` is an independent PyTorch reference.
    Same value shapes as ``fused_qkvzba_split_reshape_cat``; the difference is
    the row layout: ``mixed_qkvz`` rows are [all q | all k | all v | all z]
    (width 2*num_heads_qk*head_qk + 2*num_heads_v*head_v) and ``mixed_ba`` rows
    are [all b | all a]. Inputs are 2-D contiguous tensors on one NPU and are
    not modified.
    """
    impl_fn = get_impl_by_name(
        "attention.gdn.fused_qkvzba_split_reshape_cat_contiguous", impl
    )
    return impl_fn(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)

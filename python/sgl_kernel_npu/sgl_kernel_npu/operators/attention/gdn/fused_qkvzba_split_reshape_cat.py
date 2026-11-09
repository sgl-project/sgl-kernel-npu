"""Public QKVZ/BA split operator for per-head-group interleaved inputs."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_qkvzba_split_reshape_cat", "auto")
@register_impl("attention.gdn.fused_qkvzba_split_reshape_cat", "triton")
def _triton_impl(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_qkvzba_split_reshape_cat import (
            fused_qkvzba_split_reshape_cat,
        )

        _triton_fn = fused_qkvzba_split_reshape_cat
    return _triton_fn(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)


@register_impl("attention.gdn.fused_qkvzba_split_reshape_cat", "torch_native")
def _torch_native_impl(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_qkvzba_split_reshape_cat import (
            fused_qkvzba_split_reshape_cat,
        )

        _torch_native_fn = fused_qkvzba_split_reshape_cat
    return _torch_native_fn(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )


@register_operator("attention.gdn.fused_qkvzba_split_reshape_cat")
def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
    *,
    impl="auto",
):
    """Split per-head-group interleaved QKVZ/BA projections (Qwen3-Next GDN).

    ``auto`` selects Triton; ``torch_native`` is an independent PyTorch reference.
    Both inputs are 2-D contiguous tensors on one NPU, with ``mixed_qkvz`` of
    shape [rows, num_heads_qk * (2*head_qk + 2*(num_heads_v // num_heads_qk) *
    head_v)] and ``mixed_ba`` of [rows, 2*num_heads_v]. Returns ``mixed_qkv``
    [rows, 2*num_heads_qk*head_qk + num_heads_v*head_v], ``z``
    [rows, num_heads_v, head_v], and ``b``/``a`` of [rows, num_heads_v].
    Inputs are not modified.
    """
    impl_fn = get_impl_by_name("attention.gdn.fused_qkvzba_split_reshape_cat", impl)
    return impl_fn(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)

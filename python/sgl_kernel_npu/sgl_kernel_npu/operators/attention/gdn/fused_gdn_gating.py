"""Public GDN gate preprocessing operator with implementation selection."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_gdn_gating", "auto")
@register_impl("attention.gdn.fused_gdn_gating", "triton")
def _triton_impl(A_log, a, b, dt_bias, beta, threshold):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_gating import (
            fused_gdn_gating,
        )

        _triton_fn = fused_gdn_gating
    return _triton_fn(A_log, a, b, dt_bias, beta, threshold)


@register_impl("attention.gdn.fused_gdn_gating", "torch_native")
def _torch_native_impl(A_log, a, b, dt_bias, beta, threshold):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_gating import (
            fused_gdn_gating,
        )

        _torch_native_fn = fused_gdn_gating
    return _torch_native_fn(A_log, a, b, dt_bias, beta, threshold)


@register_operator("attention.gdn.fused_gdn_gating")
def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0, *, impl="auto"):
    """Compute GDN gate and beta outputs for [rows, num_heads] inputs.

    auto selects Triton. g = -exp(A_log) * softplus(a + dt_bias) in FP32;
    beta_output = sigmoid(b) in b.dtype. a and b must be contiguous 2-D
    tensors of the same shape on the same NPU; A_log and dt_bias are 1-D
    with num_heads elements. Inputs are not modified; empty batch is
    unsupported, and the caller guarantees beta > 0.
    """
    impl_fn = get_impl_by_name("attention.gdn.fused_gdn_gating", impl)
    return impl_fn(A_log, a, b, dt_bias, beta, threshold)

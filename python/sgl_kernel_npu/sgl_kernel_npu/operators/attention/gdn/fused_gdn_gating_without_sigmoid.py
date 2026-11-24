"""Public fused gdn gating without sigmoid."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_gdn_gating_without_sigmoid", "auto")
@register_impl("attention.gdn.fused_gdn_gating_without_sigmoid", "triton")
def _triton_impl(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_gating_without_sigmoid import (
            fused_gdn_gating_without_sigmoid,
        )

        _triton_fn = fused_gdn_gating_without_sigmoid
    return _triton_fn(A_log, a, b, dt_bias, beta, threshold)


@register_impl("attention.gdn.fused_gdn_gating_without_sigmoid", "torch_native")
def _torch_native_impl(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_gating_without_sigmoid import (
            fused_gdn_gating_without_sigmoid,
        )

        _torch_native_fn = fused_gdn_gating_without_sigmoid
    return _torch_native_fn(A_log, a, b, dt_bias, beta, threshold)


@register_operator("attention.gdn.fused_gdn_gating_without_sigmoid")
def fused_gdn_gating_without_sigmoid(
    A_log, a, b, dt_bias, beta=1.0, threshold=20.0, *, impl="auto"
):
    """Computes FP32 g=-exp(A_log)*softplus(a+dt_bias,beta,threshold), returning g with a leading singleton and forwarding b unchanged. The second result is the original b tensor, with no sigmoid applied.

    Args:
        A_log (Tensor): Floating [heads] (or equivalent singleton-expanded shape) decay weights. Default: Required.
        a (Tensor): GDN: [tokens,heads]. KDA: [tokens,Hk,K] with optional leading singleton. Default: Required.
        b (Tensor): Floating [tokens,Hv], optionally [1,tokens,Hv] for KDA. Default: Required.
        dt_bias (Tensor): GDN: [heads]; KDA: [Hk,K] gate bias. Default: Required.
        beta (float): Softplus multiplier or SiTU saturation parameter, according to the operation. Default: 1.0.
        threshold (float): Softplus linear-branch threshold. Default: 20.0.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        FP32 [1,tokens,heads] g and the unmodified b tensor.
    """
    return get_impl_by_name("attention.gdn.fused_gdn_gating_without_sigmoid", impl)(
        A_log, a, b, dt_bias, beta, threshold
    )

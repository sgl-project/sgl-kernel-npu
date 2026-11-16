"""Public fused kda gate."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.kda.fused_kda_gate", "auto")
@register_impl("attention.kda.fused_kda_gate", "triton")
def _triton_impl(
    gate, A_log, head_dim, *, gate_bias=None, beta=1.0, threshold=20.0, lower_bound=None
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.kda.kda_gate import fused_kda_gate

        _triton_fn = fused_kda_gate
    return _triton_fn(
        gate,
        A_log,
        head_dim,
        gate_bias=gate_bias,
        beta=beta,
        threshold=threshold,
        lower_bound=lower_bound,
    )


@register_impl("attention.kda.fused_kda_gate", "torch_native")
def _torch_native_impl(
    gate, A_log, head_dim, *, gate_bias=None, beta=1.0, threshold=20.0, lower_bound=None
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.kda.kda_gate import (
            fused_kda_gate,
        )

        _torch_native_fn = fused_kda_gate
    return _torch_native_fn(
        gate,
        A_log,
        head_dim,
        gate_bias=gate_bias,
        beta=beta,
        threshold=threshold,
        lower_bound=lower_bound,
    )


@register_operator("attention.kda.fused_kda_gate")
def fused_kda_gate(
    gate,
    A_log,
    head_dim,
    *,
    gate_bias=None,
    beta=1.0,
    threshold=20.0,
    lower_bound=None,
    impl="auto"
):
    """Activates KDA channel gates in FP32. The default computes -exp(A_log)*softplus(gate+bias). A lower_bound switches to lower_bound*sigmoid(exp(A_log)*(gate+bias)). The last packed dimension is expanded to [heads,head_dim].

    Args:
        gate (Tensor): Floating [...,heads*head_dim] packed gates. Default: Required.
        A_log (Tensor): Floating [heads] (or equivalent singleton-expanded shape) decay weights. Default: Required.
        head_dim (int): Channels per gate head. Default: Required.
        gate_bias (Tensor or None): Optional floating [heads*head_dim] additive gate bias. Default: None.
        beta (float): Softplus multiplier or SiTU saturation parameter, according to the operation. Default: 1.0.
        threshold (float): Softplus linear-branch threshold. Default: 20.0.
        lower_bound (float or None): Select bounded sigmoid gates when supplied; None selects softplus. Default: None.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        FP32 [...,heads,head_dim] gate activation.
    """
    return get_impl_by_name("attention.kda.fused_kda_gate", impl)(
        gate,
        A_log,
        head_dim,
        gate_bias=gate_bias,
        beta=beta,
        threshold=threshold,
        lower_bound=lower_bound,
    )

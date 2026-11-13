"""Public swiglu oai."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("activation.swiglu_oai", "auto")
@register_impl("activation.swiglu_oai", "triton")
def _triton_impl(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.swiglu_oai import swiglu_oai

        _triton_fn = swiglu_oai
    return _triton_fn(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit)


@register_impl("activation.swiglu_oai", "torch_native")
def _torch_native_impl(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.swiglu_oai import swiglu_oai

        _torch_native_fn = swiglu_oai
    return _torch_native_fn(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit)


@register_operator("activation.swiglu_oai")
def swiglu_oai(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit, *, impl="auto"):
    """Applies GPT-OSS-style SwiGLU to interleaved gate/up channels. Even channels are gate and odd channels are up. Gate is capped at limit; up is clamped to [-limit,limit]. The output is gate*sigmoid(alpha*gate)*(up+1), flattened to [rows,dim/2].

    Args:
        hidden_states (Tensor): Contiguous floating [rows,2*D] input. Default: Required.
        dim (int): Packed gate/up width, equal to the last input dimension. Default: Required.
        gemm1_alpha (float): Sigmoid input multiplier. Default: Required.
        gemm1_clamp_limit (float): Gate upper bound and symmetric up bound. Default: Required.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [rows,dim/2] output, input dtype/device.
    """
    return get_impl_by_name("activation.swiglu_oai", impl)(
        hidden_states, dim, gemm1_alpha, gemm1_clamp_limit
    )

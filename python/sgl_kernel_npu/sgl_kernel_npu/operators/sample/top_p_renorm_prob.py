"""Public top p renorm prob."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.top_p_renorm_prob", "auto")
@register_impl("sample.top_p_renorm_prob", "triton")
def _triton_impl(probs, top_ps):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.probability import top_p_renorm_prob

        _triton_fn = top_p_renorm_prob
    return _triton_fn(probs, top_ps)


@register_impl("sample.top_p_renorm_prob", "torch_native")
def _torch_native_impl(probs, top_ps):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.probability import top_p_renorm_prob

        _torch_native_fn = top_p_renorm_prob
    return _torch_native_fn(probs, top_ps)


@register_operator("sample.top_p_renorm_prob")
def top_p_renorm_prob(probs, top_ps, *, impl="auto"):
    """Filters FP32 NPU probability rows using top-p and renormalizes by max(sum,1e-20). Top-k thresholds are clamped to [1,vocab]; top-p thresholds to [0,1]. Nucleus selection compares inclusive cumulative probability minus the current value against the threshold. Inputs remain unchanged.

    Args:
        probs (Tensor): FP32 NPU [batch,vocab], 1<=vocab<=2**23. Default: Required.
        top_ps (float or Tensor): Scalar or floating [batch] nucleus thresholds. Default: Required.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        FP32 [batch,vocab] normalized probabilities, same device as probs.
    """
    return get_impl_by_name("sample.top_p_renorm_prob", impl)(probs, top_ps)

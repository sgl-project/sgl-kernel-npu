"""Public fused top-p/top-k renorm prob."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.top_k_top_p_renorm_prob", "auto")
@register_impl("sample.top_k_top_p_renorm_prob", "triton")
def _triton_impl(probs, top_ks, top_ps, max_keep):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.probability import (
            top_k_top_p_renorm_prob,
        )

        _triton_fn = top_k_top_p_renorm_prob
    return _triton_fn(probs, top_ks, top_ps, max_keep)


@register_impl("sample.top_k_top_p_renorm_prob", "torch_native")
def _torch_native_impl(probs, top_ks, top_ps, max_keep):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.probability import (
            top_k_top_p_renorm_prob,
        )

        _torch_native_fn = top_k_top_p_renorm_prob
    return _torch_native_fn(probs, top_ks, top_ps, max_keep)


@register_operator("sample.top_k_top_p_renorm_prob")
def top_k_top_p_renorm_prob(probs, top_ks, top_ps, max_keep=None, *, impl="auto"):
    """Apply top-p then top-k to each row and renormalize the joint support once.

    Top-p filters the original distribution and top-k then keeps the leading
    ranks, so the surviving columns are the intersection of the two prefix
    masks. max_keep lets the kernel select only that many leading columns; it
    must be an upper bound on every row top-k. Rows whose top-k exceeds
    max_keep are truncated to max_keep columns.

    Args:
        probs (Tensor): FP32 NPU [batch,vocab], 1<=vocab<=2**23. Default: Required.
        top_ks (int or Tensor): Scalar or integer [batch] keep counts. Default: Required.
        top_ps (float or Tensor): Scalar or floating [batch] nucleus thresholds. Default: Required.
        max_keep (int): Optional positive upper bound on every row top-k; None keeps the full vocabulary path. Default: None.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        FP32 [batch,vocab] normalized probabilities, same device as probs.
    """
    return get_impl_by_name("sample.top_k_top_p_renorm_prob", impl)(
        probs, top_ks, top_ps, max_keep
    )

"""Public top_k_top_p operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "sampler.top_k_top_p"
_torch_npu_fn = None
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(logits, p=None, k=None):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_top_k_top_p

        _torch_npu_fn = npu_top_k_top_p
    return _torch_npu_fn(logits, p, k)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(logits, p=None, k=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sampler.top_k_top_p import top_k_top_p

        _torch_native_fn = top_k_top_p
    return _torch_native_fn(logits, p, k)


@register_operator(_OP_NAME)
def top_k_top_p(logits, p=None, k=None, *, impl="auto"):
    """Filter logits by top-k followed by top-p without sampling.

    logits is a floating-point [batch, vocabulary] tensor. Provide at
    least one of p or k. The thresholds have shape [batch]; p matches
    logits.dtype and k is int32. The output matches logits and uses
    -inf for discarded entries. auto selects torch_npu.
    """
    if p is None and k is None:
        raise ValueError("Provide at least one of p or k")
    return get_impl_by_name(_OP_NAME, impl)(logits, p, k)

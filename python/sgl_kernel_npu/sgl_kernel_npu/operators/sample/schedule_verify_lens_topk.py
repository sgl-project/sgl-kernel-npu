"""Public DSpark verify-length scheduler and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.schedule_verify_lens_topk_triton", "auto")
@register_impl("sample.schedule_verify_lens_topk_triton", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.schedule_verify_lens_topk import (
            schedule_verify_lens_topk_triton,
        )

        _triton_fn = schedule_verify_lens_topk_triton
    return _triton_fn(**kwargs)


@register_impl("sample.schedule_verify_lens_topk_triton", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.schedule_verify_lens_topk import (
            schedule_verify_lens_topk_triton,
        )

        _torch_native_fn = schedule_verify_lens_topk_triton
    return _torch_native_fn(**kwargs)


@register_operator("sample.schedule_verify_lens_topk_triton")
def schedule_verify_lens_topk_triton(*, confidence, budget, cfg, impl="auto"):
    """Schedule DSpark verify lengths; auto selects Triton.

    Args:
        confidence: Floating-point tensor of shape [batch, gamma].
        budget: Maximum number of candidates to select.
        cfg: Configuration with min_verify_len, survival_eps, and
            resolved_max_verify_len().
        impl: Implementation name: auto, triton, or torch_native.

    Returns:
        A new int32 tensor of shape [batch] on the input device.
    """
    return get_impl_by_name("sample.schedule_verify_lens_topk_triton", impl)(
        confidence=confidence, budget=budget, cfg=cfg
    )

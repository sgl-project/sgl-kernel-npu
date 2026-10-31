"""Public DSpark verify-length scheduler and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("sample.schedule_verify_lens_topk_triton", "auto")
@register_impl("sample.schedule_verify_lens_topk_triton", "triton")
def _triton_impl(**kwargs):
    from ...implementation.triton.sample.schedule_verify_lens_topk import (
        schedule_verify_lens_topk_triton,
    )

    return schedule_verify_lens_topk_triton(**kwargs)


@register_impl("sample.schedule_verify_lens_topk_triton", "torch_native")
def _torch_native_impl(**kwargs):
    from ...implementation.torch_native.sample.schedule_verify_lens_topk import (
        schedule_verify_lens_topk_triton,
    )

    return schedule_verify_lens_topk_triton(**kwargs)


@register_operator("sample.schedule_verify_lens_topk_triton")
def schedule_verify_lens_topk_triton(*, confidence, budget, cfg, impl="auto"):
    """Schedule DSpark verify lengths using the selected implementation."""
    return get_impl_by_name("sample.schedule_verify_lens_topk_triton", impl)(
        confidence=confidence, budget=budget, cfg=cfg
    )

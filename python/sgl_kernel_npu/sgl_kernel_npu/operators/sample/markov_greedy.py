"""Public markov_greedy_step_triton operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.markov_greedy_step_triton", "auto")
@register_impl("sample.markov_greedy_step_triton", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.markov_greedy import (
            markov_greedy_step_triton,
        )

        _triton_fn = markov_greedy_step_triton
    return _triton_fn(**kwargs)


@register_impl("sample.markov_greedy_step_triton", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.markov_greedy import (
            markov_greedy_step_triton,
        )

        _torch_native_fn = markov_greedy_step_triton
    return _torch_native_fn(**kwargs)


@register_operator("sample.markov_greedy_step_triton")
def markov_greedy_step_triton(*, base_logits, prev_embeds, w2_weight, impl="auto"):
    """Dispatch to the selected markov_greedy_step_triton implementation."""
    return get_impl_by_name("sample.markov_greedy_step_triton", impl)(
        base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
    )

"""Public markov_sample operator and backend registration."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl(
    "sample.markov_sample", "auto", supported_generations=[Generation.ARCH22]
)
@register_impl(
    "sample.markov_sample", "triton", supported_generations=[Generation.ARCH22]
)
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.markov_sample import (
            markov_sample_step_triton,
        )

        _triton_fn = markov_sample_step_triton
    return _triton_fn(**kwargs)


@register_impl(
    "sample.markov_sample", "torch_native", supported_generations=[Generation.ARCH22]
)
def _torch_native_impl(**kwargs):
    from ...implementation.torch_native.sample.markov_sample import (
        markov_sample_step_torch,
    )

    return markov_sample_step_torch(**kwargs)


@register_operator("sample.markov_sample")
def markov_sample(
    *,
    base_logits,
    bias,
    temperatures,
    key,
    offset,
    step,
    step_dtype,
    vocab_start=0,
    gather=None,
    gather_logits=None,
    logits_out=None,
    has_sampled_rows=None,
    exp_noise=None,
    impl="auto"
):
    """Dispatch to the selected markov_sample implementation."""
    return get_impl_by_name("sample.markov_sample", impl)(
        base_logits=base_logits,
        bias=bias,
        temperatures=temperatures,
        key=key,
        offset=offset,
        step=step,
        step_dtype=step_dtype,
        vocab_start=vocab_start,
        gather=gather,
        gather_logits=gather_logits,
        logits_out=logits_out,
        has_sampled_rows=has_sampled_rows,
        exp_noise=exp_noise,
    )

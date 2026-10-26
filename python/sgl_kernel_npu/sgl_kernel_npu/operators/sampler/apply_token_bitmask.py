"""Public in-place bitmask operator with explicit implementation selection."""
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_GENERATIONS = (Generation.ARCH22, Generation.ARCH35)


@register_impl("sampler.apply_token_bitmask", "auto", supported_generations=_GENERATIONS)
@register_impl("sampler.apply_token_bitmask", "ascendc", supported_generations=_GENERATIONS)
def _ascendc_impl(logits, bitmask, indices):
    from ...implementation.ascendc.sampler.apply_token_bitmask import apply_token_bitmask
    return apply_token_bitmask(logits, bitmask, indices)


@register_impl("sampler.apply_token_bitmask", "torch_native", supported_generations=_GENERATIONS)
def _native_impl(logits, bitmask, indices):
    from ...implementation.torch_native.sampler.apply_token_bitmask import apply_token_bitmask
    return apply_token_bitmask(logits, bitmask, indices)


@register_operator("sampler.apply_token_bitmask")
def apply_token_bitmask(logits, bitmask, indices=None, *, impl="auto"):
    """Mask logits in place, returning logits. auto selects ascendc without fallback.

    logits is contiguous [batch, vocabulary] float32/float16/bfloat16;
    bitmask is contiguous [batch, words] int32 on the same NPU. Bit zero
    writes -inf; bit one preserves the value. Missing words act as zero.
    None or empty indices selects all rows, preserving the existing contract.
    """
    return get_impl_by_name("sampler.apply_token_bitmask", impl)(logits, bitmask, indices)

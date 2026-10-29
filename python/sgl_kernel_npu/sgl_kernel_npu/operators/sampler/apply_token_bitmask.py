"""Public in-place bitmask operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_cpu_reference_fn = None


@register_impl("sampler.apply_token_bitmask", "auto")
@register_impl("sampler.apply_token_bitmask", "ascendc")
def _ascendc_impl(logits, bitmask, indices):
    import torch

    return torch.ops.npu.apply_token_bitmask(logits, bitmask, indices)


@register_impl("sampler.apply_token_bitmask", "torch_native")
def _torch_native_impl(logits, bitmask, indices):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sampler.apply_token_bitmask import (
            apply_token_bitmask,
        )

        _torch_native_fn = apply_token_bitmask

    return _torch_native_fn(logits, bitmask, indices)


@register_impl("sampler.apply_token_bitmask", "cpu_reference")
def _cpu_reference_impl(logits, bitmask, indices):
    global _cpu_reference_fn
    if _cpu_reference_fn is None:
        from ...implementation.cpu_reference.sampler.apply_token_bitmask import (
            apply_token_bitmask,
        )

        _cpu_reference_fn = apply_token_bitmask

    return _cpu_reference_fn(logits, bitmask, indices)


@register_operator("sampler.apply_token_bitmask")
def apply_token_bitmask(logits, bitmask, indices=None, *, impl="auto"):
    """Mask logits in place, returning logits. auto selects ascendc without fallback.

    logits is contiguous [batch, vocabulary] float32/float16/bfloat16;
    bitmask is contiguous [batch, words] int32 on the same NPU. Bit zero
    writes -inf; bit one preserves the value. Missing words act as zero.
    None or empty indices selects all rows, preserving the existing contract.
    cpu_reference computes on CPU and copies back to the original logits;
    it also accepts CPU tensors and is intended for accuracy checks only.
    """
    impl_fn = get_impl_by_name("sampler.apply_token_bitmask", impl)
    return impl_fn(logits, bitmask, indices)

"""Public sgemmc_expand operator with explicit implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("lora.sgemmc_expand", "auto")
@register_impl("lora.sgemmc_expand", "ascendc")
def _ascendc_impl(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y):
    return torch.ops.npu.sgemmc_expand(
        x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
    )


@register_impl("lora.sgemmc_expand", "torch_native")
def _torch_native_impl(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.lora.sgemmc_expand import (
            sgemmc_expand as torch_native_sgemmc_expand,
        )

        _torch_native_fn = torch_native_sgemmc_expand

    return _torch_native_fn(
        x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
    )


@register_operator("lora.sgemmc_expand")
def sgemmc_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y, *, impl="auto"
):
    """Accumulate the LoRA B-projection into y in place, returning y.

    x is contiguous float16/bfloat16 [tokens, slices * max_rank]; weight is
    contiguous [num_loras, hidden_out, max_rank] (or [num_loras, 1, hidden_out,
    max_rank]) and y is contiguous [tokens, hidden_out]; all three share one
    dtype. The index tensors are int32 and lora_ranks values must be 0 or at
    least 16. auto selects ascendc without fallback.
    """
    impl_fn = get_impl_by_name("lora.sgemmc_expand", impl)
    return impl_fn(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y)

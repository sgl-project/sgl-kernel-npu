"""Public sgemmc_shrink operator with explicit implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("lora.sgemmc_shrink", "auto")
@register_impl("lora.sgemmc_shrink", "ascendc")
def _ascendc_impl(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
):
    torch.ops.npu.sgemmc_shrink(
        x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
    )
    return y


@register_impl("lora.sgemmc_shrink", "torch_native")
def _torch_native_impl(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.lora.sgemmc_shrink import (
            sgemmc_shrink as torch_native_sgemmc_shrink,
        )

        _torch_native_fn = torch_native_sgemmc_shrink

    return _torch_native_fn(
        x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
    )


@register_operator("lora.sgemmc_shrink")
def sgemmc_shrink(
    x,
    weight,
    lora_indices,
    seq_len,
    lora_ranks,
    lora_scales,
    y,
    slice_count,
    *,
    impl="auto",
):
    """Write the LoRA A-projection into y in place, returning y.

    Same contract as sgemmv_shrink except that y keeps the activation dtype:
    x and y are contiguous float16/bfloat16, y is [tokens, slices * max_rank],
    weight is [num_loras, slices * max_rank, hidden_in], lora_scales is float16
    and the three index tensors are int32. slice_count is required. Ranks must
    be 0 or at least 16. auto selects ascendc without fallback.
    """
    impl_fn = get_impl_by_name("lora.sgemmc_shrink", impl)
    return impl_fn(
        x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
    )

"""Public sgemmv_shrink operator with explicit implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("lora.sgemmv_shrink", "auto")
@register_impl("lora.sgemmv_shrink", "ascendc")
def _ascendc_impl(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count=1
):
    torch.ops.npu.sgemmv_shrink(
        x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
    )
    return y


@register_impl("lora.sgemmv_shrink", "torch_native")
def _torch_native_impl(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count=1
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.lora.sgemmv_shrink import (
            sgemmv_shrink as torch_native_sgemmv_shrink,
        )

        _torch_native_fn = torch_native_sgemmv_shrink

    return _torch_native_fn(
        x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
    )


@register_operator("lora.sgemmv_shrink")
def sgemmv_shrink(
    x,
    weight,
    lora_indices,
    seq_len,
    lora_ranks,
    lora_scales,
    y,
    slice_count=1,
    *,
    impl="auto",
):
    """Write the LoRA A-projection into y in place, returning y.

    x is contiguous float16/bfloat16 [tokens, hidden_in]; weight is contiguous
    [num_loras, slices * max_rank, hidden_in] (or the equivalent 4D form) with
    the same dtype as x; y is contiguous float32 [tokens, slices * max_rank].
    lora_scales must be float16 and the three index tensors int32. Ranks must be
    0, 8, 16, 32 or 64. auto selects ascendc without fallback.
    """
    impl_fn = get_impl_by_name("lora.sgemmv_shrink", impl)
    return impl_fn(
        x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
    )

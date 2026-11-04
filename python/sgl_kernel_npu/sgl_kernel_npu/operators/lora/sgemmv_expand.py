"""Public sgemmv_expand operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("lora.sgemmv_expand", "auto")
@register_impl("lora.sgemmv_expand", "ascendc")
def _ascendc_impl(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y):
    import torch

    return torch.ops.npu.sgemmv_expand(
        x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
    )


@register_impl("lora.sgemmv_expand", "torch_native")
def _torch_native_impl(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.lora.sgemmv_expand import (
            sgemmv_expand as torch_native_sgemmv_expand,
        )

        _torch_native_fn = torch_native_sgemmv_expand

    return _torch_native_fn(
        x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
    )


@register_operator("lora.sgemmv_expand")
def sgemmv_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y, *, impl="auto"
):
    """Accumulate LoRA expand output into y in place, returning y.

    x is contiguous float32 [tokens, slices * max_rank]; weight is contiguous
    [num_loras, hidden_out, max_rank] (or [num_loras, 1, hidden_out, max_rank])
    with the same dtype as y; the four index tensors are int32 vectors on the
    same NPU. lora_ranks values must be 0, 8, 16, 32 or 64. auto selects
    ascendc without fallback.
    """
    impl_fn = get_impl_by_name("lora.sgemmv_expand", impl)
    return impl_fn(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y)

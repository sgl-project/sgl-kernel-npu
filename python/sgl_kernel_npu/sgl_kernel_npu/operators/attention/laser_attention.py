"""Public LaserAttention dispatch for the independent attention CANN operator."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "attention.laser_attention", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.laser_attention", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(
    query,
    key,
    value,
    atten_mask,
    alibi_mask,
    drop_mask,
    scale_value,
    head_num,
    input_layout,
    keep_prob,
    pre_tokens,
    next_tokens,
    is_highPrecision,
):
    import torch

    return torch.ops.npu.la(
        query,
        key,
        value,
        atten_mask,
        alibi_mask,
        drop_mask,
        scale_value,
        head_num,
        input_layout,
        keep_prob,
        pre_tokens,
        next_tokens,
        is_highPrecision,
    )


@register_impl("attention.laser_attention", "torch_native")
def _torch_native_impl(
    query,
    key,
    value,
    atten_mask,
    alibi_mask,
    drop_mask,
    scale_value,
    head_num,
    input_layout,
    keep_prob,
    pre_tokens,
    next_tokens,
    is_highPrecision,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.laser_attention import (
            laser_attention,
        )

        _torch_native_fn = laser_attention
    return _torch_native_fn(
        query,
        key,
        value,
        atten_mask=atten_mask,
        alibi_mask=alibi_mask,
        drop_mask=drop_mask,
        scale_value=scale_value,
        head_num=head_num,
        input_layout=input_layout,
        keep_prob=keep_prob,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        is_high_precision=is_highPrecision,
    )


@register_operator("attention.laser_attention")
def la(
    query,
    key,
    value,
    *,
    atten_mask=None,
    alibi_mask=None,
    drop_mask=None,
    scale_value=1.0,
    head_num=2,
    input_layout="BNSD",
    keep_prob=1.0,
    pre_tokens=2147483647,
    next_tokens=1,
    is_highPrecision=True,
    impl="auto",
):
    """Return (softmax_log_max_sum, attention_out).

    auto selects ascendc on arch22 without fallback. The device Kernel computes
    full (non-causal) attention over BNSD tensors and supports GQA.

    softmax_log_max_sum (Tensor): FP32 [B, N, S1]; the current Kernel never
        writes it, so its content is undefined.
    attention_out (Tensor): FP32 [B, N, S1, D].
    """
    return get_impl_by_name("attention.laser_attention", impl)(
        query,
        key,
        value,
        atten_mask,
        alibi_mask,
        drop_mask,
        scale_value,
        head_num,
        input_layout,
        keep_prob,
        pre_tokens,
        next_tokens,
        is_highPrecision,
    )

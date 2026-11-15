"""Public packed recurrent gated delta rule with caller-owned mutable state."""

import torch

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_native = None


@register_impl(
    "attention.gdn.recurrent_gated_delta_rule",
    "auto",
)
@register_impl(
    "attention.gdn.recurrent_gated_delta_rule",
    "ascendc",
)
def _ascendc_impl(*args, **kwargs):
    return torch.ops.npu.recurrent_gated_delta_rule(*args, **kwargs)


@register_impl("attention.gdn.recurrent_gated_delta_rule", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native
    if _native is None:
        from ....implementation.torch_native.attention.gdn.recurrent_gated_delta_rule import (
            recurrent_gated_delta_rule,
        )

        _native = recurrent_gated_delta_rule
    return _native(*args, **kwargs)


@register_operator("attention.gdn.recurrent_gated_delta_rule")
def recurrent_gated_delta_rule(
    mix_qkv,
    recurrent_state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    nk,
    nv,
    intermediate_state=None,
    cache_indices=None,
    num_accepted_tokens=None,
    g=None,
    gk=None,
    *,
    impl="auto",
):
    """Return BF16 [B,S,Nv,Dv] and write per-token states; auto selects ascendc.

    Args:
        mix_qkv: BF16 Tensor [B,S,2*Nk*Dk+Nv*Dv], packed Q, K, V; S <= 8.
        recurrent_state: BF16 Tensor [N,Nv,Dv,Dk], mutable state pool or,
            when intermediate_state is supplied, the unchanged seed pool.
        beta: BF16 Tensor [B,S,Nv], pre-sigmoid update logits.
        scale: Required float Q multiplier, narrowed to FP32; zero is preserved.
        actual_seq_lengths: int32 Tensor [B], lengths in [1,S], not prefix sums.
        ssm_state_indices: int32 Tensor [B,S], initial and per-token state slots.
        nk: Positive int Q/K head count.
        nv: Positive int value head count, divisible by nk.
        intermediate_state: Optional BF16 Tensor of complete [Nv,Dv,Dk] slots;
            defaults to None. A contiguous pool receives per-token states.
        cache_indices: Optional Tensor with at least B seed slots, converted
            to contiguous int32; defaults to None, required with an intermediate pool.
        num_accepted_tokens: Optional Tensor with at least B counts in [1,S],
            converted to contiguous int32; selects initial column count-1.
            Defaults to None (column 0); does not shorten the output sequence.
        g: Optional Tensor with at least B*S*Nv log-decay values, converted to
            contiguous FP32. Defaults to None (decay 1).
        gk: Optional Tensor of any shape, converted to contiguous FP32 but
            ignored by the computation; defaults to None.
        impl: Keyword-only str, "auto" (default), "ascendc" or "torch_native".

    Returns:
        BF16 Tensor [B,S,Nv,Dv]; only tokens below each valid length are defined.

    All tensors share a device (NPU for ascendc). Required tensors retain dense
    physical-span reads regardless of logical strides. A noncontiguous optional
    working pool uses a temporary without copyback. See the API page for the
    complete state-selection protocol and native copy-size limits.
    """
    return get_impl_by_name("attention.gdn.recurrent_gated_delta_rule", impl)(
        mix_qkv,
        recurrent_state,
        beta,
        scale,
        actual_seq_lengths,
        ssm_state_indices,
        nk,
        nv,
        intermediate_state=intermediate_state,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        g=g,
        gk=gk,
    )

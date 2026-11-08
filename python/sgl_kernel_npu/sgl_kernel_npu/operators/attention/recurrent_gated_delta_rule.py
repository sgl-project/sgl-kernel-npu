"""Public packed recurrent gated delta rule with caller-owned mutable state."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_GENERATIONS = (Generation.ARCH22, Generation.ARCH35)
_native = None


@register_impl(
    "attention.recurrent_gated_delta_rule",
    "auto",
)
@register_impl(
    "attention.recurrent_gated_delta_rule",
    "ascendc",
)
def _ascendc_impl(*args, **kwargs):
    import torch

    return torch.ops.npu.recurrent_gated_delta_rule(*args, **kwargs)


@register_impl("attention.recurrent_gated_delta_rule", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native
    if _native is None:
        from ...implementation.torch_native.attention.recurrent_gated_delta_rule import (
            recurrent_gated_delta_rule,
        )

        _native = recurrent_gated_delta_rule
    return _native(*args, **kwargs)


@register_operator("attention.recurrent_gated_delta_rule")
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

    mix_qkv is BF16 [B,S,2*Nk*Dk+Nv*Dv], beta is BF16 [B,S,Nv], and
    recurrent_state is BF16 [N,Nv,Dv,Dk]. scale is required and zero is kept.
    actual_seq_lengths is int32 [B], not a cumulative prefix sum; state indices
    are int32 [B,S]. nk/nv are positive head counts, with nv divisible by nk.
    Optional intermediate_state supplies the mutable working pool; cache_indices
    selects its recurrent seeds. Optional num_accepted_tokens selects the initial
    state index column. g is an FP32 log-decay after conversion; gk is accepted
    for legacy compatibility but ignored. See the API page for full state/stride
    semantics, optional conversions and the S <= 8 restriction.
    """
    return get_impl_by_name("attention.recurrent_gated_delta_rule", impl)(
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

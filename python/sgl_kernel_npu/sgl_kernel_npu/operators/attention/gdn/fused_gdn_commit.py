"""Public Ascend 950 (arch35) single-launch all-layer Cube GDN SSM commit operator."""

from ...utils.generation import Generation
from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl(
    "attention.gdn.fused_gdn_commit", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "attention.gdn.fused_gdn_commit",
    "triton",
    supported_generations={Generation.ARCH35},
)
def _triton_impl(
    ssm_states_all,
    rec_k_all,
    rec_v_all,
    rec_g_all,
    dst_slots,
    last_steps,
    track_slots,
    track_steps,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_verify import (
            fused_gdn_commit,
        )

        _triton_fn = fused_gdn_commit
    return _triton_fn(
        ssm_states_all,
        rec_k_all,
        rec_v_all,
        rec_g_all,
        dst_slots,
        last_steps,
        track_slots=track_slots,
        track_steps=track_steps,
    )


@register_impl("attention.gdn.fused_gdn_commit", "torch_native")
def _torch_native_impl(
    ssm_states_all,
    rec_k_all,
    rec_v_all,
    rec_g_all,
    dst_slots,
    last_steps,
    track_slots,
    track_steps,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_verify import (
            fused_gdn_commit,
        )

        _torch_native_fn = fused_gdn_commit
    return _torch_native_fn(
        ssm_states_all,
        rec_k_all,
        rec_v_all,
        rec_g_all,
        dst_slots,
        last_steps,
        track_slots=track_slots,
        track_steps=track_steps,
    )


@register_operator("attention.gdn.fused_gdn_commit")
def fused_gdn_commit(
    ssm_states_all,
    rec_k_all,
    rec_v_all,
    rec_g_all,
    dst_slots,
    last_steps,
    *,
    track_slots=None,
    track_steps=None,
    impl="auto",
):
    """Commit accepted speculative GDN states in place across all layers in a single launch.

    auto selects Triton on ``Generation.ARCH35``. Computes the exact post-accept state
    ``S_last = exp(g_last) * S_0 + sum_{t<=last} exp(g_last - g_t) * v_t @ k_t^T``
    from compact verify records ``(rec_k_all, rec_v_all, rec_g_all)``, optionally
    writing an earlier Radix Cache checkpoint at ``track_steps[b]`` into
    ``ssm_states_all[:, track_slots[b]]`` before overwriting ``dst_slots[b]``.

    Args:
        ssm_states_all (torch.Tensor): Contiguous bfloat16 ``[L, num_slots, NV, 128, 128]``, updated in place.
        rec_k_all (torch.Tensor): bfloat16 ``[L, max_bs, K_VER, NK, 128]`` normalized keys from verify.
        rec_v_all (torch.Tensor): bfloat16 ``[L, max_bs, K_VER, NV, 128]`` WY innovation vectors from verify.
        rec_g_all (torch.Tensor): float32 ``[L, max_bs, K_VER, NV]`` cumulative log-decay from verify.
        dst_slots (torch.Tensor): 1D integer ``[B]`` destination slots (``< 0`` skips).
        last_steps (torch.Tensor): 1D integer ``[B]`` accepted step indices in ``[0, K_VER - 1]`` (``< 0`` skips).
        track_slots (torch.Tensor | None): Optional 1D integer ``[B]`` Radix checkpoint slots (``< 0`` skips).
        track_steps (torch.Tensor | None): Optional 1D integer ``[B]`` Radix checkpoint steps (``< 0`` skips).
        impl (str): ``"auto"``, ``"triton"`` (``ARCH35``), or ``"torch_native"``.

    Returns:
        torch.Tensor: Original ``ssm_states_all`` alias updated in place.
    """
    return get_impl_by_name("attention.gdn.fused_gdn_commit", impl)(
        ssm_states_all,
        rec_k_all,
        rec_v_all,
        rec_g_all,
        dst_slots,
        last_steps,
        track_slots,
        track_steps,
    )

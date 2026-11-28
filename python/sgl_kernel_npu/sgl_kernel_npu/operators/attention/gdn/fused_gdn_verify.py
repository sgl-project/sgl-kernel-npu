"""Public Ascend 950 (arch35) Cube-accelerated GDN speculative verify operator."""

from ...utils.generation import Generation
from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


def create_gdn_verify_workspace():
    """Create a persistent static ``FusedGDNVerifyWorkspace`` for ACLGraph capture."""
    from ....implementation.triton.attention.gdn.fused_gdn_verify import (
        FusedGDNVerifyWorkspace,
    )

    return FusedGDNVerifyWorkspace()


@register_impl(
    "attention.gdn.fused_gdn_verify", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "attention.gdn.fused_gdn_verify",
    "triton",
    supported_generations={Generation.ARCH35},
)
def _triton_impl(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    num_k_heads,
    num_v_heads,
    head_dim,
    eps,
    workspace,
    linear_layer_idx,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_verify import (
            fused_gdn_verify,
        )

        _triton_fn = fused_gdn_verify
    return _triton_fn(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        num_k_heads,
        num_v_heads,
        head_dim=head_dim,
        eps=eps,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
    )


@register_impl("attention.gdn.fused_gdn_verify", "torch_native")
def _torch_native_impl(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    num_k_heads,
    num_v_heads,
    head_dim,
    eps,
    workspace,
    linear_layer_idx,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_verify import (
            fused_gdn_verify,
        )

        _torch_native_fn = fused_gdn_verify
    return _torch_native_fn(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        num_k_heads,
        num_v_heads,
        head_dim=head_dim,
        eps=eps,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
    )


@register_operator("attention.gdn.fused_gdn_verify")
def fused_gdn_verify(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    num_k_heads,
    num_v_heads,
    head_dim=128,
    eps=1e-6,
    workspace=None,
    linear_layer_idx=0,
    *,
    impl="auto",
):
    """Run Cube-accelerated GDN speculative verify from pre-convolved ``(mixed_qkv, a, b)``.

    auto selects Triton on ``Generation.ARCH35``. Computes Q/K L2 normalization, GDN
    gating, WY chunk decomposition (single-stage for ``B <= 8`` or non-power-of-two
    ``K_VER``; multi-request R-packed two-stage otherwise), and outputs both the
    attention output and compact commit records ``(rec_k, rec_v, rec_g)`` without
    mutating ``ssm_states``.

    Args:
        mixed_qkv (torch.Tensor): Contiguous bfloat16 ``[B, K_VER, (2*NK + NV)*D]``.
        a (torch.Tensor): Contiguous ``[B, K_VER, NV]`` pre-softplus decay logits.
        b (torch.Tensor): Contiguous ``[B, K_VER, NV]`` pre-sigmoid beta logits.
        A_log (torch.Tensor): Contiguous ``[NV]`` float32 log-A parameter.
        dt_bias (torch.Tensor): Contiguous ``[NV]`` decay bias parameter.
        ssm_states (torch.Tensor): Contiguous read-only bfloat16 ``[num_slots, NV, 128, 128]``.
        cache_indices (torch.Tensor): 1D int32 active state slots ``[B]`` (``< 0`` uses zeros).
        num_k_heads (int): Number of Q/K heads (divides ``num_v_heads``).
        num_v_heads (int): Number of V heads.
        head_dim (int): Per-head dimension (must be 128). Default: 128.
        eps (float): L2Norm epsilon. Default: 1e-6.
        workspace (FusedGDNVerifyWorkspace | None): Optional pre-allocated static workspace.
        linear_layer_idx (int): Layer slot index inside ``workspace``. Default: 0.
        impl (str): ``"auto"``, ``"triton"`` (``ARCH35``), or ``"torch_native"``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: ``(out, rec_k, rec_v, rec_g)`` with shapes
        ``[B, K_VER, NV, 128]``, ``[B, K_VER, NK, 128]``, ``[B, K_VER, NV, 128]``,
        and ``[B, K_VER, NV]``.
    """
    return get_impl_by_name("attention.gdn.fused_gdn_verify", impl)(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        num_k_heads,
        num_v_heads,
        head_dim,
        eps,
        workspace,
        linear_layer_idx,
    )

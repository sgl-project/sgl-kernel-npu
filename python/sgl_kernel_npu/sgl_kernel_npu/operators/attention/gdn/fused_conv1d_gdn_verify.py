"""Public Ascend 950 (arch35) fused Conv1D + GDN speculative verify operator."""

from ...utils.generation import Generation
from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl(
    "attention.gdn.fused_conv1d_gdn_verify",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.gdn.fused_conv1d_gdn_verify",
    "triton",
    supported_generations={Generation.ARCH35},
)
def _triton_impl(
    qkvz,
    ba,
    conv_weights_t,
    conv_states,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    batch_size,
    k_ver,
    num_k_heads,
    num_v_heads,
    head_dim,
    eps,
    query_start_loc,
    workspace,
    linear_layer_idx,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_verify import (
            fused_conv1d_gdn_verify,
        )

        _triton_fn = fused_conv1d_gdn_verify
    return _triton_fn(
        qkvz,
        ba,
        conv_weights_t,
        conv_states,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        batch_size,
        k_ver,
        num_k_heads,
        num_v_heads,
        head_dim=head_dim,
        eps=eps,
        query_start_loc=query_start_loc,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
    )


@register_impl("attention.gdn.fused_conv1d_gdn_verify", "torch_native")
def _torch_native_impl(
    qkvz,
    ba,
    conv_weights_t,
    conv_states,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    batch_size,
    k_ver,
    num_k_heads,
    num_v_heads,
    head_dim,
    eps,
    query_start_loc,
    workspace,
    linear_layer_idx,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_verify import (
            fused_conv1d_gdn_verify,
        )

        _torch_native_fn = fused_conv1d_gdn_verify
    return _torch_native_fn(
        qkvz,
        ba,
        conv_weights_t,
        conv_states,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        batch_size,
        k_ver,
        num_k_heads,
        num_v_heads,
        head_dim=head_dim,
        eps=eps,
        query_start_loc=query_start_loc,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
    )


@register_operator("attention.gdn.fused_conv1d_gdn_verify")
def fused_conv1d_gdn_verify(
    qkvz,
    ba,
    conv_weights_t,
    conv_states,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    batch_size,
    k_ver,
    num_k_heads,
    num_v_heads,
    head_dim=128,
    eps=1e-6,
    query_start_loc=None,
    workspace=None,
    linear_layer_idx=0,
    *,
    impl="auto",
):
    """Run fused strided-QKVZ causal Conv1D (kernel=4, SiLU) + Cube GDN speculative verify.

    auto selects Triton on ``Generation.ARCH35``. Reads the leading ``QKV_DIM`` columns
    of ``qkvz`` directly, updates ``conv_states`` in place for valid non-padded slots
    (skipping ``cache_indices[b] < 0`` or empty ``query_start_loc`` rows), and runs
    Cube WY-chunk GDN verify without mutating ``ssm_states``.

    Args:
        qkvz (torch.Tensor): Contiguous bfloat16 ``[B * k_ver, qkvz_dim]`` with ``qkvz_dim >= QKV_DIM``.
        ba (torch.Tensor): Contiguous bfloat16 ``[B * k_ver, 2 * NV]`` (``[b | a]`` layout).
        conv_weights_t (torch.Tensor): Contiguous ``[4, QKV_DIM]`` transposed conv1d weights.
        conv_states (torch.Tensor): Contiguous bfloat16 ``[num_slots, k_ver + 2, QKV_DIM]`` updated in place.
        A_log (torch.Tensor): Contiguous ``[NV]`` float32 log-A parameter.
        dt_bias (torch.Tensor): Contiguous ``[NV]`` decay bias parameter.
        ssm_states (torch.Tensor): Contiguous read-only bfloat16 ``[num_slots, NV, 128, 128]``.
        cache_indices (torch.Tensor): 1D int32 active state slots ``[B]``.
        batch_size (int): Number of requests ``B``.
        k_ver (int): Verification width ``K_VER`` in ``[2, 16]``.
        num_k_heads (int): Number of Q/K heads (divides ``num_v_heads``).
        num_v_heads (int): Number of V heads.
        head_dim (int): Per-head dimension (must be 128). Default: 128.
        eps (float): L2Norm epsilon. Default: 1e-6.
        query_start_loc (torch.Tensor | None): Optional ``[B + 1]`` int32 cumulative token offsets;
            rows with ``qsl[b+1] <= qsl[b]`` skip ``conv_states`` updates.
        workspace (FusedGDNVerifyWorkspace | None): Optional pre-allocated static workspace.
        linear_layer_idx (int): Layer slot index inside ``workspace``. Default: 0.
        impl (str): ``"auto"``, ``"triton"`` (``ARCH35``), or ``"torch_native"``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: ``(out, rec_k, rec_v, rec_g)``.
    """
    return get_impl_by_name("attention.gdn.fused_conv1d_gdn_verify", impl)(
        qkvz,
        ba,
        conv_weights_t,
        conv_states,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        batch_size,
        k_ver,
        num_k_heads,
        num_v_heads,
        head_dim,
        eps,
        query_start_loc,
        workspace,
        linear_layer_idx,
    )

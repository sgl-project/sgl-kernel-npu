"""Public zero-copy strided-Z gated RMSNorm operator for packed GDN projections."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_rmsnorm_gated_strided_z", "auto")
@register_impl("attention.gdn.fused_rmsnorm_gated_strided_z", "triton")
def _triton_impl(
    core_attn_out,
    projected_states_qkvz,
    norm_weight,
    z_offset,
    eps,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_rmsnorm_gated_strided_z import (
            fused_rmsnorm_gated_strided_z,
        )

        _triton_fn = fused_rmsnorm_gated_strided_z
    return _triton_fn(
        core_attn_out,
        projected_states_qkvz,
        norm_weight,
        z_offset,
        eps,
    )


@register_impl("attention.gdn.fused_rmsnorm_gated_strided_z", "torch_native")
def _torch_native_impl(
    core_attn_out,
    projected_states_qkvz,
    norm_weight,
    z_offset,
    eps,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_rmsnorm_gated_strided_z import (
            fused_rmsnorm_gated_strided_z,
        )

        _torch_native_fn = fused_rmsnorm_gated_strided_z
    return _torch_native_fn(
        core_attn_out,
        projected_states_qkvz,
        norm_weight,
        z_offset,
        eps,
    )


@register_operator("attention.gdn.fused_rmsnorm_gated_strided_z")
def fused_rmsnorm_gated_strided_z(
    core_attn_out,
    projected_states_qkvz,
    norm_weight,
    z_offset,
    eps=1e-6,
    *,
    impl="auto",
):
    """Compute RMSNorm(core_attn_out) * SiLU(z) reading strided z directly from qkvz.

    auto selects Triton. Avoids materializing a contiguous copy of ``z`` from packed
    ``projected_states_qkvz [T, QKVZ_DIM]`` and uses 64-bit token-stride indexing.

    Args:
        core_attn_out (torch.Tensor): Contiguous ``[T, NV, D]`` attention output.
        projected_states_qkvz (torch.Tensor): Contiguous ``[T, QKVZ_DIM]`` packed
            projection tensor with ``z`` starting at column ``z_offset``.
        norm_weight (torch.Tensor): Contiguous ``[D]`` per-head RMSNorm weight.
        z_offset (int): Starting column offset of ``z`` inside ``projected_states_qkvz``.
        eps (float): RMSNorm epsilon. Defaults to ``1e-6``.
        impl (str): ``"auto"``, ``"triton"``, or ``"torch_native"``. Defaults to ``"auto"``.

    Returns:
        torch.Tensor: Contiguous ``[T, NV * D]`` gated normalized output.
    """
    return get_impl_by_name("attention.gdn.fused_rmsnorm_gated_strided_z", impl)(
        core_attn_out,
        projected_states_qkvz,
        norm_weight,
        z_offset,
        eps,
    )

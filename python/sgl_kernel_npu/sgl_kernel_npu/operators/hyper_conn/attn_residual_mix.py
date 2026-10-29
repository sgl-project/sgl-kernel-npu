"""Public attention-residual scoring and mixing operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("hyper_conn.attn_res_mix_fused", "auto")
@register_impl("hyper_conn.attn_res_mix_fused", "triton")
def _triton_impl(prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.hyper_conn.attn_residual_mix import (
            attn_res_mix_fused,
        )

        _triton_fn = attn_res_mix_fused
    return _triton_fn(
        prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
    )


@register_impl("hyper_conn.attn_res_mix_fused", "torch_native")
def _torch_native_impl(
    prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.attn_residual_mix import (
            attn_res_mix_fused,
        )

        _torch_native_fn = attn_res_mix_fused
    return _torch_native_fn(
        prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
    )


@register_operator("hyper_conn.attn_res_mix_fused")
def attn_res_mix_fused(
    prefix_sum,
    bank,
    num_valid_blocks,
    combined_weight,
    variance_epsilon,
    *,
    impl="auto",
):
    """Score residual rows with RMS normalization, then mix their original values.

    The default ``impl="auto"`` selects Triton. No output RMSNorm is applied.

    Args:
        prefix_sum (torch.Tensor): Current residual prefix of shape [N, H].
        bank (torch.Tensor): Residual snapshots of shape [N, C, H]. Only the
            first num_valid_blocks rows of each token are read.
        num_valid_blocks (int): Number of valid bank rows, between 0 and C.
        combined_weight (torch.Tensor): Scoring weight of shape [H], typically
            the product of RMSNorm weight and scalar projection weight.
        variance_epsilon (float): Finite positive scoring RMS stability term.
        impl (str): "auto", "triton" or "torch_native".

    Returns:
        torch.Tensor: Mixture of shape [N, H], matching prefix_sum dtype/device.
        A nonempty result is newly allocated. With N=0, returns prefix_sum itself.
        All arithmetic intermediates are FP32. Inputs are not modified.
    """
    impl_fn = get_impl_by_name("hyper_conn.attn_res_mix_fused", impl)
    return impl_fn(
        prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
    )

"""Public LPLB probability-table builder."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("moe.lplb.lp_post", "auto")
@register_impl("moe.lplb.lp_post", "triton")
def _triton_impl(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.moe.lplb.lp_post import lp_post

        _triton_fn = lp_post
    return _triton_fn(
        log2phy_prob,
        x,
        t1,
        phy_single,
        phy_replicated,
        log2phy,
    )


@register_impl("moe.lplb.lp_post", "torch_native")
def _torch_native_impl(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.moe.lplb.lp_post import lp_post

        _torch_native_fn = lp_post
    return _torch_native_fn(
        log2phy_prob,
        x,
        t1,
        phy_single,
        phy_replicated,
        log2phy,
    )


@register_operator("moe.lplb.lp_post")
def lp_post(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
    *,
    impl="auto",
):
    """Turn the LPLB LP solution into ``log2phy_prob`` in place.

    The default impl="auto" selects Triton. Only ``log2phy_prob`` is written.

    Args:
        log2phy_prob (torch.Tensor): Contiguous float32 [num_logical,
            max_copies] output buffer, overwritten.
        x (torch.Tensor): Contiguous float32 [num_red_phy] replica ratios,
            usually ``ipm_solution[:num_red_phy]``. Negative ratios clamp to 0.
        t1 (torch.Tensor): Contiguous float32 [num_single] single-copy shares
            produced by ``lp_prep``.
        phy_single (torch.Tensor): Contiguous int32/int64 [num_single] physical
            slot of each single-copy expert.
        phy_replicated (torch.Tensor): Contiguous int32/int64 [num_red_phy]
            physical slot of each replica.
        log2phy (torch.Tensor): Contiguous int32/int64 [num_logical,
            max_copies] logical-to-physical table; ``-1`` marks an unused
            replica and reads the always-zero sink slot.
        impl (str): Implementation name: "auto", "triton", or "torch_native".
            Defaults to "auto".

    Returns:
        torch.Tensor: The overwritten ``log2phy_prob``.
    """
    return get_impl_by_name("moe.lplb.lp_post", impl)(
        log2phy_prob,
        x,
        t1,
        phy_single,
        phy_replicated,
        log2phy,
    )

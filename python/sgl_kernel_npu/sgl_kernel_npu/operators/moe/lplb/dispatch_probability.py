"""Public LPLB physical-expert dispatch."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("moe.lplb.dispatch_probability", "auto")
@register_impl("moe.lplb.dispatch_probability", "triton")
def _triton_impl(topk_ids, log2phy_prob, log2phy_map, random_vals=None):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.moe.lplb.dispatch_probability import (
            dispatch_probability,
        )

        _triton_fn = dispatch_probability
    return _triton_fn(topk_ids, log2phy_prob, log2phy_map, random_vals)


@register_impl("moe.lplb.dispatch_probability", "torch_native")
def _torch_native_impl(topk_ids, log2phy_prob, log2phy_map, random_vals=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.moe.lplb.dispatch_probability import (
            dispatch_probability,
        )

        _torch_native_fn = dispatch_probability
    return _torch_native_fn(topk_ids, log2phy_prob, log2phy_map, random_vals)


@register_operator("moe.lplb.dispatch_probability")
def dispatch_probability(
    topk_ids, log2phy_prob, log2phy_map, random_vals=None, *, impl="auto"
):
    """Sample one physical expert per ``(token, slot)`` by inverse CDF.

    The default impl="auto" selects Triton. Rows of ``log2phy_prob`` that sum
    to zero fall back to a uniform choice over the valid replicas
    (``log2phy_map != -1``).

    Args:
        topk_ids (torch.Tensor): Integer int32/int64 logical expert ids of any
            shape; non-contiguous inputs are materialised.
        log2phy_prob (torch.Tensor): Contiguous float32 [num_logical,
            max_copies] LP probabilities.
        log2phy_map (torch.Tensor): Contiguous int32/int64 [num_logical,
            max_copies] physical ids; ``-1`` marks an unused replica.
        random_vals (torch.Tensor | None): Optional float32 [numel(topk_ids)]
            uniform samples in [0, 1). Fresh values are drawn when omitted.
        impl (str): Implementation name: "auto", "triton", or "torch_native".
            Defaults to "auto".

    Returns:
        torch.Tensor: Physical ids with ``topk_ids``' shape and dtype.
    """
    return get_impl_by_name("moe.lplb.dispatch_probability", impl)(
        topk_ids, log2phy_prob, log2phy_map, random_vals
    )

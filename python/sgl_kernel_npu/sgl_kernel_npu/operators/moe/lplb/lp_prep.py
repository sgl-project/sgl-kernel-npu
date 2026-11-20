"""Public LPLB LP-input builder."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("moe.lplb.lp_prep", "auto")
@register_impl("moe.lplb.lp_prep", "triton")
def _triton_impl(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.moe.lplb.lp_prep import lp_prep

        _triton_fn = lp_prep
    return _triton_fn(
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
    )


@register_impl("moe.lplb.lp_prep", "torch_native")
def _torch_native_impl(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.moe.lplb.lp_prep import lp_prep

        _torch_native_fn = lp_prep
    return _torch_native_fn(
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
    )


@register_operator("moe.lplb.lp_prep")
def lp_prep(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
    *,
    impl="auto",
):
    """Build the LPLB interior-point solver inputs in place.

    The default impl="auto" selects Triton. ``A_full[:, -1]``, ``b`` and ``t1``
    are overwritten; every other input is read-only.

    Args:
        A_full (torch.Tensor): Contiguous float32 [NC, NV]; only the last
            column is written. NC must equal ``len(log_replicated) +
            B1.shape[0]``.
        b (torch.Tensor): Contiguous float32 [NC] RHS buffer, overwritten.
        t1 (torch.Tensor): Contiguous float32 [num_single] share buffer,
            overwritten.
        global_counts (torch.Tensor): Contiguous float32 [num_logical] token
            counts per logical expert.
        log_single (torch.Tensor): Contiguous int32/int64 [num_single] logical
            ids that own exactly one physical copy.
        log_replicated (torch.Tensor): Contiguous int32/int64 [num_red_log]
            logical ids that own several physical copies.
        B1 (torch.Tensor): Contiguous float32 [num_gpus, num_single] GPU
            assignment matrix of the single-copy experts.
        A_base_row_sum (torch.Tensor): Contiguous float32 [NC] row sums of the
            constant part of the constraint matrix.
        impl (str): Implementation name: "auto", "triton", or "torch_native".
            Defaults to "auto".

    Returns:
        tuple: ``(A_full, b, t1)`` after the in-place update.
    """
    return get_impl_by_name("moe.lplb.lp_prep", impl)(
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
    )

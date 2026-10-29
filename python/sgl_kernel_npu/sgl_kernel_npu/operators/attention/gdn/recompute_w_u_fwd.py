"""Public GDN WY recomputation with lazy implementation dispatch."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.recompute_w_u_fwd", "auto")
@register_impl("attention.gdn.recompute_w_u_fwd", "triton")
def _triton_impl(k, v, beta, g_cumsum, A, cu_seqlens):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.recompute_w_u_fwd import (
            recompute_w_u_fwd,
        )

        _triton_fn = recompute_w_u_fwd
    return _triton_fn(k, v, beta, g_cumsum, A, cu_seqlens)


@register_impl("attention.gdn.recompute_w_u_fwd", "torch_native")
def _torch_native_impl(k, v, beta, g_cumsum, A, cu_seqlens):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.recompute_w_u_fwd import (
            recompute_w_u_fwd,
        )

        _torch_native_fn = recompute_w_u_fwd
    return _torch_native_fn(k, v, beta, g_cumsum, A, cu_seqlens)


@register_operator("attention.gdn.recompute_w_u_fwd")
def recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None, *, impl="auto"):
    """Return (w, u) for each GDN sequence chunk; auto selects Triton.

    Args:
        k (torch.Tensor): Keys [B, T, Hg, K], FP16/BF16/FP32.
        v (torch.Tensor): Values [B, T, H, V], FP16/BF16/FP32; H % Hg == 0.
        beta (torch.Tensor): Update weights [B, T, H], FP16/BF16/FP32.
        g_cumsum (torch.Tensor): Chunk-local cumulative log decay [B, T, H].
        A (torch.Tensor): Inverse matrices from solve_tril, [B, T, H, C].
        cu_seqlens (torch.Tensor | None): Int32/int64 boundaries [N+1]; B=1
            for packed input, positive lengths, first=0 and last=T.
        impl (str): auto (default), triton or torch_native.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: w [B,T,H,K] in k.dtype and
            u [B,T,H,V] in v.dtype, on k's device. Inputs are unchanged.
    """
    impl_fn = get_impl_by_name("attention.gdn.recompute_w_u_fwd", impl)
    return impl_fn(k, v, beta, g_cumsum, A, cu_seqlens)

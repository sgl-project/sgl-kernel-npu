"""Public LPLB interior-point LP solver."""

from ...utils.cann import detect_cann_version
from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_triton_cann920_fn = None
_torch_native_fn = None


@register_impl("moe.lplb.solve_ipm", "auto")
def _auto_impl(A, b, c, num_iters=5, result=None):
    cann_version = detect_cann_version()
    backend = "triton_cann920" if cann_version.startswith("9.2.") else "triton"
    return get_impl_by_name("moe.lplb.solve_ipm", backend)(A, b, c, num_iters, result)


@register_impl("moe.lplb.solve_ipm", "triton")
def _triton_impl(A, b, c, num_iters=5, result=None):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.moe.lplb.solve_ipm import solve_ipm

        _triton_fn = solve_ipm
    return _triton_fn(A, b, c, num_iters, result)


@register_impl("moe.lplb.solve_ipm", "triton_cann920")
def _triton_cann920_impl(A, b, c, num_iters=5, result=None):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    global _triton_cann920_fn
    if _triton_cann920_fn is None:
        from ....implementation.triton.moe.lplb.solve_ipm import solve_ipm_cann92

        _triton_cann920_fn = solve_ipm_cann92
    return _triton_cann920_fn(A, b, c, num_iters, result)


@register_impl("moe.lplb.solve_ipm", "torch_native")
def _torch_native_impl(A, b, c, num_iters=5, result=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.moe.lplb.solve_ipm import solve_ipm

        _torch_native_fn = solve_ipm
    return _torch_native_fn(A, b, c, num_iters, result)


@register_operator("moe.lplb.solve_ipm")
def solve_ipm(A, b, c, num_iters=5, result=None, *, impl="auto"):
    """Solve ``min c^T x`` subject to ``A x = b, x >= 0`` with a barrier method.

    The default impl="auto" selects triton_cann920 on CANN 9.2.x and triton on
    other releases. The single-program kernel matches the torch reference to
    float32 tolerance; it is not bit-exact because the KKT system is factored
    differently.

    Args:
        A (torch.Tensor): Contiguous float32 [NC, NV] constraint matrix.
        b (torch.Tensor): Contiguous float32 [NC] right-hand side.
        c (torch.Tensor): Contiguous float32 [NV] objective coefficients.
        num_iters (int): Number of barrier iterations. Defaults to 5.
        result (torch.Tensor | None): Optional contiguous float32 [NV] output
            buffer, written and returned. A fresh tensor is allocated when
            omitted.
        impl (str): Implementation name: "auto", "triton", "triton_cann920",
            or "torch_native". Defaults to "auto".

    Returns:
        torch.Tensor: Contiguous float32 [NV] solution. Every entry is 0.5 when
        the convergence test fails.
    """
    return get_impl_by_name("moe.lplb.solve_ipm", impl)(A, b, c, num_iters, result)

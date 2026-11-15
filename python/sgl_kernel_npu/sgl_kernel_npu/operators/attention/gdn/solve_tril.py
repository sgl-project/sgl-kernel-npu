"""Public chunk-local triangular inverse with lazy backend dispatch."""

import torch

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.solve_tril", "auto")
@register_impl("attention.gdn.solve_tril", "triton")
def _triton_impl(A, cu_seqlens, output_dtype):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.solve_tril import solve_tril

        _triton_fn = solve_tril
    return _triton_fn(A, cu_seqlens, output_dtype)


@register_impl("attention.gdn.solve_tril", "torch_native")
def _torch_native_impl(A, cu_seqlens, output_dtype):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.solve_tril import solve_tril

        _torch_native_fn = solve_tril
    return _torch_native_fn(A, cu_seqlens, output_dtype)


@register_operator("attention.gdn.solve_tril")
def solve_tril(A, cu_seqlens=None, output_dtype=torch.float, *, impl="auto"):
    """Compute (I + A)^-1 independently for each sequence chunk and head.

    The default impl="auto" selects Triton; "torch_native" is independent.

    Args:
        A (torch.Tensor): Nonempty strictly lower triangular chunk matrices
            packed as [B, T, H, C], C > 0; FP16/BF16/FP32. Triton requires
            contiguous input and C in {16, 32, 64}.
        cu_seqlens (torch.Tensor | None): Int32/int64 sequence boundaries [N+1],
            starting at 0 and ending at T. Triton requires contiguous
            boundaries on A's device; torch_native also accepts CPU boundaries.
            Lengths must be positive; packed input requires B=1.
        output_dtype (torch.dtype): FP16/BF16/FP32, defaults to FP32.
        impl (str): "auto", "triton", or "torch_native".

    Returns:
        torch.Tensor: Inverse blocks with A's shape and device, output_dtype.
            Inputs are not modified. Chunks restart at each sequence boundary.
    """
    impl_fn = get_impl_by_name("attention.gdn.solve_tril", impl)
    return impl_fn(A, cu_seqlens, output_dtype)

"""Public phases of a tensor-parallel RMSNorm.

Both operators are halves of one distributed RMSNorm: variance is reduced
locally, averaged across tensor-parallel ranks by the caller, then applied.
"""

from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_variance_triton_fn = None
_variance_torch_native_fn = None
_rsqrt_mul_triton_fn = None
_rsqrt_mul_torch_native_fn = None


# Importing this module registers these four functions, without importing Triton.
@register_impl("norm.fused_variance", "auto")
@register_impl("norm.fused_variance", "triton")
def _variance_triton_impl(x):
    global _variance_triton_fn
    if _variance_triton_fn is None:
        from sgl_kernel_npu.implementation.triton.norm.rmsnorm_split import (
            fused_variance,
        )

        _variance_triton_fn = fused_variance

    return _variance_triton_fn(x)


@register_impl("norm.fused_variance", "torch_native")
def _variance_torch_native_impl(x):
    global _variance_torch_native_fn
    if _variance_torch_native_fn is None:
        from sgl_kernel_npu.implementation.torch_native.norm.rmsnorm_split import (
            fused_variance,
        )

        _variance_torch_native_fn = fused_variance

    return _variance_torch_native_fn(x)


@register_operator("norm.fused_variance")
def fused_variance(x, *, impl="auto"):
    """The default impl="auto" selects Triton.

    Compute the local mean square over the last dimension. For an input of
    shape [B, L, C] the result has shape [B, L, 1]: it is the pre-reduction
    variance of a tensor-parallel RMSNorm and must be all-reduced by the
    caller before fused_rsqrt_mul is applied.

    Args:
        x (torch.Tensor): Input of shape [B, L, C]; the mean square runs over
            C. Triton requires a contiguous three-dimensional NPU tensor whose
            C is a power of two.
        impl (str): Implementation name: "auto", "triton", or
            "torch_native". Defaults to "auto".

    Returns:
        torch.Tensor: Variance of shape [B, L, 1] with the same dtype and
        device as x.
    """
    impl_fn = get_impl_by_name("norm.fused_variance", impl)
    return impl_fn(x)


@register_impl("norm.fused_rsqrt_mul", "auto")
@register_impl("norm.fused_rsqrt_mul", "triton")
def _rsqrt_mul_triton_impl(x, variance, weight, eps):
    global _rsqrt_mul_triton_fn
    if _rsqrt_mul_triton_fn is None:
        from sgl_kernel_npu.implementation.triton.norm.rmsnorm_split import (
            fused_rsqrt_mul,
        )

        _rsqrt_mul_triton_fn = fused_rsqrt_mul

    return _rsqrt_mul_triton_fn(x, variance, weight, eps)


@register_impl("norm.fused_rsqrt_mul", "torch_native")
def _rsqrt_mul_torch_native_impl(x, variance, weight, eps):
    global _rsqrt_mul_torch_native_fn
    if _rsqrt_mul_torch_native_fn is None:
        from sgl_kernel_npu.implementation.torch_native.norm.rmsnorm_split import (
            fused_rsqrt_mul,
        )

        _rsqrt_mul_torch_native_fn = fused_rsqrt_mul

    return _rsqrt_mul_torch_native_fn(x, variance, weight, eps)


@register_operator("norm.fused_rsqrt_mul")
def fused_rsqrt_mul(x, variance, weight, eps=1e-6, *, impl="auto"):
    """The default impl="auto" selects Triton.

    Apply the normalization scale and weight of a tensor-parallel RMSNorm to
    x. The variance must already contain the tensor-parallel average; it is
    usually the all-reduced output of fused_variance.

    Args:
        x (torch.Tensor): Input of shape [B, L, C]; normalization runs over C.
            Triton requires a contiguous three-dimensional NPU tensor.
        variance (torch.Tensor): Variance of shape [B, L, 1] or [B*L], on the
            same device and dtype as x.
        weight (torch.Tensor): Multiplicative weight of shape [C].
        eps (float): Stability term added to the variance. Defaults to 1e-6.
        impl (str): Implementation name: "auto", "triton", or
            "torch_native". Defaults to "auto".

    Returns:
        torch.Tensor: Output of shape [B, L, C] with the same dtype and device
        as x.
    """
    impl_fn = get_impl_by_name("norm.fused_rsqrt_mul", impl)
    return impl_fn(x, variance, weight, eps)

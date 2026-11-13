"""Public LayerNorm dispatch for the CANN built-in ACLNN operator."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("norm.layer_norm", "auto", supported_generations={Generation.ARCH22})
@register_impl(
    "norm.layer_norm",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH22},
)
def _cann_ops_transformer_impl(
    input,
    normalized_shape,
    weight,
    bias,
    eps,
    impl_mode,
):
    import torch

    return torch.ops.npu.layernorm(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
        impl_mode=impl_mode,
    )


@register_impl(
    "norm.layer_norm", "torch_native", supported_generations={Generation.ARCH22}
)
def _torch_native_impl(
    input,
    normalized_shape,
    weight,
    bias,
    eps,
    impl_mode,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.layer_norm import layer_norm

        _torch_native_fn = layer_norm
    return _torch_native_fn(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
        impl_mode=impl_mode,
    )


@register_operator("norm.layer_norm")
def layer_norm(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
    impl_mode=0,
    *,
    impl="auto",
):
    """Normalize the trailing dimensions of ``input`` by mean and variance.

    ``normalized_shape`` lists the trailing dimensions to normalize; ``weight``
    and ``bias`` are optional affine parameters whose dtype may differ from
    ``input`` (the operator accepts input dtype or FP32). Return
    ``(out, mean, rstd)``. ``mean``/``rstd`` use the non-normalized axes plus
    trailing ones, with the input dtype and NPU format, as in the legacy
    binding. ``impl_mode`` selects the CANN computation mode and is ignored by
    the torch_native reference. ``impl`` selects auto (cann_ops_transformer),
    cann_ops_transformer, or torch_native.
    """
    return get_impl_by_name("norm.layer_norm", impl)(
        input,
        normalized_shape,
        weight,
        bias,
        eps,
        impl_mode,
    )


# Legacy name; the call contract is unchanged.
layernorm = layer_norm

"""Public A5 dynamic MXFP8 quantization API."""

import torch
import torch_npu
from sgl_kernel_npu.operators.utils.generation import Generation
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_IMPL_NAME = "activation.dynamic_mx_quant"
_ONLY_A5 = {Generation.ARCH35}
_MX_BLOCK = 32
_torch_native_fn = None


@register_impl(_IMPL_NAME, "auto", supported_generations=_ONLY_A5)
@register_impl(_IMPL_NAME, "torch_npu", supported_generations=_ONLY_A5)
def _torch_npu_impl(x, dst_type):
    payload, scales = torch_npu.npu_dynamic_mx_quant(x, dst_type=dst_type, axis=-1)
    # Normalize the vendor scale layout to one E8M0 byte per 32-element group.
    scales = scales.reshape(*x.shape[:-1], x.shape[-1] // _MX_BLOCK).contiguous()
    return payload, scales.view(torch.float8_e8m0fnu)


@register_impl(_IMPL_NAME, "torch_native", supported_generations=_ONLY_A5)
def _torch_native_impl(x, dst_type):
    global _torch_native_fn
    if _torch_native_fn is None:
        from sgl_kernel_npu.implementation.torch_native.activation.dynamic_mx_quant import (
            dynamic_mx_quant as reference,
        )

        _torch_native_fn = reference
    return _torch_native_fn(x, dst_type)


@register_operator(_IMPL_NAME)
def dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, *, impl="auto"):
    """Quantize a tensor to MXFP8 with one E8M0 scale per 32 elements.

    Wraps the vendor ``torch_npu.npu_dynamic_mx_quant`` with ``axis=-1``
    and a fixed block size of 32, for example on the GLM-5.3-Flash DSA indexer
    query tensors reshaped to ``[-1, 128]``.

    Args:
        x: Contiguous bf16/fp16/fp32 tensor on an Ascend NPU with at least one
            dimension; the last dimension must be a positive multiple of 32.
        dst_type: Payload dtype; only ``torch.float8_e4m3fn`` is supported.
        impl: ``auto``/``torch_npu`` select the vendor torch_npu operator;
            ``torch_native`` selects the pure PyTorch reference. This operator
            only supports arch35 (Ascend950).

    Returns:
        ``(payload, scales)``. ``payload`` has the same shape as ``x`` in
        ``torch.float8_e4m3fn``; ``scales`` has shape
        ``x.shape[:-1] + (x.shape[-1] // 32,)`` in ``torch.float8_e8m0fnu``,
        one power-of-two scale per 32-element group along the last dimension.
    """
    if x.ndim < 1:
        raise ValueError("dynamic_mx_quant expects a tensor with ndim >= 1")
    last_dim = x.shape[-1]
    if last_dim <= 0 or last_dim % _MX_BLOCK != 0:
        raise ValueError(
            "dynamic_mx_quant expects the last dimension to be a positive "
            f"multiple of {_MX_BLOCK}, got {tuple(x.shape)}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"x must be bfloat16, float16 or float32, got {x.dtype}")
    if not x.is_contiguous():
        raise RuntimeError("x must be contiguous")
    if dst_type != torch.float8_e4m3fn:
        raise ValueError(f"dst_type must be torch.float8_e4m3fn, got {dst_type}")
    return get_impl_by_name(_IMPL_NAME, impl)(x, dst_type)

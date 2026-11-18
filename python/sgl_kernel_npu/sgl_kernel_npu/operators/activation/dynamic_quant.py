"""Public per-row dynamic INT8 quantization API."""

import torch
import torch_npu
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_IMPL_NAME = "activation.dynamic_quant"
_torch_native_fn = None


@register_impl(_IMPL_NAME, "auto")
@register_impl(_IMPL_NAME, "torch_npu")
def _torch_npu_impl(x, dst_type):
    rows = x.reshape(-1, x.shape[-1])
    quantized, scale = torch_npu.npu_dynamic_quant(rows, dst_type=dst_type)
    return quantized.reshape(x.shape), scale.reshape(x.shape[:-1]).to(torch.float32)


@register_impl(_IMPL_NAME, "torch_native")
def _torch_native_impl(x, dst_type):
    global _torch_native_fn
    if _torch_native_fn is None:
        from sgl_kernel_npu.implementation.torch_native.activation.dynamic_quant import (
            dynamic_quant as reference,
        )

        _torch_native_fn = reference
    return _torch_native_fn(x, dst_type)


@register_operator(_IMPL_NAME)
def dynamic_quant(x, dst_type=torch.int8, *, impl="auto"):
    """Quantize a tensor to INT8 with one FP32 scale per row (last dimension).

    Wraps the vendor ``torch_npu.npu_dynamic_quant`` (symmetric per-row
    quantization, scale = amax / 127), for example on the GLM-5.3-Flash DSA
    indexer query tensors reshaped to ``[-1, 128]``.

    Args:
        x: Contiguous bf16/fp16/fp32 tensor on an Ascend NPU with at least one
            dimension and a positive last dimension.
        dst_type: Payload dtype; only ``torch.int8`` is supported.
        impl: ``auto``/``torch_npu`` select the vendor torch_npu operator;
            ``torch_native`` selects the pure PyTorch reference. Available on
            arch22 (Ascend910) and arch35 (Ascend950).

    Returns:
        ``(payload, scale)``. ``payload`` has the same shape as ``x`` in
        ``torch.int8``; ``scale`` has shape ``x.shape[:-1]`` in
        ``torch.float32``, one dequantization scale per row along the last
        dimension. An all-zero row keeps scale ``0``.
    """
    if x.ndim < 1:
        raise ValueError("dynamic_quant expects a tensor with ndim >= 1")
    if x.shape[-1] <= 0:
        raise ValueError(
            f"dynamic_quant expects a positive last dimension, got {tuple(x.shape)}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"x must be bfloat16, float16 or float32, got {x.dtype}")
    if not x.is_contiguous():
        raise RuntimeError("x must be contiguous")
    if dst_type != torch.int8:
        raise ValueError(f"dst_type must be torch.int8, got {dst_type}")
    return get_impl_by_name(_IMPL_NAME, impl)(x, dst_type)

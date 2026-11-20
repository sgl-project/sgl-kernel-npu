"""TorchNPU quantized KV sparse flash attention adapter."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "attention.kv_quant_sparse_flash_attention_torch_npu"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-transformer")
def _torch_npu_impl(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    **kwargs,
):
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        **kwargs,
    )


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.attention.kv_quant_sparse_flash_attention_torch_npu import (
            kv_quant_sparse_flash_attention_torch_npu,
        )

        _native_impl = kv_quant_sparse_flash_attention_torch_npu
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def kv_quant_sparse_flash_attention_torch_npu(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    impl="auto",
    **kwargs,
):
    """Return TorchNPU quantized sparse attention for separate key/value inputs.

    query and sparse_indices describe the query rows and their selected KV
    positions. The quant modes describe the key/value encoding. Optional SDK
    keywords include dequant scales, page and length metadata, and layouts.
    auto selects ops-transformer. Its dtype and layout limits follow the installed SDK.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        **kwargs,
    )

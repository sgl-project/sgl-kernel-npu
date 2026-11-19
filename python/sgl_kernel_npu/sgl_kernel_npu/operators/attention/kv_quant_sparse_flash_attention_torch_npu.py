"""TorchNPU quantized KV sparse flash attention adapter."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_npu_fn = None
_OP_NAME = "attention.kv_quant_sparse_flash_attention_torch_npu"


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
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
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_kv_quant_sparse_flash_attention

        _torch_npu_fn = npu_kv_quant_sparse_flash_attention
    return _torch_npu_fn(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        **kwargs,
    )


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
    auto selects torch_npu. Its dtype and layout limits follow the installed SDK.
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

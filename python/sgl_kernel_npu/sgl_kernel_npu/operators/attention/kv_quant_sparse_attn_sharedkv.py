"""Public A5 quantized shared-KV attention APIs.

The native bindings stay under torch.ops.npu for existing SGLang callers.
The torch_native implementation is an independent, test-only reference for
the BF16 BSND paged sliding-window path without compressed KV.
"""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_ATTN_OP = "attention.kv_quant_sparse_attn_sharedkv"
_META_OP = "attention.kv_quant_sparse_attn_sharedkv_metadata_host"
_reference = None


@register_impl(_ATTN_OP, "auto", supported_generations={Generation.ARCH35})
@register_impl(_ATTN_OP, "ascendc", supported_generations={Generation.ARCH35})
def _ascendc_attention(q, **kwargs):
    import torch

    return torch.ops.npu.kv_quant_sparse_attn_sharedkv(q, **kwargs)


@register_impl(
    _ATTN_OP, "torch_native", supported_generations={Generation.ARCH35}
)
def _torch_native_attention(q, **kwargs):
    global _reference
    if _reference is None:
        from ...implementation.torch_native.attention.kv_quant_sparse_attn_sharedkv import (
            kv_quant_sparse_attn_sharedkv as reference,
        )

        _reference = reference
    return _reference(q, **kwargs)


@register_operator(_ATTN_OP)
def kv_quant_sparse_attn_sharedkv(q, *, impl="auto", **kwargs):
    """Compute A5 shared-KV sparse attention; return (output, empty LSE).

    ``auto`` and ``ascendc`` use the existing native binding. The explicit
    ``torch_native`` reference covers only BF16 BSND paged SWA and is intended
    for accuracy checks, not model execution.
    """
    return get_impl_by_name(_ATTN_OP, impl)(q, **kwargs)


@register_impl(_META_OP, "auto", supported_generations={Generation.ARCH35})
@register_impl(_META_OP, "host", supported_generations={Generation.ARCH35})
def _host_metadata(*args, **kwargs):
    import torch

    return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(*args, **kwargs)


@register_operator(_META_OP)
def kv_quant_sparse_attn_sharedkv_metadata_host(*args, impl="auto", **kwargs):
    """Schedule A5 attention from CPU int32 lengths and return NPU metadata.

    This host scheduler remains in the legacy Torch extension: PR #880 only
    supports device-side CANN operators in its new declaration path.
    """
    return get_impl_by_name(_META_OP, impl)(*args, **kwargs)

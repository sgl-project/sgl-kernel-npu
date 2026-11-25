"""TorchNPU reduction of partitioned attention outputs."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_native_impl = None


@register_impl("attention.attention_update", "auto")
@register_impl("attention.attention_update", "ops-transformer")
def _torch_npu_impl(lse, local_out, update_type):
    return torch_npu.npu_attention_update(lse, local_out, update_type)


@register_impl("attention.attention_update", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.attention.attention_update import (
            attention_update,
        )

        _native_impl = attention_update
    return _native_impl(*args, **kwargs)


@register_operator("attention.attention_update")
def attention_update(lse, local_out, update_type, *, impl="auto"):
    """Merge partial attention output and log-sum-exp (LSE) across partitions.

    lse and local_out contain one tensor per partition, shaped [N] and
    [N, head_dim], respectively. update_type is the TorchNPU reduction mode.
    Returns the merged output and the SDK's LSE result. auto selects ops-transformer.
    """
    return get_impl_by_name("attention.attention_update", impl)(
        lse, local_out, update_type
    )

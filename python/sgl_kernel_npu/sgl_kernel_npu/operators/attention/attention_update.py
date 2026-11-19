"""TorchNPU reduction of partitioned attention outputs."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_npu_fn = None


@register_impl("attention.attention_update", "auto")
@register_impl("attention.attention_update", "torch_npu")
def _torch_npu_impl(lse, local_out, update_type):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_attention_update

        _torch_npu_fn = npu_attention_update
    return _torch_npu_fn(lse, local_out, update_type)


@register_operator("attention.attention_update")
def attention_update(lse, local_out, update_type, *, impl="auto"):
    """Merge partial attention output and log-sum-exp (LSE) across partitions.

    lse and local_out contain one tensor per partition, shaped [N] and
    [N, head_dim], respectively. update_type is the TorchNPU reduction mode.
    Returns the merged output and the SDK's LSE result. auto selects torch_npu.
    """
    return get_impl_by_name("attention.attention_update", impl)(
        lse, local_out, update_type
    )

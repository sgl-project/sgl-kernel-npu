"""Public sparse_flash_mla operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._sharedkv_compat import compute


@register_impl("attention.sparse_flash_mla", "auto")
@register_impl("attention.sparse_flash_mla", "cann_ops_transformer")
def _cann_sparse_flash_mla(q, **kwargs):
    return compute(False, q, kwargs)


@register_impl("attention.sparse_flash_mla", "torch_native")
def _torch_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.attention.sparse_flash_mla import (
        sparse_flash_mla,
    )

    return sparse_flash_mla(q, **kwargs)


@register_operator("attention.sparse_flash_mla")
def sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute BF16 shared-KV sparse attention; return output and LSE.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.sparse_flash_mla", impl)(q, **kwargs)

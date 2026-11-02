"""Public sparse_flash_mla_metadata operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.sparse_flash_mla_metadata", "auto")
@register_impl("attention.sparse_flash_mla_metadata", "cann_ops_transformer")
def _cann_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.cann_ops_transformer.attention.sparse_flash_mla_metadata import (
        sparse_flash_mla_metadata,
    )

    return sparse_flash_mla_metadata(**kwargs)


@register_impl("attention.sparse_flash_mla_metadata", "torch_native")
def _torch_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.attention.sparse_flash_mla_metadata import (
        sparse_flash_mla_metadata,
    )

    return sparse_flash_mla_metadata(**kwargs)


@register_operator("attention.sparse_flash_mla_metadata")
def sparse_flash_mla_metadata(*, impl="auto", **kwargs):
    """Build BF16 attention metadata.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.sparse_flash_mla_metadata", impl)(**kwargs)

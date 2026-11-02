"""Public mixed_quant_sparse_flash_mla_metadata operator with registered implementations."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.cann_ops_transformer.attention.mixed_quant_sparse_flash_mla_metadata import (
        mixed_quant_sparse_flash_mla_metadata,
    )

    return mixed_quant_sparse_flash_mla_metadata(**kwargs)


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.attention.mixed_quant_sparse_flash_mla_metadata import (
        mixed_quant_sparse_flash_mla_metadata,
    )

    return mixed_quant_sparse_flash_mla_metadata(**kwargs)


@register_operator("attention.mixed_quant_sparse_flash_mla_metadata")
def mixed_quant_sparse_flash_mla_metadata(*, impl="auto", **kwargs):
    """Build A5 mixed-quant attention metadata.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla_metadata", impl)(
        **kwargs
    )

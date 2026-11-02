"""Public mixed_quant_sparse_flash_mla operator with registered implementations."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla(q, **kwargs):
    from ...implementation.cann_ops_transformer.attention.mixed_quant_sparse_flash_mla import (
        mixed_quant_sparse_flash_mla,
    )

    return mixed_quant_sparse_flash_mla(q, **kwargs)


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.attention.mixed_quant_sparse_flash_mla import (
        mixed_quant_sparse_flash_mla,
    )

    return mixed_quant_sparse_flash_mla(q, **kwargs)


@register_operator("attention.mixed_quant_sparse_flash_mla")
def mixed_quant_sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute A5 mixed-quant sparse attention; return output and LSE.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla", impl)(q, **kwargs)

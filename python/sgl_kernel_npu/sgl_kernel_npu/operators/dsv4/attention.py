"""DSV4 attention operators with explicit implementation selection."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("dsv4.sparse_flash_mla_metadata", "auto")
@register_impl("dsv4.sparse_flash_mla_metadata", "cann_ops_transformer")
def _cann_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import sparse_flash_mla_metadata

    return sparse_flash_mla_metadata(**kwargs)


@register_impl("dsv4.sparse_flash_mla_metadata", "torch_native")
def _torch_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.dsv4.attention import sparse_flash_mla_metadata

    return sparse_flash_mla_metadata(**kwargs)


@register_operator("dsv4.sparse_flash_mla_metadata")
def sparse_flash_mla_metadata(*, impl="auto", **kwargs):
    """Build BF16 attention metadata.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.sparse_flash_mla_metadata", impl)(**kwargs)


@register_impl("dsv4.sparse_flash_mla", "auto")
@register_impl("dsv4.sparse_flash_mla", "cann_ops_transformer")
def _cann_sparse_flash_mla(q, **kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import sparse_flash_mla

    return sparse_flash_mla(q, **kwargs)


@register_impl("dsv4.sparse_flash_mla", "torch_native")
def _torch_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.dsv4.attention import sparse_flash_mla

    return sparse_flash_mla(q, **kwargs)


@register_operator("dsv4.sparse_flash_mla")
def sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute BF16 shared-KV sparse attention; return output and LSE.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.sparse_flash_mla", impl)(q, **kwargs)


@register_impl(
    "dsv4.mixed_quant_sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "dsv4.mixed_quant_sparse_flash_mla_metadata",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import (
        mixed_quant_sparse_flash_mla_metadata,
    )

    return mixed_quant_sparse_flash_mla_metadata(**kwargs)


@register_impl(
    "dsv4.mixed_quant_sparse_flash_mla_metadata",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.dsv4.attention import (
        mixed_quant_sparse_flash_mla_metadata,
    )

    return mixed_quant_sparse_flash_mla_metadata(**kwargs)


@register_operator("dsv4.mixed_quant_sparse_flash_mla_metadata")
def mixed_quant_sparse_flash_mla_metadata(*, impl="auto", **kwargs):
    """Build A5 mixed-quant attention metadata.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.mixed_quant_sparse_flash_mla_metadata", impl)(
        **kwargs
    )


@register_impl(
    "dsv4.mixed_quant_sparse_flash_mla",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "dsv4.mixed_quant_sparse_flash_mla",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla(q, **kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import mixed_quant_sparse_flash_mla

    return mixed_quant_sparse_flash_mla(q, **kwargs)


@register_impl(
    "dsv4.mixed_quant_sparse_flash_mla",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.dsv4.attention import (
        mixed_quant_sparse_flash_mla,
    )

    return mixed_quant_sparse_flash_mla(q, **kwargs)


@register_operator("dsv4.mixed_quant_sparse_flash_mla")
def mixed_quant_sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute A5 mixed-quant sparse attention; return output and LSE.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.mixed_quant_sparse_flash_mla", impl)(q, **kwargs)

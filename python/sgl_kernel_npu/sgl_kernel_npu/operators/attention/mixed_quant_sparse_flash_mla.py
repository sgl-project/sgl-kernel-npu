"""Public mixed_quant_sparse_flash_mla operator with registered implementations."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._sharedkv_compat import compute


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
    return compute(True, q, kwargs)


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
def mixed_quant_sparse_flash_mla(
    q,
    *,
    quant_mode=1,
    rope_head_dim=64,
    cmp_ratio=1,
    softmax_scale=1.0,
    impl="auto",
    **kwargs,
):
    """Compute A5 mixed-quant sparse attention; return output and LSE.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla", impl)(
        q,
        quant_mode=quant_mode,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        softmax_scale=softmax_scale,
        **kwargs,
    )

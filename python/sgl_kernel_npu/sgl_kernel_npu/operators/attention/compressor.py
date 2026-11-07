"""Public compressor operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.compressor", "auto")
def _auto_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = (
            "ops_transformer_cann910"
            if get_hardware().generation == Generation.ARCH35
            else "ascendc"
        )
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformer_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("attention.compressor", backend)(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )


@register_impl("attention.compressor", "cann_ops_transformer_cann920")
def _cann_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    # Preserve the repository's fused RMSNorm/RoPE and state-cache contract.
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    op = getattr(torch.ops.npu, "compressor", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator npu.compressor in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    return op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl(
    "attention.compressor",
    "ops_transformer_cann910",
    supported_generations={Generation.ARCH35},
)
def _ops_transformer_cann910(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(torch.ops.sgl_cann_recipes_infer, "compressor", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.compressor in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    return op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl(
    "attention.compressor", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    op = getattr(torch.ops.npu, "compressor", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator npu.compressor in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    return op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_impl("attention.compressor", "torch_native")
def _torch_compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    from ...implementation.torch_native.attention.compressor import compressor

    return compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)


@register_operator("attention.compressor")
def compressor(x, wkv, wgate, state_cache, ape, norm_weight, *, impl="auto", **kwargs):
    """Compress KV and update the compressor state cache in place.

    impl: auto (SDK-selected), ops_transformer_cann910 (A5), ascendc (A3), cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.compressor", impl)(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )

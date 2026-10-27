"""Public in-place bitmask operator with explicit implementation selection."""

import json
from functools import lru_cache
from pathlib import Path

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_GENERATIONS = (Generation.ARCH22, Generation.ARCH35)


@lru_cache(maxsize=1)
def _operator():
    import torch

    manifest_path = Path(__file__).resolve().parents[2] / "_build_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if "apply_token_bitmask" not in manifest["operators"]:
            raise RuntimeError(
                "This wheel does not include ascendc apply_token_bitmask; rebuild with --ops apply_token_bitmask or --ops all"
            )
        if int(manifest["arch_version"]) != get_hardware().arch_id:
            raise RuntimeError(
                "The wheel ARCH does not match this NPU; install a wheel built for its --arch-version"
            )
    try:
        return torch.ops.npu.apply_token_bitmask
    except AttributeError as error:
        raise RuntimeError(
            "The installed native library does not register npu::apply_token_bitmask"
        ) from error


@register_impl(
    "sampler.apply_token_bitmask", "auto", supported_generations=_GENERATIONS
)
@register_impl(
    "sampler.apply_token_bitmask", "ascendc", supported_generations=_GENERATIONS
)
def _ascendc_impl(logits, bitmask, indices):
    return _operator()(logits, bitmask, indices)


@register_impl(
    "sampler.apply_token_bitmask", "torch_native", supported_generations=_GENERATIONS
)
def _native_impl(logits, bitmask, indices):
    from ...implementation.torch_native.sampler.apply_token_bitmask import (
        apply_token_bitmask,
    )

    return apply_token_bitmask(logits, bitmask, indices)


@register_operator("sampler.apply_token_bitmask")
def apply_token_bitmask(logits, bitmask, indices=None, *, impl="auto"):
    """Mask logits in place, returning logits. auto selects ascendc without fallback.

    logits is contiguous [batch, vocabulary] float32/float16/bfloat16;
    bitmask is contiguous [batch, words] int32 on the same NPU. Bit zero
    writes -inf; bit one preserves the value. Missing words act as zero.
    None or empty indices selects all rows, preserving the existing contract.
    """
    return get_impl_by_name("sampler.apply_token_bitmask", impl)(
        logits, bitmask, indices
    )

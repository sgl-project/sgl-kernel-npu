"""Public indexer_prologue_k_dsl operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from importlib import import_module

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_cannbot_dsl_fn = None


@register_impl(
    "indexer.indexer_prologue_k_dsl",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.indexer_prologue_k_dsl import (
            indexer_prologue_k_dsl,
        )

        _torch_native_fn = indexer_prologue_k_dsl
    return _torch_native_fn(*args, **kwargs)


@register_impl(
    "indexer.indexer_prologue_k_dsl", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "indexer.indexer_prologue_k_dsl",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(latent, *args, **kwargs):
    global _cannbot_dsl_fn
    with torch.npu.device(latent.device):
        if _cannbot_dsl_fn is None:
            _cannbot_dsl_fn = import_module(
                "sgl_kernel_npu.implementation.cannbot-dsl.indexer.indexer_prologue_k"
            ).indexer_prologue_k
        return _cannbot_dsl_fn(latent, *args, **kwargs)


@register_operator("indexer.indexer_prologue_k_dsl")
def indexer_prologue_k_dsl(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    *,
    cache_index: torch.Tensor,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
    impl="auto",
):
    """Update the key cache and optional scale cache in place.

    ``auto`` selects the lazy A5 CANNBot-DSL backend.

    Args:
        latent: BF16 [T,H], compressor latent before RoPE.
        wk: BF16 [D,H], projection weight; FRACTAL_NZ on the DSL path.
        norm_weight: FP32 [D], RMSNorm weight.
        rope_sin: FP32 [T,Dr], sine coefficients for the last Dr features.
        rope_cos: FP32 [T,Dr], cosine coefficients for the last Dr features.
        k_cache: UINT8 writable cache: mode 0 [P,S,1,D/2]; mode 1 [P,S/G,1,G*(D/2+D/32)].
        k_scale_cache: Optional UINT8 [P,S,1,D/32], separate E8M0 scales for mode 0.
        cache_index: INT64 [T], flattened cache slots; -1 skips a row.
        storage_mode: int: 0 for separate data/scales, 1 for combined storage.
        norm_eps: float, RMSNorm epsilon.
        combined_block_size: int G, rows per combined record; mode 1 requires a positive divisor of S.
        impl: str, auto selects cannbot_dsl; torch_native is an independent compute reference.
    """
    return get_impl_by_name("indexer.indexer_prologue_k_dsl", impl)(
        latent,
        wk,
        norm_weight,
        rope_sin,
        rope_cos,
        k_cache,
        k_scale_cache,
        cache_index=cache_index,
        storage_mode=storage_mode,
        norm_eps=norm_eps,
        combined_block_size=combined_block_size,
    )

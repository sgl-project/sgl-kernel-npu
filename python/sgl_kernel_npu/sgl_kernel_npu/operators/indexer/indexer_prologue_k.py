"""Public indexer_prologue_k operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

import torch

from ..utils.cannbot_dsl import ascend950_device, load_cannbot_dsl
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.indexer_prologue_k", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "indexer.indexer_prologue_k",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    with ascend950_device(*args, **kwargs):
        operator = load_cannbot_dsl("indexer_prologue_k")
        return operator(*args, **kwargs)


@register_operator("indexer.indexer_prologue_k")
def indexer_prologue_k(
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

    ``auto`` and ``cannbot_dsl`` select the same lazy-loaded A5 implementation.
    """
    return get_impl_by_name("indexer.indexer_prologue_k", impl)(
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

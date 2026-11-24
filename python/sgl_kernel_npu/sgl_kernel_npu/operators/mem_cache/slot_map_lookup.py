"""Public sparse KV slot map lookup."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "mem_cache.slot_map_lookup", "auto", supported_generations=[Generation.ARCH22]
)
@register_impl(
    "mem_cache.slot_map_lookup", "ascendc", supported_generations=[Generation.ARCH22]
)
def _ascendc_impl(slot_map, req_indices, topk_indices, block_dim):
    token_on_device = torch.empty_like(topk_indices, dtype=torch.int32)
    device_token_pos = torch.empty_like(topk_indices, dtype=torch.int32)
    torch.ops.npu.slot_map_lookup(
        slot_map,
        req_indices,
        topk_indices,
        token_on_device,
        device_token_pos,
        block_dim,
    )
    return token_on_device, device_token_pos


@register_impl(
    "mem_cache.slot_map_lookup",
    "torch_native",
    supported_generations=[Generation.ARCH22],
)
def _torch_native_impl(slot_map, req_indices, topk_indices, block_dim):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.slot_map_lookup import (
            slot_map_lookup,
        )

        _torch_native_fn = slot_map_lookup
    return _torch_native_fn(slot_map, req_indices, topk_indices, block_dim)


@register_operator("mem_cache.slot_map_lookup")
def slot_map_lookup(
    slot_map: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    block_dim: int = 0,
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve logical top-k KV positions against the device slot map.

    Args:
        slot_map (Tensor): INT32 [slots, context] map, with -1 for missing slots.
        req_indices (Tensor): INT32 [batch] indices of rows in slot_map.
        topk_indices (Tensor): INT32 [batch, 2048] logical KV indices.
        block_dim (int): Number of AI Vector cores. Defaults to 0, which uses
            all available vector cores in the AscendC implementation.
        impl (str): Defaults to "auto", which selects "ascendc".

    Returns:
        Tuple[Tensor, Tensor]: INT32 [batch, 2048] hit flags and slot positions.
        Invalid request or token indices yield a zero flag and position -1.
    """
    return get_impl_by_name("mem_cache.slot_map_lookup", impl)(
        slot_map, req_indices, topk_indices, block_dim
    )

import torch


def slot_map_lookup(slot_map, req_indices, topk_indices, block_dim):
    for tensor, name, ndim in (
        (slot_map, "slot_map", 2),
        (req_indices, "req_indices", 1),
        (topk_indices, "topk_indices", 2),
    ):
        if tensor.device != slot_map.device:
            raise RuntimeError(f"{name} must be on the same device as slot_map")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
        if tensor.ndim != ndim:
            raise RuntimeError(f"{name} must be {ndim}-D")
    for tensor, name in (
        (slot_map, "slot_map"),
        (req_indices, "req_indices"),
        (topk_indices, "topk_indices"),
    ):
        if tensor.dtype != torch.int32:
            raise RuntimeError(f"{name} must be int32")

    slots, context = slot_map.shape
    batch = req_indices.numel()
    if min(slots, context, batch) <= 0:
        raise RuntimeError("slot_map dimensions and batch size must be positive")
    if topk_indices.shape[1] != 2048:
        raise RuntimeError("slot_map_lookup requires topk=2048")
    if context % 8:
        raise RuntimeError(
            "slot_map_lookup requires max_context_len to be a multiple of 8"
        )
    if batch != topk_indices.shape[0]:
        raise RuntimeError("req_indices dim0 must match topk_indices dim0")
    if block_dim < 0 or max(slots * context, topk_indices.numel(), block_dim) >= 2**32:
        raise RuntimeError("slot map geometry exceeds the uint32 range")

    req = req_indices[:, None]
    valid = (req >= 0) & (req < slots) & (topk_indices >= 0) & (topk_indices < context)
    positions = slot_map[
        req.clamp(0, slots - 1).long(), topk_indices.clamp(0, context - 1).long()
    ]
    positions = torch.where(valid, positions, -1)
    hits = (positions + 1).clamp(max=1)
    return hits, positions

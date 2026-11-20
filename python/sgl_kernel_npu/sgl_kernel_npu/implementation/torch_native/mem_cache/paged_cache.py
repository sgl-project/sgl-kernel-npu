"""PyTorch references for ND cache updates and in-place ScatterND."""

import torch

from ..norm.rms_norm import rms_norm
from ..rope.rotary_mul import rotary_mul


def _write_slots(cache, slots, rows):
    slots = slots.detach().cpu().reshape(-1).long()
    valid = slots >= 0
    count = slots.numel()
    slots = slots[valid]
    updated = cache.detach().cpu().clone()
    values = rows.detach().cpu().reshape(count, *cache.shape[2:])
    updated[slots // cache.size(1), slots % cache.size(1)] = values[valid]
    cache.copy_(updated.to(cache.device))


def reshape_and_cache(key, value, key_cache, value_cache, slot_indices):
    _write_slots(key_cache, slot_indices, key)
    _write_slots(value_cache, slot_indices, value)


def scatter_pa_kv_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    *,
    compress_lens=None,
    compress_seq_offsets=None,
    seq_lens=None,
    cache_mode="PA_NZ",
):
    if cache_mode != "Norm" or any(
        arg is not None for arg in (compress_lens, compress_seq_offsets, seq_lens)
    ):
        raise NotImplementedError("torch_native requires uncompressed ND scatter")
    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def _write_norm(cache, index, rows):
    updated = cache.detach().cpu().clone()
    positions = index.detach().cpu().reshape(rows.size(0), rows.size(2)).long()
    batch, token = torch.where(positions >= 0)
    updated[batch, 0, positions[batch, token]] = rows[batch, 0, token]
    cache.copy_(updated.to(cache.device))


def kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    *,
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    v=None,
    epsilon=1e-5,
    cache_mode="Norm",
    is_output_kv=False,
):
    if (
        cache_mode not in ("Norm", "PA", "PA_BNSD")
        or any(
            arg is not None
            for arg in (k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v)
        )
        or k_cache.dtype != kv.dtype
        or ckv_cache.dtype != kv.dtype
    ):
        raise NotImplementedError("torch_native requires non-quantized ND KV caches")
    src = kv.detach().cpu()
    latent, rope = src.split((gamma.numel(), src.size(-1) - gamma.numel()), dim=-1)
    normalized = rms_norm(latent, gamma.detach().cpu(), epsilon)[0]
    # KV RMSNorm RoPE interleaves input pairs before the rotation in half order.
    rope = rope.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
    rotated = rotary_mul(rope, cos.detach().cpu(), sin.detach().cpu())
    if cache_mode == "Norm":
        _write_norm(k_cache, index, rotated)
        _write_norm(ckv_cache, index, normalized)
    else:
        _write_slots(k_cache, index, rotated)
        _write_slots(ckv_cache, index, normalized)
    if is_output_kv:
        return k_cache, ckv_cache, rotated.to(kv.device), normalized.to(kv.device)
    return (
        k_cache,
        ckv_cache,
        torch.empty_like(cos),
        torch.empty_like(kv[..., : gamma.numel()]),
    )


def scatter_nd_update_(input, indices, updates):
    coordinates = indices.to(device="cpu", dtype=torch.int64)
    if coordinates.ndim < 1 or not 0 < coordinates.size(-1) <= input.ndim:
        raise ValueError("ScatterND index depth must be between one and input rank")
    if (coordinates < 0).any():
        raise IndexError("ScatterND indices must be nonnegative")
    if updates.dtype != input.dtype:
        raise ValueError("ScatterND updates must use the destination dtype")
    # Byte views also support FP8 cache rows on runtimes lacking FP8 index_put.
    target, values = input, updates
    if input.element_size() == 1:
        target, values = input.view(torch.uint8), updates.view(torch.uint8)
    result = target.detach().cpu().clone()
    result[tuple(coordinates.unbind(-1))] = values.detach().cpu()
    target.copy_(result.to(target.device))
    return input

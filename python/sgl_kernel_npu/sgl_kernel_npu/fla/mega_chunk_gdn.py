import os
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

import torch
from sgl_kernel_npu.fla.utils import (
    _lru_get,
    _lru_put,
    cached_cu_lens_cpu,
    is_gdn_meta_cache_enabled,
)

HEAD_DIM = 128
CHUNK_SIZE = 128


# Grow-only token slab for kernel-internal + caller-discarded tensors. The
# AscendC kernel addresses them as h*total_tokens+t and the op host never
# validates their shapes, so an oversized contiguous buffer is safe: buffers
# grow to the longest prefill seen and are reused across layers and steps.
_POOL_OUTPUTS = os.getenv("SGLANG_NPU_GDN_POOL_OUTPUTS", "1") != "0"
# The kernel fully covers every element it consumes: kkt writes all valid rows
# of A; solve_tril loads only the valid_size triangle and TFILLPADs the rest;
# chunk_h stores S/final_state stripes unconditionally. Zero-init is therefore
# unnecessary (verified against csrc/mega_chunk_gdn kernels).
_SKIP_OUTPUT_ZERO = os.getenv("SGLANG_NPU_GDN_SKIP_OUTPUT_ZERO", "1") != "0"

# Cross-layer mega scratch / cu32 (shape or content keyed). Not returned to callers.
_MEGA_CU32_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_MEGA_NUM_CHUNKS_CACHE: "OrderedDict[tuple, int]" = OrderedDict()
_MEGA_SCRATCH_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_MEGA_SLAB_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_DUMMY_F32_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()


def clear_mega_workspace_cache() -> None:
    _MEGA_CU32_CACHE.clear()
    _MEGA_NUM_CHUNKS_CACHE.clear()
    _MEGA_SCRATCH_CACHE.clear()
    _MEGA_SLAB_CACHE.clear()
    _DUMMY_F32_CACHE.clear()


def _device_key(device: torch.device) -> tuple[str, int]:
    return device.type, 0 if device.index is None else device.index


def _device_from_key(device_type: str, device_index: int) -> torch.device:
    return torch.device(device_type, device_index)


@lru_cache(maxsize=16)
def _masks(device_type: str, device_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = _device_from_key(device_type, device_index)
    mask_lower = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=-1
    ).float()
    mask_full = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=0
    ).float()
    return mask_lower, mask_full


@lru_cache(maxsize=16)
def _minus_identity(device_type: str, device_index: int) -> torch.Tensor:
    device = _device_from_key(device_type, device_index)
    minus_identity = torch.zeros(
        CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16
    )
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def _dummy_f32(device_type: str, device_index: int) -> torch.Tensor:
    """Placeholder for the op's a_inv_f32 argument: the kernel signature takes
    it but never reads it (the fp32->fp16 cast stage was removed). Saves a
    full T*H*C fp32 allocation + memset per layer."""
    key = (device_type, device_index)
    dummy = _DUMMY_F32_CACHE.get(key)
    if dummy is None:
        dummy = torch.zeros(1, device=_device_from_key(device_type, device_index),
                            dtype=torch.float32)
        _DUMMY_F32_CACHE[key] = dummy
    return dummy


def _total_chunks_from_cpu(cu: list[int]) -> int:
    total = 0
    for start, end in zip(cu, cu[1:]):
        total += (end - start + CHUNK_SIZE - 1) // CHUNK_SIZE
    return total


def _total_chunks(cu_seqlens: torch.Tensor) -> int:
    _, cu = cached_cu_lens_cpu(cu_seqlens)
    return _total_chunks_from_cpu(cu)


def _block_dim(device: torch.device) -> int:
    try:
        props = torch.npu.get_device_properties(device)
        return max(1, int(getattr(props, "cube_core_num", 24)))
    except (RuntimeError, AttributeError, AssertionError):
        return 24


def _get_cu32_and_chunks(
    cu_seqlens: Optional[torch.Tensor],
    total_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Return (cu32, num_sequences, num_chunks), caching cu32 by content across layers."""
    if cu_seqlens is None:
        cu_cpu = [0, total_tokens]
        num_chunks = _total_chunks_from_cpu(cu_cpu)
        if not is_gdn_meta_cache_enabled():
            cu32 = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
            return cu32, 1, num_chunks
        key = (tuple(cu_cpu), _device_key(device))
        cached = _lru_get(_MEGA_CU32_CACHE, key)
        if cached is not None:
            return cached, 1, num_chunks
        cu32 = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
        _lru_put(_MEGA_CU32_CACHE, key, cu32)
        return cu32, 1, num_chunks

    content_key, cu_cpu = cached_cu_lens_cpu(cu_seqlens)
    num_sequences = len(cu_cpu) - 1
    if is_gdn_meta_cache_enabled():
        nkey = (content_key, CHUNK_SIZE)
        num_chunks = _lru_get(_MEGA_NUM_CHUNKS_CACHE, nkey)
        if num_chunks is None:
            num_chunks = _total_chunks_from_cpu(cu_cpu)
            _lru_put(_MEGA_NUM_CHUNKS_CACHE, nkey, num_chunks)
        ckey = (content_key, _device_key(device))
        cu32 = _lru_get(_MEGA_CU32_CACHE, ckey)
        if cu32 is None:
            cu32 = cu_seqlens.to(torch.int32)
            _lru_put(_MEGA_CU32_CACHE, ckey, cu32)
        return cu32, num_sequences, num_chunks

    cu32 = cu_seqlens.to(torch.int32)
    return cu32, num_sequences, _total_chunks_from_cpu(cu_cpu)


def _get_mega_scratch(block_dim: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> dict:
    """Scratch buffers rewritten each launch; safe to reuse across GDN layers."""
    if not is_gdn_meta_cache_enabled():
        return _alloc_mega_scratch(block_dim, head_dim, device, dtype)

    key = (int(block_dim), int(head_dim), str(dtype), _device_key(device))
    cached = _lru_get(_MEGA_SCRATCH_CACHE, key)
    if cached is not None:
        return cached
    pack = _alloc_mega_scratch(block_dim, head_dim, device, dtype)
    return _lru_put(_MEGA_SCRATCH_CACHE, key, pack)


def _alloc_mega_scratch(block_dim: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> dict:
    wy_a1 = torch.zeros(
        block_dim, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=dtype
    )
    o_qk = torch.zeros(
        block_dim, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=dtype
    )
    return {
        "kkt": torch.zeros(
            block_dim * 2, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=dtype
        ),
        "wy_a1": wy_a1,
        "wy_a2": torch.zeros_like(wy_a1),
        "h": torch.zeros(
            block_dim * 4, head_dim, head_dim, device=device, dtype=dtype
        ),
        "o_qk": o_qk,
        "o_qs": torch.zeros(
            block_dim, CHUNK_SIZE, head_dim, device=device, dtype=dtype
        ),
        "o_gated": torch.zeros_like(o_qk),
    }


def _alloc_slab(
    cap: int, num_value_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> dict:
    """Token-scaled buffers sized to `cap` >= total_tokens. The kernel only
    touches elements below total_tokens, so oversizing is safe."""
    z = torch.zeros if not _SKIP_OUTPUT_ZERO else torch.empty
    A = z(1, cap, num_value_heads, CHUNK_SIZE, device=device, dtype=dtype)
    return {
        "cap": cap,
        "g_sum": torch.empty(1, cap, num_value_heads, device=device, dtype=torch.float32),
        "g_t": torch.empty(num_value_heads, cap, device=device, dtype=torch.float32),
        "beta_t": torch.empty(num_value_heads, cap, device=device, dtype=dtype),
        "A": A,
        # Returned by the wrapper but discarded by every caller in the serving
        # stack (chunk_gated_delta_rule_npu keeps only o/final_state/h).
        "A_inv": z(1, cap, num_value_heads, CHUNK_SIZE, device=device, dtype=dtype),
        "w": torch.empty(1, cap, num_value_heads, head_dim, device=device, dtype=dtype),
        "u": torch.empty(1, cap, num_value_heads, head_dim, device=device, dtype=dtype),
        "v_new": torch.empty(1, cap, num_value_heads, head_dim, device=device, dtype=dtype),
    }


def _get_mega_slab(
    total_tokens: int,
    num_value_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    key = (int(num_value_heads), int(head_dim), str(dtype), _device_key(device))
    slab = _MEGA_SLAB_CACHE.get(key)
    if slab is not None and slab["cap"] >= total_tokens:
        return slab
    cap = total_tokens if slab is None else max(total_tokens, slab["cap"])
    slab = _alloc_slab(cap, num_value_heads, head_dim, device, dtype)
    _MEGA_SLAB_CACHE[key] = slab
    return slab


def run_mega_chunk_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    return_internals: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q_dtype, k_dtype, v_dtype = q.dtype, k.dtype, v.dtype
    q, k, v, beta = (t.half() for t in (q, k, v, beta))

    _, total_tokens, _, head_dim = q.shape
    num_value_heads = v.shape[-2]
    device_type, device_index = _device_key(q.device)

    cu32, num_sequences, num_chunks = _get_cu32_and_chunks(
        cu_seqlens, total_tokens, q.device
    )
    num_matrices = num_chunks * num_value_heads

    mask_lower, mask_full = _masks(device_type, device_index)
    minus_identity = _minus_identity(device_type, device_index)
    a_inv_f32 = _dummy_f32(device_type, device_index)

    # Outputs / states that escape this call — allocate fresh every layer.
    z = torch.zeros if not _SKIP_OUTPUT_ZERO else torch.empty
    h = z(
        num_chunks * num_value_heads,
        head_dim,
        head_dim,
        device=q.device,
        dtype=torch.float16,
    )
    final_state = z(
        num_sequences * num_value_heads,
        head_dim,
        head_dim,
        device=q.device,
        dtype=torch.float16,
    )
    out = torch.empty_like(v)

    cache_on = is_gdn_meta_cache_enabled()
    if cache_on and _POOL_OUTPUTS:
        slab = _get_mega_slab(total_tokens, num_value_heads, head_dim, q.device, torch.float16)
        g_sum = slab["g_sum"]
        A_inv = slab["A_inv"]
        w = slab["w"]
        u = slab["u"]
        v_new = slab["v_new"]
        g_t = slab["g_t"]
        beta_t = slab["beta_t"]
        A = slab["A"]
    else:
        g_sum = torch.empty_like(g, dtype=torch.float32)
        A_inv = z(
            1, total_tokens, num_value_heads, CHUNK_SIZE,
            device=q.device, dtype=torch.float16,
        )
        w = torch.empty_like(v)
        u = torch.empty_like(v)
        v_new = torch.empty_like(v)
        g_t = torch.empty(num_value_heads, total_tokens, device=q.device, dtype=torch.float32)
        beta_t = torch.empty(num_value_heads, total_tokens, device=q.device, dtype=torch.float16)
        A = z(
            1, total_tokens, num_value_heads, CHUNK_SIZE,
            device=q.device, dtype=torch.float16,
        )

    has_initial_state = initial_state is not None
    initial_state = (
        initial_state.to(torch.float16) if has_initial_state else final_state
    )

    block_dim = _block_dim(q.device)
    scratch = _get_mega_scratch(block_dim, head_dim, q.device, torch.float16)

    torch.ops.npu.mega_chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        mask_lower,
        mask_full,
        minus_identity,
        cu32,
        out,
        g_sum,
        g_t,
        beta_t,
        A,
        a_inv_f32,
        A_inv,
        w,
        u,
        h,
        v_new,
        final_state,
        initial_state,
        has_initial_state,
        scratch["kkt"],
        scratch["wy_a1"],
        scratch["wy_a2"],
        scratch["h"],
        scratch["o_qk"],
        scratch["o_qs"],
        scratch["o_gated"],
        block_dim,
        num_sequences,
        total_tokens,
        total_tokens,
        num_matrices,
    )

    h = h.view(1, num_chunks, num_value_heads, head_dim, head_dim)
    if output_final_state:
        final_state_out = final_state.view(
            num_sequences, num_value_heads, head_dim, head_dim
        ).to(torch.float32)
    else:
        final_state_out = None
    # Slab buffers may be oversized (cap >= total_tokens); the kernel only
    # writes below total_tokens, so slice the escaped tensors back to size.
    T = total_tokens
    if not return_internals:
        # chunk_gated_delta_rule_fwd discards A_inv/w/v_new at default
        # SUPPRESS_LEVEL: skip the .to() casts (~1.5 GB/layer copies at 128k).
        return (
            g_sum[:, :T],
            (out * scale).to(q_dtype),
            None,
            final_state_out,
            None,
            h,
            None,
        )
    return (
        g_sum[:, :T],
        (out * scale).to(q_dtype),
        A_inv[:, :T].to(k_dtype),
        final_state_out,
        w[:, :T].to(k_dtype),
        h.to(k_dtype),
        v_new[:, :T].to(v_dtype),
    )

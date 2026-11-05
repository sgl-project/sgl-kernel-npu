"""Fused pooled/tail commit; speculative rollback projection is owned here."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _write_cache_kernel(
    Pooled,
    State,
    Key,
    Gate,
    Slots,
    Writes,
    StateRows,
    Tokens,
    Cache,
    TailK,
    TailG,
    TAIL_ROWS: tl.constexpr,
    POOLED_ROWS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    PAGE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    TAIL_CAPACITY: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    TOKEN_ROWS: tl.constexpr,
    SPEC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    if row < TAIL_ROWS:
        dest = tl.load(Writes + row).to(tl.int64)
        valid = (dest >= 0) & (dest < TAIL_CAPACITY)
        if SPEC:
            token = tl.load(Tokens + row)
            valid = valid & (token >= 0) & (token < TOKEN_ROWS)
            key = tl.load(Key + row * D + cols, valid & (cols < D), other=0)
            gate = tl.load(Gate + row * D + cols, valid & (cols < D), other=0)
        else:
            source = tl.load(StateRows + row).to(tl.int64)
            valid = valid & (source >= 0) & (source < STATE_ROWS)
            address = source // R * STATE_STRIDE + source % R * (2 * D)
            key = tl.load(State + address + cols, valid & (cols < D), other=0)
            gate = tl.load(State + address + D + cols, valid & (cols < D), other=0)
        # Persistent rings stay BF16; compute/state values remain FP32 until store.
        tl.store(
            TailK + dest * D + cols, key.to(TailK.dtype.element_ty), valid & (cols < D)
        )
        tl.store(
            TailG + dest * D + cols, gate.to(TailG.dtype.element_ty), valid & (cols < D)
        )
    else:
        source = row - TAIL_ROWS
        slot = tl.load(Slots + source).to(tl.int64)
        valid = (source < POOLED_ROWS) & (slot >= 0) & (slot < CACHE_PAGES * PAGE)
        # A -1 mapping never reads uninitialized CANN output capacity.
        value = tl.load(Pooled + source * D + cols, valid & (cols < D), other=0)
        address = slot // PAGE * CACHE_STRIDE + slot % PAGE * D + cols
        tl.store(Cache + address, value, valid & (cols < D))


def key_pool_write_cache(
    pooled_key,
    state_cache,
    pooled_slot_mapping,
    tail_write_indices,
    tail_state_indices,
    tail_token_indices,
    index_cache,
    tail_k,
    tail_gate,
    *,
    hidden_states=None,
    wk=None,
    gate_weight=None,
    norm_weight=None,
    norm_bias=None,
    mode="normal",
    cmp_ratio=4,
    norm_eps=1e-6,
):
    if mode not in ("normal", "mtp"):
        raise ValueError("mode must be 'normal' or 'mtp'")
    if cmp_ratio < 2 or cmp_ratio > 128 or cmp_ratio & (cmp_ratio - 1):
        raise ValueError("invalid cmp_ratio")
    if tail_k.ndim != 3 or tail_gate.shape != tail_k.shape:
        raise ValueError("tail K/Gate must have matching [R,C,D] shapes")
    width = tail_k.shape[-1]
    batch = tail_write_indices.shape[0] if tail_write_indices.ndim == 2 else -1
    if batch < 0 or (mode == "normal" and tail_write_indices.shape[1] != cmp_ratio):
        raise ValueError("tail mappings must be [B,r] in normal mode or [B,N] in MTP")
    if mode == "mtp" and tail_k.shape[1] < cmp_ratio + tail_write_indices.shape[1]:
        raise ValueError("MTP ring capacity must cover r+N")
    if any(
        t.shape != tail_write_indices.shape
        for t in (tail_state_indices, tail_token_indices)
    ):
        raise ValueError("tail mapping shapes must match")
    if state_cache.shape != (1 + 2 * batch, cmp_ratio, 2 * width):
        raise ValueError("state shape does not match tail mappings")
    if pooled_key.ndim != 2 or pooled_key.shape[1] != width:
        raise ValueError("pooled_key must be [Pcap,D]")
    if (
        pooled_slot_mapping.ndim != 1
        or pooled_slot_mapping.numel() < pooled_key.shape[0]
    ):
        raise ValueError("pooled mapping does not cover output capacity")
    if (
        index_cache.ndim not in (3, 4)
        or index_cache.shape[-1] != width
        or (index_cache.ndim == 4 and index_cache.shape[2] != 1)
    ):
        raise ValueError(
            "index_cache must be [pages,page_size,1,D] or [pages,page_size,D]"
        )
    if (
        index_cache.shape[1] <= 0
        or index_cache.stride(-1) != 1
        or index_cache.stride(1) != width
    ):
        raise ValueError("index page rows must be contiguous")
    if (
        state_cache.stride(-1) != 1
        or state_cache.stride(1) != 2 * width
        or state_cache.stride(0) < cmp_ratio * 2 * width
    ):
        raise ValueError("state rows must be contiguous")
    typed = [
        (pooled_key, torch.bfloat16),
        (index_cache, torch.bfloat16),
        (state_cache, torch.float32),
        (tail_k, torch.bfloat16),
        (tail_gate, torch.bfloat16),
    ]
    typed += [
        (t, torch.int64)
        for t in (
            pooled_slot_mapping,
            tail_write_indices,
            tail_state_indices,
            tail_token_indices,
        )
    ]
    for tensor, dtype in typed:
        if tensor.dtype != dtype or tensor.device != pooled_key.device:
            raise ValueError("invalid cache/mapping dtype or device")
        if (
            tensor is not state_cache
            and tensor is not index_cache
            and not tensor.is_contiguous()
        ):
            raise ValueError("tail/pooled data and mappings must be contiguous")
    speculative = mode == "mtp"
    key = gate = pooled_key
    if speculative:
        if hidden_states is None or wk is None or gate_weight is None:
            raise ValueError("MTP requires hidden_states and K/Gate weights")
        if norm_eps <= 0 or (norm_weight is None) != (norm_bias is None):
            raise ValueError("invalid norm epsilon/paired parameters")
        if (
            hidden_states.ndim != 2
            or wk.shape != (width, hidden_states.shape[1])
            or gate_weight.shape != wk.shape
        ):
            raise ValueError("MTP projection requires TH and [D,H] weights")
        for tensor in (hidden_states, wk, gate_weight):
            if (
                tensor.dtype != torch.bfloat16
                or tensor.device != pooled_key.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "MTP projection requires contiguous BF16 inputs on the cache device"
                )
        if norm_weight is not None:
            for tensor in (norm_weight, norm_bias):
                if (
                    tensor.shape != (width,)
                    or tensor.dtype != torch.float32
                    or tensor.device != pooled_key.device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "normalization requires contiguous FP32 [D] on the cache device"
                    )
        indices = tail_token_indices.flatten()
        if hidden_states.shape[0]:
            retained = torch.index_select(
                hidden_states, 0, indices.clamp(0, hidden_states.shape[0] - 1)
            )
            # Same rollback convention as the existing adapter: BF16 GEMM,
            # FP32 LayerNorm/state. Invalid gathered rows are never committed.
            key = F.linear(retained, wk).float()
            if norm_weight is not None:
                key = F.layer_norm(key, (width,), norm_weight, norm_bias, norm_eps)
            gate = F.linear(retained, gate_weight).float()
        else:
            key = tail_k.new_zeros((indices.numel(), width), dtype=torch.float32)
            gate = tail_gate.new_zeros((indices.numel(), width), dtype=torch.float32)
    total = tail_write_indices.numel() + pooled_key.shape[0]
    if not total:
        return
    _write_cache_kernel[(total,)](
        pooled_key,
        state_cache,
        key,
        gate,
        pooled_slot_mapping,
        tail_write_indices,
        tail_state_indices,
        tail_token_indices,
        index_cache,
        tail_k,
        tail_gate,
        tail_write_indices.numel(),
        pooled_key.shape[0],
        width,
        cmp_ratio,
        index_cache.shape[1],
        index_cache.stride(0),
        index_cache.shape[0],
        tail_k.shape[0] * tail_k.shape[1],
        state_cache.shape[0] * cmp_ratio,
        state_cache.stride(0),
        hidden_states.shape[0] if speculative else 0,
        speculative,
        triton.next_power_of_2(width),
    )

"""Fuse physical NZ indices for snapshot-safe MLA latent/RoPE relocation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_nz_rows(
    loc, tiles, TILES: tl.constexpr, PAGE: tl.constexpr, SHIFT: tl.constexpr
):
    if SHIFT >= 0:
        # Avoid int32 division lowering through float32 near the slot limit.
        page = loc >> SHIFT
        slot = loc & (PAGE - 1)
    else:
        page = loc.to(tl.int64) // PAGE
        slot = loc.to(tl.int64) % PAGE
    return (page.to(tl.int64) * TILES + tiles) * PAGE + slot


@triton.jit
def _mla_move_indices_kernel(
    source,
    target,
    src_k_indices,
    dst_k_indices,
    src_r_indices,
    dst_r_indices,
    COUNT,
    SRC_STRIDE: tl.constexpr,
    TGT_STRIDE: tl.constexpr,
    K_TILES: tl.constexpr,
    R_TILES: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    LARGE_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k_count = COUNT * K_TILES
    r_count = COUNT * R_TILES
    k_mask = rows < k_count
    r_mask = rows < r_count
    # Keep division/remainder in int32 for normal pools; physical NZ row
    # products below are widened independently. Huge pools retain int64 slots.
    slot_dtype = tl.int64 if LARGE_SLOTS else tl.int32
    k_src = tl.load(source + (rows // K_TILES) * SRC_STRIDE, k_mask, other=0).to(
        slot_dtype
    )
    k_dst = tl.load(target + (rows // K_TILES) * TGT_STRIDE, k_mask, other=0).to(
        slot_dtype
    )
    r_src = tl.load(source + (rows // R_TILES) * SRC_STRIDE, r_mask, other=0).to(
        slot_dtype
    )
    r_dst = tl.load(target + (rows // R_TILES) * TGT_STRIDE, r_mask, other=0).to(
        slot_dtype
    )
    k_src_rows = _mla_nz_rows(k_src, rows % K_TILES, K_TILES, PAGE, PAGE_SHIFT)
    k_dst_rows = _mla_nz_rows(k_dst, rows % K_TILES, K_TILES, PAGE, PAGE_SHIFT)
    r_src_rows = _mla_nz_rows(r_src, rows % R_TILES, R_TILES, PAGE, PAGE_SHIFT)
    r_dst_rows = _mla_nz_rows(r_dst, rows % R_TILES, R_TILES, PAGE, PAGE_SHIFT)
    tl.store(src_k_indices + rows, k_src_rows, k_mask)
    tl.store(dst_k_indices + rows, k_dst_rows, k_mask)
    tl.store(src_r_indices + rows, r_src_rows, r_mask)
    tl.store(dst_r_indices + rows, r_dst_rows, r_mask)


def move_mla_kv_cache(
    k_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    page_size: int,
    use_fia_nz: bool = False,
) -> None:
    """Relocate split 16-bit MLA caches, fusing all NZ index arithmetic.

    Caches have contiguous [layer, page, slot, 1, dim] storage. NZ storage is
    addressed as [layer, page, dim/16, slot, 16]. Locations can be strided int32
    or int64 tensors. Targets must be unique; sources may repeat or overlap
    targets, including cycles. Each native gather snapshots all source rows
    before the corresponding scatter writes to the cache. ND needs no physical
    index expansion and retains the native gather/scatter path.
    """
    count = src_loc.numel()
    if count == 0:
        return
    if tgt_loc.numel() != count:
        raise ValueError("MLA source and target location counts must match.")
    src_loc, tgt_loc = src_loc.reshape(-1), tgt_loc.reshape(-1)
    # The native BF16 copy path preserves 16-bit payloads without the FP16
    # conversion that canonicalizes NaNs or the slower int16 gather path.
    if k_cache.dtype != torch.bfloat16:
        k_cache = k_cache.view(torch.bfloat16)
    if rope_cache.dtype != torch.bfloat16:
        rope_cache = rope_cache.view(torch.bfloat16)
    if use_fia_nz:
        k_tiles, r_tiles = k_cache.shape[-1] // 16, rope_cache.shape[-1] // 16
        k_count, r_count = count * k_tiles, count * r_tiles
        # Keep zero-offset outputs for the native NPU index operators.
        k_src, k_dst, r_src, r_dst = [
            torch.empty(n, dtype=torch.int64, device=src_loc.device)
            for n in (k_count, k_count, r_count, r_count)
        ]
        _mla_move_indices_kernel[(triton.cdiv(max(k_count, r_count), 256),)](
            src_loc,
            tgt_loc,
            k_src,
            k_dst,
            r_src,
            r_dst,
            count,
            SRC_STRIDE=src_loc.stride(0),
            TGT_STRIDE=tgt_loc.stride(0),
            K_TILES=k_tiles,
            R_TILES=r_tiles,
            PAGE=page_size,
            PAGE_SHIFT=(
                page_size.bit_length() - 1 if page_size & (page_size - 1) == 0 else -1
            ),
            LARGE_SLOTS=k_cache.shape[1] * page_size > 2**31,
            BLOCK=256,
        )
        k_rows = k_cache.view(k_cache.shape[0], -1, 16)
        r_rows = rope_cache.view(rope_cache.shape[0], -1, 16)
    else:
        k_src = r_src = src_loc.long()
        k_dst = r_dst = tgt_loc.long()
        k_rows = k_cache.flatten(1, 2)
        r_rows = rope_cache.flatten(1, 2)
    k_rows.index_copy_(1, k_dst, k_rows.index_select(1, k_src))
    r_rows.index_copy_(1, r_dst, r_rows.index_select(1, r_src))

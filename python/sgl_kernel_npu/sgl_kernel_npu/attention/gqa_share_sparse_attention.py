from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.indexer.flash_block_score_decode import (
    _floor_power_of_2,
    _get_vectorcore_num_safe,
    _native_sanitize_topk_kernel,
    _normalize_topk_idx_for_gqa,
)

# --- Native Ascend block-sparse attention (npu_sparse_attention_score) ---
# Routes decode/verify main attention through the native op; falls back to Triton split-K.
logger = logging.getLogger(__name__)
_native_warned = False


@functools.lru_cache(maxsize=1)
def _get_native_sparse_op():
    # native block-sparse attention: the direct-launch op compiled into
    # sgl_kernel_npu (csrc/sparse_attention_score). Self-contained -- no external
    # vendor package / ASCEND_CUSTOM_OPP_PATH needed. Returns None -> Triton
    # split-K fallback.
    try:
        import sgl_kernel_npu  # noqa: F401  (registers torch.ops.npu.*)

        return getattr(torch.ops.npu, "npu_sparse_attention_score", None)
    except (ImportError, RuntimeError, AttributeError):
        return None


def _warn_native_unavailable() -> None:
    global _native_warned
    if _native_warned:
        return
    logger.warning(
        "The native npu_sparse_attention_score op could not be loaded "
        "(sgl_kernel_npu not installed or torch.ops.npu.npu_sparse_attention_score "
        "not registered). Falling back to the Triton split-K path."
    )
    _native_warned = True


def _native_decode_main(
    q,
    k,
    v,
    topk_idx,
    seq_lens,
    block_size,
    sm_scale,
    block_table,
    req_to_token,
    req_pool_indices,
    max_num_blocks,
    num_kv_heads,
    head_dim,
):
    """Run decode attention via the native aclnn sparse attention op (cuda-graph safe)."""
    op = _get_native_sparse_op()
    if op is None:
        return None
    device = q.device
    batch_size = q.shape[0]
    if block_table is not None:
        bt = block_table.to(torch.int32)
    else:
        # Gather only the B x maxBlocks block-start slots (NOT the full B x max_ctx),
        # then // block_size -> physical page id per (request, logical block).
        bidx = torch.arange(max_num_blocks, device=device, dtype=torch.int32)
        slots = req_to_token[
            req_pool_indices[:, None].to(torch.int64),
            (bidx * block_size)[None, :].to(torch.int64),
        ]
        bt = (slots // block_size).to(torch.int32)
    num_kv_heads = topk_idx.shape[0]
    batch_size = q.shape[0]
    num_pages = k.shape[0]
    # Sanitize OOB select_idx beyond per-query KV in one triton kernel (in-place;
    # no host sync -> cuda-graph safe). No fold/cap needed.
    sel = topk_idx.to(torch.int32).clone()
    select_num_idx = torch.empty(
        (num_kv_heads, batch_size), dtype=torch.int32, device=q.device
    )
    _native_sanitize_topk_kernel[(num_kv_heads, batch_size)](
        sel,
        select_num_idx,
        seq_lens.to(torch.int32),
        sel.stride(0),
        sel.stride(1),
        sel.stride(2),
        select_num_idx.stride(0),
        select_num_idx.stride(1),
        block_size=block_size,
        SLOTS=sel.shape[-1],
        num_warps=1,
        num_stages=1,
    )
    # -1 sentinels must not reach the C++ op (block_table[-1] OOB -> NaN); clamp to 0.
    sel = torch.where(sel < 0, torch.zeros_like(sel), sel)
    # bt valid for attended slots (sanitize ensures sel < nblocks). Do NOT MAX-pad
    # actual_seq_lengths_kv -- causes OOB on replay.
    actual_kv = seq_lens.to(torch.int32)
    out = op(
        q,
        k,
        v,
        sel,
        bt,
        select_num_idx=select_num_idx,
        actual_seq_lengths=torch.ones(batch_size, dtype=torch.int32, device=device),
        actual_seq_lengths_kv=actual_kv,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale if sm_scale is not None else head_dim**-0.5,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=4,
    )
    return out


# Tunable launch configs (not triton.autotune -- each shape bucket compiles to
# a single deterministic artifact, cuda-graph capture safe).
_SPARSE_DECODE_NW = 4
_SPARSE_DECODE_NS = 2
_MERGE_NW = 4
_MERGE_NS = 2

# MiniMax-M3 selects 16 scored blocks + 1 forced local block. Short TopK list is
# launch/merge bound; small batches use the single-chunk fast path (skip merge).
_MINIMAX_SINGLE_CHUNK_MAX_TOPK = 17
_MINIMAX_SINGLE_CHUNK_MAX_BATCH = 4


def _choose_num_topk_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_topk: int,
    max_num_topk_chunks: int = 8,
) -> int:
    """Choose split-topk chunks in an SGLang-like but Ascend-conservative way."""
    if max_topk <= 1:
        return 1

    if (
        num_kv_heads == 1
        and batch_size <= _MINIMAX_SINGLE_CHUNK_MAX_BATCH
        and max_topk <= _MINIMAX_SINGLE_CHUNK_MAX_TOPK
    ):
        return 1

    num_vectorcore = _get_vectorcore_num_safe()
    # Ascend: saturate vector cores (1 program/core). Once B*nkvh >= vc, extra
    # chunks only add wave + merge overhead. Small batches still split.
    target_grid = num_vectorcore
    target = max(1, target_grid // max(1, batch_size * num_kv_heads))
    target = min(max_topk, max_num_topk_chunks, target)
    return _floor_power_of_2(target)


# ============================ Sparse BNSD Decode Kernel ============================


from sgl_kernel_npu.implementation.triton.attention.gqa_share_sparse_decode import (
    _gqa_share_sparse_decode_bnsd_kernel,
)

# ====================== Merge split-topk sparse attention output ======================


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_bnsd_kernel(
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    out_ptr,  # [B, QH, D]
    head_dim,
    # strides
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    # meta
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Merge split-topk chunk outputs via online softmax reduction."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for c in tl.static_range(0, NUM_TOPK_CHUNKS):
        lse_c = tl.load(
            lse_ptr + c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h
        )

        o_c = tl.load(
            o_ptr
            + c * stride_o_c
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + off_d * stride_o_d,
            mask=off_d < head_dim,
            other=0.0,
        ).to(tl.float32)

        # Avoid -inf - -inf -> NaN for all-empty chunks.
        valid = lse_c > float("-inf")
        m_new = tl.maximum(m, lse_c)

        scale_old = tl.where(
            m > float("-inf"),
            tl.exp(m - m_new),
            tl.zeros_like(m),
        )
        scale_new = tl.where(
            valid,
            tl.exp(lse_c - m_new),
            tl.zeros_like(lse_c),
        )

        acc = acc * scale_old + o_c * scale_new
        l = l * scale_old + scale_new
        m = m_new

    out = tl.where(l > 0.0, acc / l, acc)

    tl.store(
        out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + off_d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


# ================================= Python Wrapper =================================


@torch.no_grad()
def flash_decode_bnsd_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # optional [num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    block_table: Optional[torch.Tensor],  # [batch_size, max_num_blocks]
    seq_lens: torch.Tensor,  # [batch_size]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads or num_q_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    num_topk_chunks: Optional[int] = None,
    max_num_topk_chunks: int = 8,
    req_to_token: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    max_num_blocks: Optional[int] = None,
    num_pages: Optional[int] = None,
    sanitize_page_ids: bool = False,
    # Selected blocks fused into one K/V tile + dot per loop step (1 = per-block
    # path). Must be a power of two so BLOCK_SIZE_N stays aligned.
    topk_blocks_per_step: int = 1,
    # Hoist the chunk's topk-idx + page-id gathers into a vectorized prologue
    # (breaks the per-step idx->page->K load dependency chain).
    prefetch_idx: bool = False,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    # AB switch: False forces the Triton split-K path even when the native op is
    # available (decode vs verify isolation for accept-rate debugging).
    use_native: bool = True,
) -> torch.Tensor:
    """Sparse decode attention using BNSD KV cache and precomputed topk blocks.
    Returns [batch_size, num_q_heads, head_dim]."""
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape
    # Native block-sparse decode main, before direct-page-lookup asserts (hoisted
    # block_table). Falls back to Triton split-K if unavailable or use_native=False.
    if use_native and _get_native_sparse_op() is not None:
        _nkvh = k_cache_bnsd.shape[2]
        if topk_idx.shape[0] == _nkvh and (
            block_table is not None or req_to_token is not None
        ):
            out = _native_decode_main(
                q,
                k_cache_bnsd,
                v_cache_bnsd,
                topk_idx,
                seq_lens,
                block_size,
                sm_scale,
                block_table,
                req_to_token,
                req_pool_indices,
                max_num_blocks,
                _nkvh,
                q.shape[2],
            )
            if out is not None:
                return out
            _warn_native_unavailable()

    use_direct_page_lookup = req_to_token is not None
    assert (req_pool_indices is not None) == use_direct_page_lookup
    if use_direct_page_lookup:
        assert block_table is None
        assert req_to_token.ndim == 2
        assert req_to_token.dtype in (torch.int32, torch.int64)
        assert req_pool_indices.ndim == 1
        assert req_pool_indices.dtype in (torch.int32, torch.int64)
        assert max_num_blocks is not None and max_num_blocks > 0
        assert num_pages is not None and num_pages > 0
    else:
        assert block_table is not None
        assert block_table.dtype in (torch.int32, torch.int64)

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert seq_lens.shape[0] == batch_size
    assert topk_idx.shape[1] == batch_size
    if use_direct_page_lookup:
        assert req_pool_indices.shape[0] == batch_size
        assert max_num_blocks * block_size <= req_to_token.shape[1]
        page_source = req_to_token
        page_source_rows = req_pool_indices
        max_kv_len = int(max_num_blocks) * block_size
        direct_num_pages = int(num_pages)
    else:
        assert block_table.shape[0] == batch_size
        page_source = block_table
        page_source_rows = seq_lens
        max_kv_len = block_table.shape[1] * block_size
        direct_num_pages = 1

    gqa_group_size = num_q_heads // num_kv_heads

    topk_idx = _normalize_topk_idx_for_gqa(
        topk_idx,
        num_q_heads,
        num_kv_heads,
        gqa_group_size,
    )

    max_topk = topk_idx.shape[2]

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if num_topk_chunks is None:
        num_topk_chunks = _choose_num_topk_chunks(
            batch_size,
            num_kv_heads,
            max_topk,
            max_num_topk_chunks=max_num_topk_chunks,
        )
    else:
        num_topk_chunks = int(num_topk_chunks)

    assert num_topk_chunks >= 1
    assert (num_topk_chunks & (num_topk_chunks - 1)) == 0
    assert num_topk_chunks <= max(1, max_topk)

    chunk_size_topk = (max_topk + num_topk_chunks - 1) // num_topk_chunks
    # Min static topk loop width of 2 (backend corner case at CHUNK_SIZE_T=1);
    # extra iterations masked by topk_pos < max_topk and logical_block >= 0.
    chunk_size_topk = max(2, chunk_size_topk)

    blocks_per_step = max(1, int(topk_blocks_per_step))
    assert (blocks_per_step & (blocks_per_step - 1)) == 0
    # When several topk blocks are fused per step, the static loop width must
    # cover at least one fused tile so every selected block is visited.
    chunk_size_topk = max(chunk_size_topk, blocks_per_step)

    # Single-chunk fast path: kernel writes already-final-normalized output, so
    # alias o_partial to the output buffer and skip the merge launch + temp alloc.
    single_chunk = num_topk_chunks == 1
    out = torch.empty_like(q)
    if single_chunk:
        o_partial = out.view(1, batch_size, num_q_heads, head_dim)
    else:
        o_partial = torch.empty(
            (num_topk_chunks, batch_size, num_q_heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
    # lse_partial is always written by the kernel; unused on the single-chunk path
    # but required as a store target (small, [C,B,QH]).
    lse_partial = torch.empty(
        (num_topk_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )

    # Triton type-checks pointer args in constexpr-dead branches on Ascend:
    # pass a typed tensor pointer, control the real behavior with HAS_SINK.
    sink_arg = sink if sink is not None else q

    grid = (batch_size * num_topk_chunks, num_kv_heads)
    _gqa_share_sparse_decode_bnsd_kernel[grid](
        q,
        sink_arg,
        k_cache_bnsd,
        v_cache_bnsd,
        page_source,
        page_source,
        page_source_rows,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        block_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink_arg.stride(0) if sink is not None else 0,
        sink_arg.stride(1) if sink is not None else 0,
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        v_cache_bnsd.stride(0),
        v_cache_bnsd.stride(1),
        v_cache_bnsd.stride(2),
        v_cache_bnsd.stride(3),
        page_source.stride(0),
        page_source.stride(1),
        req_to_token.stride(0) if use_direct_page_lookup else 0,
        req_to_token.stride(1) if use_direct_page_lookup else 0,
        req_to_token.shape[1] if use_direct_page_lookup else 1,
        direct_num_pages,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size * blocks_per_step,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        CHUNK_SIZE_T=chunk_size_topk,
        BLOCKS_PER_STEP=blocks_per_step,
        PREFETCH_IDX=prefetch_idx and blocks_per_step == 1,
        HAS_SINK=sink is not None,
        USE_DIRECT_PAGE_LOOKUP=use_direct_page_lookup,
        SANITIZE_PAGE_IDS=sanitize_page_ids,
        num_warps=_SPARSE_DECODE_NW if num_warps is None else num_warps,
        num_stages=_SPARSE_DECODE_NS if num_stages is None else num_stages,
    )

    if not single_chunk:
        merge_grid = (batch_size, num_q_heads)
        _merge_topk_attn_out_bnsd_kernel[merge_grid](
            o_partial,
            lse_partial,
            out,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            num_warps=_MERGE_NW,
            num_stages=_MERGE_NS,
        )

    return out

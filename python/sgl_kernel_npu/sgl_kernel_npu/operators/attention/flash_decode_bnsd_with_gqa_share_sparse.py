"""Public GQA-shared sparse decode attention with implementation selection."""

import logging
from functools import lru_cache

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "attention.flash_decode_bnsd_with_gqa_share_sparse"

logger = logging.getLogger(__name__)
_native_warned = False
_triton_fn = None
_torch_native_fn = None

# csrc/sparse_attention_score generates tilingKeys only for these two sizes;
# any other shape silently falls back to the D=128 tiling and produces wrong
# values, so ``auto`` keeps those shapes on Triton and ``ascendc`` rejects them.
_NATIVE_HEAD_DIM = 128
_NATIVE_BLOCK_SIZE = 128


@lru_cache(maxsize=1)
def _get_native_sparse_op():
    """Return the in-repo native op, or None when this build lacks it.

    ``npu_sparse_attention_score`` is compiled into A2/A3 builds only; a load
    failure counts as unavailable so ``auto`` can pick Triton.
    """
    try:
        import sgl_kernel_npu  # noqa: F401  (registers torch.ops.npu.*)

        return getattr(torch.ops.npu, "npu_sparse_attention_score", None)
    except Exception:
        return None


def _warn_native_unavailable():
    global _native_warned
    if _native_warned:
        return
    logger.warning(
        "The native npu_sparse_attention_score op could not be loaded "
        "(sgl_kernel_npu not installed or torch.ops.npu.npu_sparse_attention_score "
        "not registered). Falling back to the Triton split-K path."
    )
    _native_warned = True


@register_impl(OP, "auto")
def _auto_impl(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
):
    """Native AscendC when it can serve the request, else Triton.

    A capability route, not a fallback on failure.
    """
    native_ready = (
        use_native
        and sink is None
        and q.ndim == 3
        and q.shape[2] == _NATIVE_HEAD_DIM
        and block_size == _NATIVE_BLOCK_SIZE
        and _get_native_sparse_op() is not None
    )
    if native_ready:
        return get_impl_by_name(OP, "ascendc")(
            q=q,
            sink=sink,
            k_cache_bnsd=k_cache_bnsd,
            v_cache_bnsd=v_cache_bnsd,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            topk_idx=topk_idx,
            sm_scale=sm_scale,
            num_topk_chunks=num_topk_chunks,
            max_num_topk_chunks=max_num_topk_chunks,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            max_num_blocks=max_num_blocks,
            num_pages=num_pages,
            sanitize_page_ids=sanitize_page_ids,
            topk_blocks_per_step=topk_blocks_per_step,
            prefetch_idx=prefetch_idx,
            num_warps=num_warps,
            num_stages=num_stages,
            use_native=use_native,
        )
    if use_native and sink is None:
        _warn_native_unavailable()
    return get_impl_by_name(OP, "triton")(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache_bnsd,
        v_cache_bnsd=v_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
        num_topk_chunks=num_topk_chunks,
        max_num_topk_chunks=max_num_topk_chunks,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
        topk_blocks_per_step=topk_blocks_per_step,
        prefetch_idx=prefetch_idx,
        num_warps=num_warps,
        num_stages=num_stages,
        use_native=use_native,
    )


@register_impl(OP, "triton")
def _triton_impl(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.flash_decode_bnsd_with_gqa_share_sparse import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )

        _triton_fn = flash_decode_bnsd_with_gqa_share_sparse

    return _triton_fn(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache_bnsd,
        v_cache_bnsd=v_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
        num_topk_chunks=num_topk_chunks,
        max_num_topk_chunks=max_num_topk_chunks,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
        topk_blocks_per_step=topk_blocks_per_step,
        prefetch_idx=prefetch_idx,
        num_warps=num_warps,
        num_stages=num_stages,
        use_native=use_native,
    )


@register_impl(OP, "ascendc", supported_generations={Generation.ARCH22})
def _ascendc_impl(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
):
    """Native ``npu_sparse_attention_score`` adapter.

    Chunking, block fusion, prefetch and launch knobs tune Triton only; they are
    accepted and ignored.
    """
    op = _get_native_sparse_op()
    if op is None:
        raise RuntimeError(
            "impl='ascendc' requested, but torch.ops.npu.npu_sparse_attention_score "
            "is not registered in this build (A2/A3-only op); use impl='triton'"
        )

    # The four original contract asserts, then the two capability gates;
    # page-source asserts stay Triton-only (decision D4).
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape
    assert sink is None, (
        "impl='ascendc' does not support sink: the native op has no sink input; "
        f"got shape {tuple(sink.shape)}; use impl='triton'"
    )
    assert q.shape[2] == _NATIVE_HEAD_DIM and block_size == _NATIVE_BLOCK_SIZE, (
        "impl='ascendc' only generates tiling keys for head_dim="
        f"{_NATIVE_HEAD_DIM}, block_size={_NATIVE_BLOCK_SIZE}; got "
        f"head_dim={q.shape[2]}, block_size={block_size}; use impl='triton'"
    )

    # Shared with the Triton path; imported here so importing this module does
    # not require Triton.
    from ...implementation.triton.attention.flash_decode_bnsd_with_gqa_share_sparse import (
        _native_sanitize_topk_kernel,
        _normalize_topk_idx_for_gqa,
    )

    device = q.device
    batch_size = q.shape[0]
    head_dim = q.shape[2]
    num_kv_heads = k_cache_bnsd.shape[2]
    # Legacy skipped the native path for per-QH topk; normalize instead so the
    # native backend accepts the same input surface as Triton.
    topk_idx = _normalize_topk_idx_for_gqa(
        topk_idx, q.shape[1], num_kv_heads, q.shape[1] // num_kv_heads
    )

    if block_table is not None:
        block_table_i32 = block_table.to(torch.int32)
    else:
        # Gather only the B x max_num_blocks block-start slots, then -> page id.
        block_index = torch.arange(max_num_blocks, device=device, dtype=torch.int32)
        slots = req_to_token[
            req_pool_indices[:, None].to(torch.int64),
            (block_index * block_size)[None, :].to(torch.int64),
        ]
        block_table_i32 = (slots // block_size).to(torch.int32)
        if sanitize_page_ids and num_pages is not None:
            # Triton clamps the page id derived from the request map; mirror it
            # so a stale map cannot make the two backends diverge.
            block_table_i32 = block_table_i32.clamp(min=0, max=int(num_pages) - 1)

    select_idx = topk_idx.to(torch.int32).clone()
    select_num_idx = torch.empty(
        (num_kv_heads, batch_size), dtype=torch.int32, device=device
    )
    _native_sanitize_topk_kernel[(num_kv_heads, batch_size)](
        select_idx,
        select_num_idx,
        seq_lens.to(torch.int32),
        select_idx.stride(0),
        select_idx.stride(1),
        select_idx.stride(2),
        select_num_idx.stride(0),
        select_num_idx.stride(1),
        block_size=block_size,
        SLOTS=select_idx.shape[-1],
        num_warps=1,
        num_stages=1,
    )
    # -1 sentinels must not reach the native op (block_table[-1] -> OOB -> NaN).
    select_idx = torch.where(select_idx < 0, torch.zeros_like(select_idx), select_idx)
    return op(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        select_idx,
        block_table_i32,
        select_num_idx=select_num_idx,
        actual_seq_lengths=torch.ones(batch_size, dtype=torch.int32, device=device),
        actual_seq_lengths_kv=seq_lens.to(torch.int32),
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale if sm_scale is not None else head_dim**-0.5,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=4,
    )


@register_impl(OP, "torch_native")
def _torch_native_impl(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.flash_decode_bnsd_with_gqa_share_sparse import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )

        _torch_native_fn = flash_decode_bnsd_with_gqa_share_sparse

    # Only the semantic arguments are forwarded: chunking, block fusion,
    # prefetch, launch tuning and the native switch do not change the result.
    return _torch_native_fn(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache_bnsd,
        v_cache_bnsd=v_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
    )


@register_operator(OP)
def flash_decode_bnsd_with_gqa_share_sparse(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
    *,
    impl="auto",
):
    """Sparse decode attention over precomputed topk KV blocks, with GQA sharing.

    ``impl="auto"`` (default) uses the native AscendC op when the build has it
    and ``use_native`` is true, otherwise the Triton split-K entry; ``"triton"``
    and ``"ascendc"`` pin a backend, ``"torch_native"`` is an independent
    reference. ``q`` and the caches are fp16/bf16, ``topk_idx`` is
    ``[KVH or QH, B, topk]`` with ``-1`` for empty slots, and the page source is
    ``block_table`` or ``req_to_token`` + ``req_pool_indices``. Returns a new
    ``[B, QH, D]`` tensor in ``q.dtype``; no input is modified.
    """
    return get_impl_by_name(OP, impl)(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache_bnsd,
        v_cache_bnsd=v_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
        num_topk_chunks=num_topk_chunks,
        max_num_topk_chunks=max_num_topk_chunks,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
        topk_blocks_per_step=topk_blocks_per_step,
        prefetch_idx=prefetch_idx,
        num_warps=num_warps,
        num_stages=num_stages,
        use_native=use_native,
    )

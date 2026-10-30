"""Public decode BNSD score/topk operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "indexer.flash_decode_bnsd_with_topk_idx"

_triton_fn = None
_torch_native_fn = None


# Importing this module registers the three names without importing Triton.
@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    max_seqlen,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    score_type="max",
    disable_index_value=False,
    num_kv_chunks=None,
    max_num_kv_chunks=8,
    use_triton_topk=None,
    num_topk_chunks=None,
    use_chunked_score=True,
    score_blocks_per_chunk=16,
    score_max_chunks=32,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    packed_seq_lens=False,
    runtime_fill_only=False,
    runtime_score_short_max_blocks=0,
    runtime_score_short_chunks=0,
    fused_append_local=False,
    use_native=True,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.flash_decode_bnsd_with_topk_idx import (
            flash_decode_bnsd_with_topk_idx,
        )

        _triton_fn = flash_decode_bnsd_with_topk_idx

    # use_native / use_triton_topk / num_topk_chunks / use_chunked_score are
    # accepted for call-site compatibility and intentionally absorbed here.
    return _triton_fn(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache_bnsd,
        v_cache_bnsd=v_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        num_kv_chunks=num_kv_chunks,
        max_num_kv_chunks=max_num_kv_chunks,
        score_blocks_per_chunk=score_blocks_per_chunk,
        score_max_chunks=score_max_chunks,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
        packed_seq_lens=packed_seq_lens,
        runtime_fill_only=runtime_fill_only,
        runtime_score_short_max_blocks=runtime_score_short_max_blocks,
        runtime_score_short_chunks=runtime_score_short_chunks,
        fused_append_local=fused_append_local,
    )


@register_impl(OP, "torch_native")
def _torch_native_impl(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    max_seqlen,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    score_type="max",
    disable_index_value=False,
    num_kv_chunks=None,
    max_num_kv_chunks=8,
    use_triton_topk=None,
    num_topk_chunks=None,
    use_chunked_score=True,
    score_blocks_per_chunk=16,
    score_max_chunks=32,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    packed_seq_lens=False,
    runtime_fill_only=False,
    runtime_score_short_max_blocks=0,
    runtime_score_short_chunks=0,
    fused_append_local=False,
    use_native=True,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.flash_decode_bnsd_with_topk_idx import (
            flash_decode_bnsd_with_topk_idx,
        )

        _torch_native_fn = flash_decode_bnsd_with_topk_idx

    # The reference covers the score-only branch only: it supports max/lse,
    # forced blocks, both page sources, packed/sanitize and fused append, and
    # rejects the attention branch. It absorbs sink/v_cache, chunking,
    # graph-replay hints and compatibility flags.
    return _torch_native_fn(
        q=q,
        k_cache_bnsd=k_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
        packed_seq_lens=packed_seq_lens,
        fused_append_local=fused_append_local,
    )


@register_operator(OP)
def flash_decode_bnsd_with_topk_idx(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    max_seqlen,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    score_type="max",
    disable_index_value=False,
    num_kv_chunks=None,
    max_num_kv_chunks=8,
    use_triton_topk=None,
    num_topk_chunks=None,
    use_chunked_score=True,
    score_blocks_per_chunk=16,
    score_max_chunks=32,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    packed_seq_lens=False,
    runtime_fill_only=False,
    runtime_score_short_max_blocks=0,
    runtime_score_short_chunks=0,
    fused_append_local=False,
    use_native=True,
    *,
    impl="auto",
):
    """Decode over a BNSD block cache and return block-level topk indices.

    The default ``impl="auto"`` resolves to the Triton entry; ``"auto"`` and
    ``"triton"`` are registered to the same function, so they are equivalent.
    ``"torch_native"`` is an independent PyTorch reference used for accuracy
    comparison; it covers the score-only branch (``disable_index_value=True``)
    only, including ``score_type`` ``"max"``/``"lse"``, ``init_blocks``/
    ``local_blocks`` forcing, the direct page source and ``block_table``,
    ``packed_seq_lens``, ``sanitize_page_ids`` and ``fused_append_local``; the
    attention branch (``disable_index_value=False``) is rejected.

    Branches (mutually exclusive):

    * ``disable_index_value=True`` and ``topk<=0``: no indices are produced and
      ``topk_idx`` is empty with shape ``[QH, B, 0]``.
    * ``disable_index_value=True`` and ``topk>0``: score-only path. One score is
      computed per KV block, then the ``topk`` best blocks are returned; ``o``
      is ``None``.
    * ``disable_index_value=False``: full attention path, returning ``o`` and
      ``topk_idx``. It requires a materialized ``block_table``. Triton only:
      the ``torch_native`` reference rejects this branch and it is outside the
      validated scope.

    Args:
        q (torch.Tensor): Queries ``[B, QH, D]`` in ``torch.float16`` or
            ``torch.bfloat16``, on the NPU.
        sink (torch.Tensor | None): Optional attention sink ``[QH, D]`` with the
            same dtype and device as ``q``. Only the full attention path uses it:
            it adds ``q @ sink * sm_scale`` to the softmax denominator and never
            contributes a value or changes the topk scores. The score-only path
            ignores it, and production decode passes ``None``.
        k_cache_bnsd (torch.Tensor): Key cache
            ``[num_blocks, block_size, KVH, D]`` with the dtype of ``q``.
        v_cache_bnsd (torch.Tensor | None): Value cache with the same shape and
            dtype as ``k_cache_bnsd``; required only when
            ``disable_index_value=False``.
        block_table (torch.Tensor | None): ``[B, max_num_blocks]`` logical to
            physical block map, ``int32``/``int64``. Required unless the direct
            request map (``req_to_token`` + ``req_pool_indices``) is used; the
            direct map is score-only and requires ``block_table is None``.
            The full attention path supports only ``block_table``. Its column
            count must cover the actual block count (not asserted).
        seq_lens (torch.Tensor): Causal KV lengths, ``int32``/``int64``.
            ``[B]`` normally, or ``[B * QH // KVH]`` when
            ``packed_seq_lens=True`` (row ``b*G+h`` for batch ``b`` and GQA slot
            ``h``; score-only only, requires ``num_kv_heads == 1``).
        max_seqlen (int): Upper bound of the KV length used to size the score
            chunk grid; it does not change the selected blocks when scores are
            distinct.
        block_size (int): Tokens per KV block; must match the cache and be a
            power of two.
        topk (int): Number of block indices to return.
        init_blocks (int): Leading blocks whose candidate score is forced to
            ``1e30``.
        local_blocks (int): Trailing blocks whose candidate score is forced to
            ``1e29``; ``local_blocks`` wins over ``init_blocks``. Forcing only
            rewrites the topk candidate scores, never the attention softmax, and
            is skipped by the pass-through case.
        sm_scale (float | None): Softmax scale; ``D ** -0.5`` when ``None``.
        score_type (str): ``"max"`` or ``"lse"``, the per-block token reduction.
        disable_index_value (bool): ``True`` selects the score-only path,
            ``False`` the full attention path.
        num_kv_chunks (int | None): KV chunks per row on the full attention path;
            chosen from the vector-core count when ``None``.
        max_num_kv_chunks (int): Power-of-two cap of the automatic chunk choice.
        use_triton_topk (bool | None): Retained for call-site compatibility; no
            effect.
        num_topk_chunks (int | None): Retained for call-site compatibility; no
            effect.
        use_chunked_score (bool): Retained for call-site compatibility; no
            effect.
        score_blocks_per_chunk (int): Target block tiles per score program.
        score_max_chunks (int): Upper bound of the score chunk count.
        req_to_token (torch.Tensor | None): ``[num_requests, max_context]``
            direct page map holding the token slot at each position; requires
            ``req_pool_indices``.
        req_pool_indices (torch.Tensor | None): ``[B]`` request rows into
            ``req_to_token``.
        max_num_blocks (int | None): Page-table capacity. With a direct page map
            it is always required and must satisfy
            ``max_num_blocks * block_size <= req_to_token.shape[1]``; the check
            runs before the empty-topk branch. With ``block_table`` it is
            required only for the score-only path with
            ``fused_append_local=True``.
        num_pages (int | None): Physical page count; required by the direct page
            map.
        sanitize_page_ids (bool): Clamp direct-map physical block ids to
            ``[0, num_pages-1]``; ignored with ``block_table``.
        packed_seq_lens (bool): Interpret ``seq_lens`` as ``[B * QH // KVH]``
            per-row lengths (draft-token verify); score-only only and requires
            ``num_kv_heads == 1``.
        runtime_fill_only (bool): Let the score kernel pick the direct-fill
            candidate path from the runtime chunk length.
        runtime_score_short_max_blocks (int): Block-count threshold of the
            short-context adaptive chunking.
        runtime_score_short_chunks (int): Chunk count used below that threshold.
        fused_append_local (bool): Fuse candidate merge with the causal local
            block append; branch 3 with ``topk>0`` then returns
            ``[QH, B, topk+1]`` with the local block, or ``-1`` when it was
            already selected, in the last slot.
        use_native (bool): Retained for call-site compatibility; no effect,
            because the native backend is not registered.
        impl (str): Implementation name: ``"auto"``, ``"triton"`` or
            ``"torch_native"``. Defaults to ``"auto"``.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor]: ``o`` is ``None`` on the
        score-only paths and ``[B, QH, D]`` in ``q.dtype`` on the full attention
        path. ``topk_idx`` is ``int32`` with shape ``[QH, B, topk]``, or
        ``[QH, B, topk+1]`` for branch 3 with ``topk>0`` and
        ``fused_append_local=True``. ``-1`` marks an invalid, out-of-range or
        de-duplicated slot; once the valid candidates are exhausted block 0 is
        repeated. The order of equally scored blocks is undefined.
    """
    impl_fn = get_impl_by_name(OP, impl)
    return impl_fn(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache_bnsd,
        v_cache_bnsd=v_cache_bnsd,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        num_kv_chunks=num_kv_chunks,
        max_num_kv_chunks=max_num_kv_chunks,
        use_triton_topk=use_triton_topk,
        num_topk_chunks=num_topk_chunks,
        use_chunked_score=use_chunked_score,
        score_blocks_per_chunk=score_blocks_per_chunk,
        score_max_chunks=score_max_chunks,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=max_num_blocks,
        num_pages=num_pages,
        sanitize_page_ids=sanitize_page_ids,
        packed_seq_lens=packed_seq_lens,
        runtime_fill_only=runtime_fill_only,
        runtime_score_short_max_blocks=runtime_score_short_max_blocks,
        runtime_score_short_chunks=runtime_score_short_chunks,
        fused_append_local=fused_append_local,
        use_native=use_native,
    )

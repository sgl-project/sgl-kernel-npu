"""Triton implementation of kpool_write_tail_and_maybe_compress."""

from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties

INDEX_HEAD_DIM = 128
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _kpool_write_tail_and_maybe_compress_kernel(
    key_ptr,  # Draft keys: [BATCH_SIZE * N, HEAD_DIM].
    score_ptr,  # Scores paired with key_ptr.
    tail_k_ptr,  # Per-request BF16 ring buffer: [REQS, TAIL_SIZE, HEAD_DIM].
    tail_score_ptr,  # Scores paired with tail_k_ptr.
    ape_ptr,  # Additive pooling position scores: [POOL_SIZE, HEAD_DIM].
    req_pool_indices_ptr,  # Request/ring-buffer row for each batch item.
    write_start_ptr,  # Logical tail position of the first draft token.
    tail_logical_start_ptr,  # Logical start of the pool to compress.
    write_loc_ptr,  # Cache rows: [BATCH_SIZE, MAX_CLOSED_POOLS].
    out_cache_loc_ptr,  # Token cache locations; zero marks a padded batch.
    effective_n_ptr,  # Optional valid draft-token count per batch item.
    cache_ptr,  # Flattened index cache: [CACHE_ROWS, HEAD_DIM].
    scale_ptr,  # Per-cache-row scales: FP16 for INT8, uint8 E8M0 for mxFP8.
    batch_size,
    key_stride_0,
    score_stride_0,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    ape_stride_0,
    write_loc_stride_0,
    cache_stride_0,
    N: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
    HAS_EFFECTIVE_N: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_MXSCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    # Programs consume multiple batch rows when the logical batch is larger
    # than the physical Vector Core count.
    for batch in range(pid, batch_size, program_count):
        cache_loc_0 = tl.load(out_cache_loc_ptr + batch * N)
        if cache_loc_0 != 0:
            req = tl.load(req_pool_indices_ptr + batch)
            write_start = tl.load(write_start_ptr + batch)

            # Tail writes always cover all N graph rows. effective_n only gates
            # compression, matching the CUDA target-verify kernel contract.
            for draft_offset in tl.static_range(0, N):
                draft_row = batch * N + draft_offset
                draft_key = tl.load(
                    key_ptr + draft_row * key_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                )
                draft_score = tl.load(
                    score_ptr + draft_row * score_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                )
                draft_tail_slot = (write_start + draft_offset) % TAIL_SIZE
                draft_tail_k_offset = (
                    req * tail_k_stride_0 + draft_tail_slot * tail_k_stride_1 + offs
                )
                draft_tail_score_offset = (
                    req * tail_score_stride_0
                    + draft_tail_slot * tail_score_stride_1
                    + offs
                )
                tl.store(
                    tail_k_ptr + draft_tail_k_offset,
                    draft_key,
                    mask=dim_mask,
                )
                tl.store(
                    tail_score_ptr + draft_tail_score_offset,
                    draft_score,
                    mask=dim_mask,
                )

            if HAS_EFFECTIVE_N:
                effective_n = tl.load(effective_n_ptr + batch).to(tl.int32)
            else:
                effective_n = N
            effective_n = tl.minimum(tl.maximum(effective_n, 0), N)

            base_pool = write_start // POOL_SIZE
            completed_pools = (write_start + effective_n) // POOL_SIZE - base_pool
            completed_pools = tl.minimum(
                tl.maximum(completed_pools, 0), MAX_CLOSED_POOLS
            )
            tail_logical_start = tl.load(tail_logical_start_ptr + batch)

            # write_loc[b, p] maps the p-th candidate pool to its cache row.
            # Only the runtime-completed prefix is reduced and written.
            for pool_offset in tl.static_range(0, MAX_CLOSED_POOLS):
                if pool_offset < completed_pools:
                    pool_logical_start = tail_logical_start + pool_offset * POOL_SIZE
                    running_max = tl.full((BLOCK_D,), -float("inf"), tl.float32)
                    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
                    denom = tl.full((BLOCK_D,), 0.0, tl.float32)

                    for slot in tl.static_range(0, POOL_SIZE):
                        pool_tail_slot = (pool_logical_start + slot) % TAIL_SIZE
                        pool_tail_k_offset = (
                            req * tail_k_stride_0
                            + pool_tail_slot * tail_k_stride_1
                            + offs
                        )
                        pool_tail_score_offset = (
                            req * tail_score_stride_0
                            + pool_tail_slot * tail_score_stride_1
                            + offs
                        )
                        pool_key = tl.load(
                            tail_k_ptr + pool_tail_k_offset,
                            mask=dim_mask,
                            other=0.0,
                        ).to(tl.float32)
                        pool_score = tl.load(
                            tail_score_ptr + pool_tail_score_offset,
                            mask=dim_mask,
                            other=0.0,
                        ).to(tl.float32)
                        pool_score += tl.load(
                            ape_ptr + slot * ape_stride_0 + offs,
                            mask=dim_mask,
                            other=0.0,
                        ).to(tl.float32)

                        new_max = tl.maximum(running_max, pool_score)
                        rescale = tl.exp(running_max - new_max)
                        prob = tl.exp(pool_score - new_max)
                        denom = denom * rescale + prob
                        acc = acc * rescale + pool_key * prob
                        running_max = new_max

                    pooled = acc / denom
                    write_loc = tl.load(
                        write_loc_ptr + batch * write_loc_stride_0 + pool_offset
                    )
                    cache_offsets = write_loc * cache_stride_0 + offs
                    if HAS_SCALE:
                        amax = tl.max(tl.abs(pooled), axis=0)
                        scale = tl.where(amax > 0.0, amax / 127.0, 1.0)
                        scaled = pooled / scale
                        rounded = scaled + tl.where(scaled >= 0.0, 0.5, -0.5)
                        quantized = tl.clamp(rounded, -127.0, 127.0).to(tl.int8)
                        tl.store(cache_ptr + cache_offsets, quantized, mask=dim_mask)
                        tl.store(scale_ptr + write_loc, scale.to(tl.float16))
                    elif HAS_MXSCALE:
                        blocks = tl.reshape(pooled, (HEAD_DIM // 32, 32))
                        group_max = tl.max(tl.abs(blocks), axis=1)
                        finite = (group_max == group_max) & (group_max != float("inf"))
                        nonzero = group_max > 0.0
                        magnitude = tl.where(nonzero & finite, group_max, 1.0)
                        exponent = tl.floor(tl.log2(magnitude))
                        exponent = tl.where(
                            tl.exp2(exponent) > magnitude, exponent - 1.0, exponent
                        )
                        exponent = tl.where(
                            tl.exp2(exponent + 1.0) <= magnitude,
                            exponent + 1.0,
                            exponent,
                        )
                        scale_bytes = tl.maximum(exponent - 8.0 + 127.0, 0.0)
                        scale_bytes = tl.where(nonzero, scale_bytes, 0.0)
                        scale_bytes = tl.where(finite, scale_bytes, 255.0)
                        scale = tl.exp2(scale_bytes - 127.0)
                        scaled = tl.clamp(blocks / scale[:, None], -448.0, 448.0)
                        scaled = tl.where(finite[:, None], scaled, float("nan"))
                        payload = tl.reshape(scaled, (BLOCK_D,)).to(tl.float8e4nv)
                        tl.store(cache_ptr + cache_offsets, payload, mask=dim_mask)
                        scale_offsets = write_loc * (HEAD_DIM // 64 * 2) + tl.arange(
                            0, HEAD_DIM // 64 * 2
                        )
                        tl.store(scale_ptr + scale_offsets, scale_bytes.to(tl.uint8))
                    else:
                        tl.store(
                            cache_ptr + cache_offsets,
                            pooled.to(tl.bfloat16),
                            mask=dim_mask,
                        )


def kpool_write_tail_and_maybe_compress(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    *,
    scale_cache: Optional[torch.Tensor] = None,
    num_programs: Optional[int] = None,
) -> None:
    """Write draft tokens to tail rings and compress newly completed pools.

    The input rows are grouped as ``[batch, num_draft_tokens, 128]``. For each
    valid batch item, all draft rows are written to its request-local tail ring.
    Every pool boundary crossed by ``write_start + effective_n`` is reduced
    with a dimension-wise softmax over ``tail_score + ape`` and written to the
    corresponding index-cache row in ``write_loc``.

    Args:
        index_k_cache: Contiguous cache flattenable to ``[cache_rows, 128]``.
            BF16 stores the pooled key; INT8 requires ``scale_cache`` and
            stores the symmetric INT8 key; mxFP8 (``float8_e4m3fn``) requires
            ``scale_cache`` and stores E4M3FN keys with per-32-channel E8M0
            scales. Updated in place.
        key: BF16 draft keys with shape ``[batch * N, 128]``.
        score: Draft scores with the same shape as ``key`` and dtype FP16,
            BF16, or FP32.
        tail_k: Contiguous BF16 tail rings with shape
            ``[R, pool_size + N, 128]``. Updated in place for valid batches.
        tail_score: Contiguous score tail rings matching ``tail_k``. Updated in
            place for valid batch items.
        ape: FP32 additive pooling position scores with shape
            ``[pool_size, 128]``.
        req_pool_indices: Request row in the tail buffers for each batch item.
        write_start: Logical tail position where each batch starts writing.
        tail_logical_start: Logical start of the candidate completed pool.
        write_loc: Candidate cache rows with shape
            ``[batch, ceil(num_draft_tokens / pool_size)]``.
        out_cache_loc: Token cache locations with at least ``batch * N``
            entries. A zero first entry skips the entire batch item.
        num_draft_tokens: Static number ``N`` of draft rows per batch item.
        effective_n_per_batch: Optional effective draft count used only to
            decide whether a pool was completed. Tail writes still cover ``N``.
        scale_cache: Optional dequantization scales per cache row. Required for
            INT8 (FP16, one element per row) and mxFP8 (uint8 E8M0 bytes shaped
            ``[cache_rows, 128/64, 2]``).
        num_programs: Optional Triton grid size. Defaults to the NPU Vector Core
            count and uses a grid-stride loop for larger batches.

    Raises:
        AssertionError: If an input violates the kernel layout contract.
        ValueError: If ``num_programs`` is not positive.
    """

    assert num_draft_tokens > 0
    assert key.ndim == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert score.shape == key.shape
    assert key.dtype == torch.bfloat16
    assert score.dtype in KPOOL_SCORE_DTYPES
    assert tail_k.ndim == 3 and tail_k.shape[2] == INDEX_HEAD_DIM
    assert tail_score.shape == tail_k.shape
    assert tail_k.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.ndim == 2 and ape.shape[1] == INDEX_HEAD_DIM
    assert ape.dtype == torch.float32
    assert index_k_cache.dtype in (torch.bfloat16, torch.int8, torch.float8_e4m3fn)
    assert (scale_cache is not None) == (
        index_k_cache.dtype in (torch.int8, torch.float8_e4m3fn)
    )
    if scale_cache is not None:
        if index_k_cache.dtype == torch.int8:
            assert scale_cache.dtype == torch.float16
        else:
            assert scale_cache.dtype == torch.uint8
        assert scale_cache.is_contiguous()
        cache_rows = index_k_cache.numel() // INDEX_HEAD_DIM
        if index_k_cache.dtype == torch.int8:
            assert scale_cache.numel() == cache_rows
        else:
            assert scale_cache.numel() == cache_rows * (INDEX_HEAD_DIM // 64) * 2
    assert index_k_cache.numel() % INDEX_HEAD_DIM == 0

    total_rows = key.shape[0]
    if total_rows == 0:
        return
    assert total_rows % num_draft_tokens == 0
    batch_size = total_rows // num_draft_tokens
    pool_size = int(ape.shape[0])
    tail_size = int(tail_k.shape[1])
    assert pool_size > 0
    max_closed_pools = (num_draft_tokens + pool_size - 1) // pool_size
    assert tail_size == pool_size + num_draft_tokens

    batch_metadata = (
        req_pool_indices,
        write_start,
        tail_logical_start,
    )
    assert all(t.ndim == 1 and t.numel() >= batch_size for t in batch_metadata)
    assert write_loc.shape == (batch_size, max_closed_pools), write_loc.shape
    assert write_loc.stride(1) == 1, write_loc.stride()
    assert out_cache_loc.ndim == 1 and out_cache_loc.numel() >= total_rows
    assert req_pool_indices.dtype == torch.int64
    assert write_start.dtype in (torch.int32, torch.int64)
    assert tail_logical_start.dtype in (torch.int32, torch.int64)
    assert write_loc.dtype == torch.int64
    assert out_cache_loc.dtype in (torch.int32, torch.int64)
    if effective_n_per_batch is not None:
        assert effective_n_per_batch.ndim == 1
        assert effective_n_per_batch.numel() >= batch_size
        assert effective_n_per_batch.dtype in (torch.int32, torch.int64)

    # Normalize read-only inputs that may be views
    key = key.contiguous()
    score = score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    write_start = write_start.contiguous()
    tail_logical_start = tail_logical_start.contiguous()
    out_cache_loc = out_cache_loc.contiguous()
    if effective_n_per_batch is not None:
        effective_n_per_batch = effective_n_per_batch.contiguous()

    inputs = (
        index_k_cache,
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
    )
    if effective_n_per_batch is not None:
        inputs += (effective_n_per_batch,)
    if scale_cache is not None:
        inputs += (scale_cache,)
    assert all(t.device == index_k_cache.device for t in inputs)
    assert all(t.is_contiguous() for t in inputs)

    if num_programs is None:
        num_programs = min(batch_size, get_device_properties()[1])
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")
    num_programs = min(num_programs, batch_size)

    if effective_n_per_batch is None:
        effective_n_per_batch = torch.empty((1,), dtype=torch.int32, device=key.device)
        has_effective_n = False
    else:
        has_effective_n = True

    cache_2d = index_k_cache.view(-1, INDEX_HEAD_DIM)
    scale_1d = (
        scale_cache.view(-1)
        if scale_cache is not None
        else torch.empty((1,), dtype=torch.float16, device=key.device)
    )
    _kpool_write_tail_and_maybe_compress_kernel[(num_programs,)](
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        effective_n_per_batch,
        cache_2d,
        scale_1d,
        batch_size,
        key.stride(0),
        score.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        ape.stride(0),
        write_loc.stride(0),
        cache_2d.stride(0),
        N=num_draft_tokens,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=INDEX_HEAD_DIM,
        MAX_CLOSED_POOLS=max_closed_pools,
        HAS_EFFECTIVE_N=has_effective_n,
        HAS_SCALE=index_k_cache.dtype == torch.int8,
        HAS_MXSCALE=index_k_cache.dtype == torch.float8_e4m3fn,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )

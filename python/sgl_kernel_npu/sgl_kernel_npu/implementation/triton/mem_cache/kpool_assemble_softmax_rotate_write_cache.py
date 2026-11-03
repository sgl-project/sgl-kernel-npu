"""Triton implementation of kpool_assemble_softmax_rotate_write_cache."""

from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties

INDEX_HEAD_DIM = 128
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_kernel(
    cache_ptr,  # Flattened BF16 cache: [CACHE_ROWS, HEAD_DIM].
    chunk_k_ptr,  # Current extend keys: [CHUNK_TOKENS, HEAD_DIM].
    chunk_score_ptr,  # Scores paired with chunk_k_ptr.
    tail_k_ptr,  # Per-request BF16 ring buffer: [REQS, TAIL_SIZE, HEAD_DIM].
    tail_score_ptr,  # Scores paired with tail_k_ptr.
    req_pool_idx_ptr,  # Request/ring-buffer row for each output pool.
    n_from_tail_ptr,  # Number of leading pool slots sourced from the tail.
    chunk_src_start_ptr,  # First chunk row used by each output pool.
    tail_logical_base_ptr,  # Logical start in the request tail ring.
    ape_ptr,  # Additive pooling position scores: [POOL_SIZE, HEAD_DIM].
    loc_ptr,  # Flattened cache row written by each output pool.
    write_mask_ptr,  # Optional per-pool write mask.
    n_pools,  # Number of entries in every plan tensor above.
    # Only leading strides are passed; HEAD_DIM is contiguous by contract.
    cache_stride_0,
    chunk_k_stride_0,
    chunk_score_stride_0,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    ape_stride_0,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Cap the launch grid at the Vector Core count. Programs consume any
    # remaining pools through this grid-stride loop.
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    head_mask = offs < HEAD_DIM

    for row in range(pid, n_pools, program_count):
        active = True
        if HAS_WRITE_MASK:
            active = tl.load(write_mask_ptr + row)

        if active:
            # Load the plan entry that describes how this pool is assembled.
            n_tail = tl.load(n_from_tail_ptr + row)
            req = tl.load(req_pool_idx_ptr + row)
            chunk_src = tl.load(chunk_src_start_ptr + row)
            tail_base = tl.load(tail_logical_base_ptr + row)

            # Dimension-wise online softmax keeps the reduction numerically
            # stable without loading the complete pool into local memory.
            running_max = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)

            for slot in tl.static_range(0, POOL_SIZE):
                # A pool consists of a tail prefix followed by current chunk
                # rows. Tail positions wrap around the request ring buffer.
                if slot < n_tail:
                    tail_slot = (tail_base + slot) % TAIL_SIZE
                    k_offset = (
                        req * tail_k_stride_0 + tail_slot * tail_k_stride_1 + offs
                    )
                    score_offset = (
                        req * tail_score_stride_0
                        + tail_slot * tail_score_stride_1
                        + offs
                    )
                    key = tl.load(tail_k_ptr + k_offset, mask=head_mask, other=0.0).to(
                        tl.float32
                    )
                    score = tl.load(
                        tail_score_ptr + score_offset,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    chunk_row = chunk_src + slot - n_tail
                    k_offset = chunk_row * chunk_k_stride_0 + offs
                    score_offset = chunk_row * chunk_score_stride_0 + offs
                    key = tl.load(chunk_k_ptr + k_offset, mask=head_mask, other=0.0).to(
                        tl.float32
                    )
                    score = tl.load(
                        chunk_score_ptr + score_offset,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)

                score += tl.load(
                    ape_ptr + slot * ape_stride_0 + offs,
                    mask=head_mask,
                    other=0.0,
                ).to(tl.float32)
                new_max = tl.maximum(running_max, score)
                rescale = tl.exp(running_max - new_max)
                prob = tl.exp(score - new_max)
                denom = denom * rescale + prob
                acc = acc * rescale + key * prob
                running_max = new_max

            # NPU index cache stores raw BF16 keys: no Hadamard, FP8, or scale.
            pooled = (acc / denom).to(tl.bfloat16)

            write_loc = tl.load(loc_ptr + row)
            out_offsets = write_loc * cache_stride_0 + offs
            tl.store(cache_ptr + out_offsets, pooled, mask=head_mask)


def kpool_assemble_softmax_rotate_write_cache(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
) -> None:
    """Assemble pooled BF16 keys and write them to the NPU index cache.

    Each plan row assembles one pool. Its first ``n_from_tail[row]`` slots are
    read from the request-local tail ring; the remaining slots are read from
    ``chunk_k`` starting at ``chunk_src_start[row]``. A dimension-wise softmax
    over ``score + ape`` produces one BF16 key, which is written in place to
    flattened cache row ``loc[row]``. Dynamic inputs produced by the current
    forward are materialized as contiguous tensors before the kernel launch.

    Args:
        index_k_cache: Contiguous BF16 index cache. Its storage must be
            flattenable to ``[cache_rows, 128]``.
        chunk_k: BF16 current-chunk keys with shape ``[N, 128]``.
        chunk_score: Scores for ``chunk_k`` with shape ``[N, 128]`` and dtype
            FP16, BF16, or FP32.
        tail_k: Contiguous BF16 request tail rings with shape ``[R, T, 128]``.
        tail_score: Scores for ``tail_k`` with shape ``[R, T, 128]`` and dtype
            FP16, BF16, or FP32.
        req_pool_idx: Int64 tensor of shape ``[n_pools]`` selecting the request
            row in ``tail_k`` for each pool.
        n_from_tail: Int32 or int64 tensor of shape ``[n_pools]`` containing
            the number of leading slots read from the tail ring.
        chunk_src_start: Int64 tensor of shape ``[n_pools]`` containing the
            first row in ``chunk_k`` for each pool's chunk suffix.
        tail_logical_base: Int32 or int64 tensor of shape ``[n_pools]`` giving
            the logical first tail slot; physical slots wrap by ``T``.
        ape: FP32 additive pooling position scores with shape
            ``[pool_size, 128]``.
        loc: Int64 tensor of shape ``[n_pools]`` containing flattened cache
            rows to update.
        write_mask: Optional bool tensor of shape ``[n_pools]``. False entries
            leave their corresponding cache rows unchanged.
        num_programs: Optional Triton grid size. By default it is capped at the
            NPU Vector Core count; each program handles additional pools with
            a grid-stride loop.

    Raises:
        AssertionError: If an input violates the kernel layout contract.
        ValueError: If ``num_programs`` is not positive.
    """

    # These assertions protect the dtype and layout assumptions used by the
    # kernel's pointer arithmetic. Plan value ranges are guaranteed upstream.
    assert index_k_cache.dtype == torch.bfloat16
    assert chunk_k.dtype == torch.bfloat16 and tail_k.dtype == torch.bfloat16
    assert chunk_score.dtype in KPOOL_SCORE_DTYPES
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.dtype == torch.float32
    assert chunk_k.ndim == 2 and chunk_k.shape[1] == INDEX_HEAD_DIM
    assert chunk_score.shape == chunk_k.shape
    assert tail_k.ndim == 3 and tail_k.shape[2] == INDEX_HEAD_DIM
    assert tail_score.shape == tail_k.shape
    assert ape.ndim == 2 and ape.shape[1] == INDEX_HEAD_DIM
    assert index_k_cache.numel() % INDEX_HEAD_DIM == 0

    n_pools = req_pool_idx.numel()
    pool_size = int(ape.shape[0])
    tail_size = int(tail_k.shape[1])
    assert pool_size > 0 and tail_size > 0

    plan_tensors = (
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        loc,
    )
    assert all(t.ndim == 1 and t.numel() == n_pools for t in plan_tensors)
    assert req_pool_idx.dtype == torch.int64
    assert chunk_src_start.dtype == torch.int64 and loc.dtype == torch.int64
    assert n_from_tail.dtype in (torch.int32, torch.int64)
    assert tail_logical_base.dtype in (torch.int32, torch.int64)
    if write_mask is not None:
        assert write_mask.dtype == torch.bool and write_mask.shape == (n_pools,)

    # Match the GPU wrapper contract for dynamic, read-only forward inputs.
    # Calling contiguous() is a no-op when the input already has this layout.
    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is not None:
        write_mask = write_mask.contiguous()

    inputs = (
        index_k_cache,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
    )
    if write_mask is not None:
        inputs += (write_mask,)
    assert all(t.device == index_k_cache.device for t in inputs)
    assert all(t.is_contiguous() for t in inputs)

    if n_pools == 0:
        return

    if num_programs is None:
        num_programs = min(n_pools, get_device_properties()[1])
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")
    num_programs = min(num_programs, n_pools)

    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        has_write_mask = True

    cache_2d = index_k_cache.view(-1, INDEX_HEAD_DIM)
    _kpool_assemble_softmax_rotate_write_cache_kernel[(num_programs,)](
        cache_2d,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        n_pools,
        cache_2d.stride(0),
        chunk_k.stride(0),
        chunk_score.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        ape.stride(0),
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=INDEX_HEAD_DIM,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )

"""Triton implementation of scatter_kpool_tail_updates."""

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties

INDEX_HEAD_DIM = 128


@triton.jit
def _scatter_kpool_tail_updates_kernel(
    # Chunk tensor pointers.
    chunk_k_ptr,
    chunk_score_ptr,
    # Tail tensor pointers (updated in place).
    tail_k_ptr,
    tail_score_ptr,
    # Metadata pointers.
    req_pool_idx_ptr,
    dst_logical_start_ptr,
    chunk_src_start_ptr,
    n_write_ptr,
    # Stride parameters.
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    # Shape parameters.
    n_rows,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """Scatter chunk K/score rows into tail rings on NPU.

    The fixed (NUM_CORES,) grid uses a grid-stride loop over rows, with an
    inner loop over n_write[row] slots. Each work item loads one 128-element
    BF16 K/score row from chunk and writes it to the tail ring.
    """
    pid = tl.program_id(0)

    # Each core processes rows with a grid-stride loop.
    for row in range(pid, n_rows, NUM_CORES):
        # Load this row's metadata once.
        n_w = tl.load(n_write_ptr + row).to(tl.int32)
        req = tl.load(req_pool_idx_ptr + row).to(tl.int64)
        dst_start = tl.load(dst_logical_start_ptr + row).to(tl.int32)
        src_base = tl.load(chunk_src_start_ptr + row).to(tl.int64)

        # Iterate over the slots written by this row.
        for slot in range(n_w):
            # Compute the chunk source row and physical tail slot (ring modulo).
            src_row = src_base + slot.to(tl.int64)
            phys_slot = (dst_start + slot) % TAIL_SIZE

            # Compute the offset in chunk.
            src_offset = src_row * chunk_stride_0.to(tl.int64)

            # Load chunk_k[src_row, :] with a block pointer.
            k_block = tl.make_block_ptr(
                base=chunk_k_ptr + src_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            k = tl.load(k_block)  # [HEAD_DIM] BF16 full-row DMA

            # Load chunk_score[src_row, :] with a block pointer.
            s_block = tl.make_block_ptr(
                base=chunk_score_ptr + src_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            s = tl.load(s_block)  # [HEAD_DIM] BF16 full-row DMA

            # Compute the tail destination offset.
            dst_offset = req * tail_stride_0.to(tl.int64) + phys_slot.to(
                tl.int64
            ) * tail_stride_1.to(tl.int64)

            # Store tail_k[req, phys_slot, :] with a block pointer.
            tail_k_block = tl.make_block_ptr(
                base=tail_k_ptr + dst_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            tl.store(tail_k_block, k)

            # Store tail_score[req, phys_slot, :] with a block pointer.
            tail_score_block = tl.make_block_ptr(
                base=tail_score_ptr + dst_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            tl.store(tail_score_block, s)


def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    """Copy selected chunk keys and scores into request-local tail rings."""
    pool_size = pool.index_kpool
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    assert chunk_k.ndim == 2 and chunk_k.shape[1] == INDEX_HEAD_DIM
    assert chunk_score.shape == chunk_k.shape
    assert tail_k.ndim == 3 and tail_k.shape[2] == INDEX_HEAD_DIM
    assert tail_score.shape == tail_k.shape
    assert tail_k.is_contiguous(), "tail_k must be contiguous"
    assert tail_score.is_contiguous(), "tail_score must be contiguous"
    for name, tensor in (
        ("req_pool_idx", req_pool_idx),
        ("dst_logical_start", dst_logical_start),
        ("chunk_src_start", chunk_src_start),
        ("n_write", n_write),
    ):
        assert tensor.ndim == 1 and tensor.shape[0] == n_rows, name
        assert tensor.is_contiguous(), f"{name} must be contiguous"

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()

    num_programs = min(n_rows, get_device_properties()[1])

    _scatter_kpool_tail_updates_kernel[(num_programs,)](
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        n_rows,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        NUM_CORES=num_programs,
    )

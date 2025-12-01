'''
1. make block larger to accelerate
2. cumsum at 0-axis by transpose, (later could be removed as currently the none 0-axis cumsum
    is unrolled by for loop and low-efficient)
'''

from inspect import signature
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sgl_kernel_npu.utils.index import prepare_chunk_indices


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,       
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr=64,
): 
    """
    Computes a chunk-wise cumulative sum (optionally reversed) over input tensor `s` and writes the result to `o`.
    This kernel operates on sequences that may be either fixed-length (batched) or variable-length (packed)..

    The layout of the input/output tensors depends on the `HEAD_FIRST` flag:
      - If `HEAD_FIRST=True`: tensors are shaped `(B, H, T)`
      - If `HEAD_FIRST=False`: tensors are shaped `(B, T, H)`

    For variable-length sequences (`IS_VARLEN=True`), sequence boundaries are defined by `cu_seqlens`,
    and valid computation blocks are specified via `chunk_indices`.

    Args:
        s (tl.pointer): Input tensor pointer. Shape depends on `HEAD_FIRST` and batching mode.
        o (tl.pointer): Output tensor pointer. Same shape and layout as `s`.
        scale (float or None): Optional scalar multiplier applied to the output if `HAS_SCALE=True`.
        cu_seqlens (tl.pointer or None): Cumulative sequence lengths for variable-length batching.
                                         Required if `IS_VARLEN=True`.
        chunk_indices (tl.pointer or None): Pairs of (sequence_id, block_id) indicating which
                                            sequence and which time-block to process.
                                            Only used when `IS_VARLEN=True`.
        T (int): Total sequence length per batch (for fixed-length) or max sequence length (for varlen).
        B (tl.constexpr): Batch size (number of sequences).
        H (tl.constexpr): Number of heads or feature dimension.
        BLOCK_T (tl.constexpr): Number of time steps processed per kernel launch per batch item.
        REVERSE (tl.constexpr): If True, computes reverse cumulative sum within each chunk.
        HAS_SCALE (tl.constexpr): If True, applies `scale` to the output.
        IS_VARLEN (tl.constexpr): If True, uses packed variable-length layout via `cu_seqlens`.
        HEAD_FIRST (tl.constexpr): Controls tensor memory layout (head-first vs time-first).
        CHUNK_SIZE (tl.constexpr, optional): Size of each local chunk for cumsum. Default: 64.

    Notes:
        - The kernel assumes `BLOCK_T` is divisible into chunks of `CHUNK_SIZE` (padding handled internally).
        - Boundary checks are applied during load/store to avoid out-of-bounds access.
        - All computations are performed in fp32 for numerical stability, then cast back to input dtype.

        - reverse cumsum requires T is multiple of CHUNK_SIZE, same as orig code.
    """
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    N_CHUNKS: tl.constexpr = (BLOCK_T + (CHUNK_SIZE - 1)) // CHUNK_SIZE
    
    if IS_VARLEN:
        i_s, i_block = tl.load(chunk_indices + i_block * 2).to(tl.int32), tl.load(
            chunk_indices + i_block * 2 + 1
        ).to(tl.int32)
        
        bos, eos = tl.load(cu_seqlens + i_s).to(tl.int32), tl.load(
            cu_seqlens + i_s + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    
    
    if HEAD_FIRST:
        ptr_s = tl.make_block_ptr(s + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0))
        b_s = tl.load(ptr_s,  boundary_check=(0, 1)).to(tl.float32)
        
        b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
        b_s = tl.trans(b_s, (2, 0, 1))
        b_o = tl.cumsum(b_s, axis=0)
        if REVERSE:
            b_z = tl.sum(b_s, axis=0)
            b_o = -b_o + b_z[None, :, :] + b_s
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (1, 2, 0))
        b_o = tl.reshape(b_o, (H, BLOCK_T))
        tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0, 1))
        
    else:
        ptr_s = tl.make_block_ptr(s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        b_s = tl.load(ptr_s,  boundary_check=(0, 1)).to(tl.float32)
        b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
        b_s = tl.trans(b_s, (1, 0, 2))
        b_o = tl.cumsum(b_s, axis=0)
        if REVERSE:
            b_z = tl.sum(b_s, axis=0)
            b_o = -b_o + b_z[None, :, :]+ b_s
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (1, 0, 2))
        b_o = tl.reshape(b_o, (BLOCK_T, H))
        tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0, 1))
    return


def chunk_local_cumsum_scalar_npu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    OPTIM_BLOCK_SIZE = triton.next_power_of_2((2 ** 18) // (H * chunk_size))
    block_indices = prepare_chunk_indices(cu_seqlens, chunk_size=OPTIM_BLOCK_SIZE) if cu_seqlens is not None else None
    num_blocks = len(block_indices) if cu_seqlens is not None else triton.cdiv(T, OPTIM_BLOCK_SIZE)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (num_blocks, B)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=block_indices,
        T=T,
        B=B,
        H=H,
        BLOCK_T = OPTIM_BLOCK_SIZE,
        CHUNK_SIZE =chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g
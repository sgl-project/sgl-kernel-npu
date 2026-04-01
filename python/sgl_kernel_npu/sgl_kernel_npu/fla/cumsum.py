# Adapted and Merge from
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/cumsum.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, By Triton_Ascend & sglang_ascend

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import (
    exp,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    safe_exp,
)


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
    BLOCK_T: tl.constexpr,  # 每个BLOCK的T维度size, 是CHUNK_SIZE的倍数
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,  # 每个chunk的大小, 增加的新参数,与算法相关
):
    # g_block: 当前prograd 的全局block idx
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    N_CHUNKS: tl.constexpr = BLOCK_T // CHUNK_SIZE

    if IS_VARLEN:
        # [当前block在当前序列的idx, 当前序列的bos对应的idx]
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
        ptr_s = tl.make_block_ptr(
            s + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0)
        )
        ptr_o = tl.make_block_ptr(
            o + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0)
        )
        b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
        b_s = tl.trans(b_s, (2, 0, 1))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (2, 0, 1))
        b_o = tl.reshape(b_o, (H, BLOCK_T))
    else:
        ptr_s = tl.make_block_ptr(
            s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0)
        )
        ptr_o = tl.make_block_ptr(
            o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0)
        )
        b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
        b_s = tl.trans(b_s, (1, 0, 2))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)

        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (1, 0, 2))
        b_o = tl.reshape(b_o, (BLOCK_T, H))

    tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))
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
    OPTIM_BLOCK_SIZE = triton.next_power_of_2((2**18) // (H * chunk_size))
    block_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size=OPTIM_BLOCK_SIZE)
        if cu_seqlens is not None
        else None
    )
    num_blocks = (
        len(block_indices)
        if cu_seqlens is not None
        else triton.cdiv(T, OPTIM_BLOCK_SIZE)
    )
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
        BLOCK_T=OPTIM_BLOCK_SIZE,
        CHUNK_SIZE=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=8,
        num_stages=3,
    )
    return g


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar_npu(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(g.shape) == 4:
        raise NotImplementedError(
            f"Unsupported input shape {g.shape} in ascend, "
            f"chunk_local_cumsum_vector is not implemented in NPU."
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )

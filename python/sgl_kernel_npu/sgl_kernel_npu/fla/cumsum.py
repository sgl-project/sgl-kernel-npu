"""Compatibility entry point for the migrated scalar cumsum operator."""

import torch
from sgl_kernel_npu.operators.attention import chunk_local_cumsum as _cumsum


def chunk_local_cumsum_scalar_npu(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float,
):
    return _cumsum(g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype)


def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float,
    **kwargs,
):
    return chunk_local_cumsum_scalar_npu(
        g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype
    )

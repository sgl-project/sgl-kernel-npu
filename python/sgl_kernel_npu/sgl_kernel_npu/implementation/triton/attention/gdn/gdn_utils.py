# Adapted from fla/utils.py; chunk metadata helpers for GDN Triton implementations.
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from ....torch_native.attention.gdn.prepare_chunk_indices import (
    prepare_chunk_indices as prepare_chunk_indices,
)
from ....torch_native.attention.gdn.prepare_chunk_indices import (
    prepare_lens,
    tensor_cache,
)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
else:
    exp = tl.exp


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))

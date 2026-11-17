# SPDX-License-Identifier: Apache-2.0
"""Public device-length paged attention for fixed-width DSpark draft blocks."""

import math
from numbers import Real

import torch

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.paged_mtp_attention_decode", "auto")
@register_impl("attention.paged_mtp_attention_decode", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if kwargs["query"].device.type != "npu":
        raise ValueError("Triton paged MTP attention requires NPU tensors")
    if _triton_fn is None:
        from ...implementation.triton.attention.paged_mtp_attention import (
            paged_mtp_attention_decode,
        )

        _triton_fn = paged_mtp_attention_decode
    return _triton_fn(
        **kwargs, is_arch22=get_hardware().generation == Generation.ARCH22
    )


@register_impl("attention.paged_mtp_attention_decode", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.paged_mtp_attention import (
            paged_mtp_attention_decode,
        )

        _torch_native_fn = paged_mtp_attention_decode
    return _torch_native_fn(**kwargs)


@register_operator("attention.paged_mtp_attention_decode")
def paged_mtp_attention_decode(
    *,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gamma: int,
    softmax_scale: float,
    max_kv_len: int,
    causal: bool = False,
    impl: str = "auto",
) -> torch.Tensor:
    """Attend to paged KV using device lengths; auto selects Triton.

    query is FP16/BF16 [B*gamma, Hq, D]; matching ND K/V caches have shape
    [pages, page_size, Hkv*D]. seq_lens is contiguous int32 [B] and includes
    the query block. cu_q_lens is contiguous int32 [B+1] with values b*gamma.
    block_tables is int32/int64 [B, max_pages]; all tensors share a device.
    Hq must be divisible by Hkv. max_kv_len only controls work partitioning.
    Causal query i sees keys through seq_lens[b]-gamma+i; empty rows return
    zeros. Returns a new tensor with query's shape, dtype and device.
    impl selects auto, triton or the independent torch_native reference.
    """
    tensors = (query, k_cache, v_cache, seq_lens, cu_q_lens, block_tables)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("Paged MTP attention inputs must be tensors")
    dimensions = (page_size, num_q_heads, num_kv_heads, head_dim, gamma, max_kv_len)
    if any(type(value) is not int for value in dimensions):
        raise TypeError(
            "Paged MTP attention dimensions and max_kv_len must be integers"
        )
    if (
        gamma <= 0
        or num_q_heads <= 0
        or num_kv_heads <= 0
        or num_q_heads % num_kv_heads
    ):
        raise ValueError(
            "Paged MTP attention requires a positive query width and integral GQA groups"
        )
    if page_size < 16 or page_size & (page_size - 1):
        raise ValueError("Paged MTP attention requires a power-of-two page size >= 16")
    if not 16 <= head_dim <= 256:
        raise ValueError(
            "Paged MTP attention requires a head dimension between 16 and 256"
        )
    if max_kv_len < 0:
        raise ValueError("Paged MTP attention requires max_kv_len >= 0")
    if (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, Real)
        or not math.isfinite(softmax_scale)
    ):
        raise ValueError("Paged MTP attention requires a finite softmax_scale")
    if not isinstance(causal, bool):
        raise TypeError("Paged MTP attention requires a boolean causal flag")
    length_shape = seq_lens.shape
    if len(length_shape) != 1:
        raise ValueError("Paged MTP attention requires one-dimensional seq_lens")
    batch_size = length_shape[0]
    if query.shape != (batch_size * gamma, num_q_heads, head_dim):
        raise ValueError(
            "Paged MTP attention requires one fixed-width query block per request"
        )
    cache_shape = k_cache.shape
    if (
        len(cache_shape) != 3
        or cache_shape[1:] != (page_size, num_kv_heads * head_dim)
        or v_cache.shape != cache_shape
    ):
        raise ValueError(
            "Paged MTP attention requires ND paged K/V with equal head dimensions"
        )
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("Paged MTP attention requires contiguous KV head dimensions")
    if seq_lens.dtype != torch.int32 or cu_q_lens.dtype != torch.int32:
        raise ValueError("Paged MTP attention requires int32 device length tensors")
    if not seq_lens.is_contiguous() or not cu_q_lens.is_contiguous():
        raise ValueError("Paged MTP attention requires contiguous length tensors")
    table_shape = block_tables.shape
    if (
        cu_q_lens.shape != (batch_size + 1,)
        or len(table_shape) != 2
        or table_shape[0] != batch_size
    ):
        raise ValueError(
            "Paged MTP attention query offsets and page rows must match the batch"
        )
    if block_tables.dtype not in (torch.int32, torch.int64):
        raise ValueError("Paged MTP attention requires int32 or int64 page indices")
    query_dtype = query.dtype
    if query_dtype not in (torch.float16, torch.bfloat16) or not (
        query_dtype == k_cache.dtype == v_cache.dtype
    ):
        raise ValueError("Paged MTP attention requires matching BF16 or FP16 Q/K/V")
    query_device = query.device
    if any(tensor.device != query_device for tensor in tensors[1:]):
        raise ValueError("Paged MTP attention inputs must reside on the same device")
    impl_fn = get_impl_by_name("attention.paged_mtp_attention_decode", impl)
    return impl_fn(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        block_tables=block_tables,
        page_size=page_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        gamma=gamma,
        softmax_scale=softmax_scale,
        max_kv_len=max_kv_len,
        causal=causal,
    )

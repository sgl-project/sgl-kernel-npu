# SPDX-License-Identifier: Apache-2.0
"""Shared metadata checks for the paged attention sink operators."""

import math
from numbers import Real

import torch

_QUERY_DTYPES = (torch.float16, torch.bfloat16)


def validate_paged_sink_inputs(
    op_name,
    q,
    k_cache,
    v_cache,
    block_tables,
    sinks,
    gqa_interleave,
    softmax_scale,
    max_head_dim=256,
):
    """Validate shared Q/K/V/sink metadata and return the resolved scale.

    ``max_head_dim`` bounds the tile a backend can hold; length tensors and
    page-table values are checked by the implementations, which read them on
    the host to build their schedules.
    """
    for name, tensor in (
        ("q", q),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("block_tables", block_tables),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("{} requires {} to be a tensor".format(op_name, name))
        if not tensor.is_contiguous():
            raise ValueError("{} requires a contiguous {}".format(op_name, name))

    if q.dim() != 3:
        raise ValueError(
            "{} requires q with shape [tokens, heads, head_dim]".format(op_name)
        )
    cache_shape = k_cache.shape
    if len(cache_shape) != 4 or v_cache.shape != cache_shape:
        raise ValueError(
            "{} requires matching K/V caches with shape "
            "[pages, page_size, kv_heads, head_dim]".format(op_name)
        )
    if block_tables.dim() != 2:
        raise ValueError(
            "{} requires a two-dimensional [batch, max_pages] block_tables".format(
                op_name
            )
        )
    if block_tables.dtype not in (torch.int32, torch.int64):
        raise ValueError("{} requires int32 or int64 page indices".format(op_name))

    num_q_heads = q.shape[1]
    num_kv_heads = cache_shape[2]
    head_dim = q.shape[2]
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError(
            "{} requires positive head counts where q heads are divisible "
            "by KV heads".format(op_name)
        )
    if cache_shape[3] != head_dim:
        raise ValueError(
            "{} requires equal query and cache head dimensions".format(op_name)
        )
    if not 16 <= head_dim <= max_head_dim:
        raise ValueError(
            "{} requires a head dimension in [16, {}]".format(op_name, max_head_dim)
        )

    page_size = cache_shape[1]
    if page_size < 16 or page_size & (page_size - 1):
        raise ValueError(
            "{} requires a power-of-two page size of at least 16".format(op_name)
        )

    query_dtype = q.dtype
    if query_dtype not in _QUERY_DTYPES:
        raise ValueError("{} requires float16 or bfloat16 inputs".format(op_name))
    if k_cache.dtype != query_dtype or v_cache.dtype != query_dtype:
        raise ValueError("{} requires matching Q/K/V dtypes".format(op_name))

    if not isinstance(gqa_interleave, bool):
        raise TypeError("{} requires a boolean gqa_interleave".format(op_name))

    if sinks is not None:
        if not isinstance(sinks, torch.Tensor):
            raise TypeError("{} requires sinks to be a tensor or None".format(op_name))
        if sinks.shape != (num_q_heads,):
            raise ValueError(
                "{} requires sinks with shape ({},)".format(op_name, num_q_heads)
            )
        if sinks.dtype != query_dtype:
            raise ValueError("{} requires sinks to use the query dtype".format(op_name))
        if not sinks.is_contiguous():
            raise ValueError("{} requires a contiguous sinks tensor".format(op_name))

    device = q.device
    for tensor in (
        k_cache,
        v_cache,
        block_tables,
    ):
        if tensor.device != device:
            raise ValueError("{} requires all tensors on one device".format(op_name))
    if sinks is not None and sinks.device != device:
        raise ValueError("{} requires all tensors on one device".format(op_name))

    if softmax_scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    else:
        if (
            isinstance(softmax_scale, bool)
            or not isinstance(softmax_scale, Real)
            or not math.isfinite(softmax_scale)
            or softmax_scale <= 0
        ):
            raise ValueError(
                "{} requires a finite positive softmax_scale".format(op_name)
            )
        scale = float(softmax_scale)

    return scale


def validate_length_tensor(op_name, lengths, expected_shape, name, device):
    """Check an int32 length tensor's dtype, shape, contiguity and device."""
    if not isinstance(lengths, torch.Tensor):
        raise TypeError("{} requires {} to be a tensor".format(op_name, name))
    if lengths.dtype != torch.int32:
        raise ValueError("{} requires an int32 {}".format(op_name, name))
    if lengths.shape != expected_shape:
        raise ValueError(
            "{} requires {} with shape {}".format(op_name, name, expected_shape)
        )
    if not lengths.is_contiguous():
        raise ValueError("{} requires a contiguous {}".format(op_name, name))
    if lengths.device != device:
        raise ValueError("{} requires {} on the query device".format(op_name, name))

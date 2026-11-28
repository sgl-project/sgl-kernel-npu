# SPDX-License-Identifier: Apache-2.0
"""Public paged decode attention with sink logits."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from .paged_attention_sink_utils import (
    validate_length_tensor,
    validate_paged_sink_inputs,
)

_OP_NAME = "attention.paged_attention_sink_decode"
_triton_fn = None
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if kwargs["q"].device.type != "npu":
        raise ValueError("Triton paged attention sink decode requires NPU tensors")
    if _triton_fn is None:
        from ...implementation.triton.attention.paged_attention_sink import (
            paged_attention_sink_decode,
        )

        _triton_fn = paged_attention_sink_decode
    return _triton_fn(**kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.paged_attention_sink import (
            paged_attention_sink_decode,
        )

        _torch_native_fn = paged_attention_sink_decode
    return _torch_native_fn(**kwargs)


@register_operator(_OP_NAME)
def paged_attention_sink_decode(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor = None,
    gqa_interleave: bool = False,
    softmax_scale: float = None,
    max_kv_len: int = None,
    impl: str = "auto",
) -> torch.Tensor:
    """Paged decode attention with per-head sink logits; auto selects Triton.

    q is contiguous FP16/BF16 [B, Hq, D] with one query row per request.
    seqlens is contiguous int32 [B] holding each request's KV length, so the
    last KV entry is the current token and every request sees its whole KV row.
    block_tables is int32/int64 [B, max_pages] of physical page IDs with at
    least ceil(seqlens[b] / page_size) valid columns per row. K/V caches are
    matching [pages, page_size, Hkv, D] contiguous tensors. sinks is an optional
    [Hq] tensor in the query dtype; its logit joins the softmax denominator with
    a zero value contribution. gqa_interleave selects the interleaved
    query-to-KV head mapping instead of contiguous groups. softmax_scale
    defaults to D**-0.5. max_kv_len optionally supplies the longest KV row known
    to the caller, which avoids a device-to-host read when choosing the split
    path. Hq is divisible by Hkv, page_size is a power of two of at least 16,
    and head_dim is in [16, 256]. Long, under-filled batches may split the KV
    scan across cores (flash decode) and merge the partial results in a second
    pass.
    Returns a new [B, Hq, D] tensor with q's dtype and device.
    impl selects auto, triton or the independent torch_native reference.
    """
    scale = validate_paged_sink_inputs(
        "paged_attention_sink_decode",
        q,
        k_cache,
        v_cache,
        block_tables,
        sinks,
        gqa_interleave,
        softmax_scale,
    )

    batch_size = q.shape[0]
    validate_length_tensor(
        "paged_attention_sink_decode",
        seqlens,
        (batch_size,),
        "seqlens",
        q.device,
    )
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "paged_attention_sink_decode requires one page-table row per request"
        )
    if max_kv_len is not None and (
        isinstance(max_kv_len, bool)
        or not isinstance(max_kv_len, int)
        or max_kv_len <= 0
    ):
        raise ValueError(
            "paged_attention_sink_decode requires max_kv_len to be a positive "
            "integer or None"
        )

    return get_impl_by_name(_OP_NAME, impl)(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        seqlens=seqlens,
        block_tables=block_tables,
        sinks=sinks,
        gqa_interleave=gqa_interleave,
        softmax_scale=scale,
        max_kv_len=max_kv_len,
    )

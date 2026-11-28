# SPDX-License-Identifier: Apache-2.0
"""Public causal paged prefill attention with sink logits."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from .paged_attention_sink_utils import (
    validate_length_tensor,
    validate_paged_sink_inputs,
)

_OP_NAME = "attention.paged_attention_sink_prefill"
_triton_fn = None
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if kwargs["q"].device.type != "npu":
        raise ValueError("Triton paged attention sink prefill requires NPU tensors")
    if _triton_fn is None:
        from ...implementation.triton.attention.paged_attention_sink import (
            paged_attention_sink_prefill,
        )

        _triton_fn = paged_attention_sink_prefill
    return _triton_fn(**kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.paged_attention_sink import (
            paged_attention_sink_prefill,
        )

        _torch_native_fn = paged_attention_sink_prefill
    return _torch_native_fn(**kwargs)


@register_operator(_OP_NAME)
def paged_attention_sink_prefill(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens_kv: torch.Tensor = None,
    sinks: torch.Tensor = None,
    gqa_interleave: bool = False,
    softmax_scale: float = None,
    impl: str = "auto",
) -> torch.Tensor:
    """Causal paged prefill attention with per-head sink logits; auto selects Triton.

    q is contiguous FP16/BF16 [tokens, Hq, D] with packed requests. cu_q_lens
    is contiguous int32 [B+1] starting at 0; seqlens_kv optionally gives the
    int32 [B] KV length per request (including its own q rows), otherwise that
    length equals the query length. Query row i of a request sees KV positions
    up to the prefixed cache length plus i. block_tables is int32/int64
    [B, max_pages] of physical page IDs, and K/V caches are matching
    [pages, page_size, Hkv, D] contiguous tensors. sinks is an optional [Hq]
    tensor in the query dtype; its logit joins the softmax denominator with a
    zero value contribution.
    gqa_interleave selects the interleaved query-to-KV head mapping instead of
    contiguous groups. softmax_scale defaults to D**-0.5. Hq divisible by Hkv,
    page_size a power of two at least 16, head_dim in [16, 208]. Returns a new
    [tokens, Hq, D] tensor with q's dtype and device.
    impl selects auto, triton or the independent torch_native reference.
    """
    scale = validate_paged_sink_inputs(
        "paged_attention_sink_prefill",
        q,
        k_cache,
        v_cache,
        block_tables,
        sinks,
        gqa_interleave,
        softmax_scale,
        max_head_dim=208,
    )

    if cu_q_lens.dim() != 1:
        raise ValueError(
            "paged_attention_sink_prefill requires one-dimensional cu_q_lens"
        )
    batch_size = cu_q_lens.shape[0] - 1
    validate_length_tensor(
        "paged_attention_sink_prefill",
        cu_q_lens,
        (batch_size + 1,),
        "cu_q_lens",
        q.device,
    )
    if batch_size < 0:
        raise ValueError("paged_attention_sink_prefill requires at least one boundary")
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "paged_attention_sink_prefill requires one page-table row per request"
        )
    if seqlens_kv is not None:
        validate_length_tensor(
            "paged_attention_sink_prefill",
            seqlens_kv,
            (batch_size,),
            "seqlens_kv",
            q.device,
        )

    return get_impl_by_name(_OP_NAME, impl)(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_q_lens=cu_q_lens,
        block_tables=block_tables,
        seqlens_kv=seqlens_kv,
        sinks=sinks,
        gqa_interleave=gqa_interleave,
        softmax_scale=scale,
    )

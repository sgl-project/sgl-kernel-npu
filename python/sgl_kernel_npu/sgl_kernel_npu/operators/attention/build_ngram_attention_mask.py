"""Public build_ngram_attention_mask operator and implementation registration."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.build_ngram_attention_mask", "auto")
@register_impl("attention.build_ngram_attention_mask", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.build_ngram_attention_mask import (
            build_ngram_attention_mask,
        )

        _triton_fn = build_ngram_attention_mask
    return _triton_fn(*args, **kwargs)


@register_impl("attention.build_ngram_attention_mask", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.build_ngram_attention_mask import (
            build_ngram_attention_mask,
        )

        _torch_native_fn = build_ngram_attention_mask
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.build_ngram_attention_mask")
def build_ngram_attention_mask(
    tree_mask: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_token_num: int,
    padded_kv_len: int,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Build the paged FIA mask, where True blocks attention.

    ``tree_mask`` is the contiguous bool [B, D, D] ancestry matrix (or its flat
    view). ``prefix_lens`` is contiguous int32/int64 [B]. ``draft_token_num`` is
    D and ``padded_kv_len`` is the padded key width, at least the largest prefix
    plus D. The mask keeps each prefix and each node's ancestors visible.
    ``impl`` selects a backend; ``auto`` selects Triton. Returns a bool
    [B, 1, D, padded_kv_len] tensor on the input device.
    """
    return get_impl_by_name("attention.build_ngram_attention_mask", impl)(
        tree_mask, prefix_lens, draft_token_num, padded_kv_len
    )

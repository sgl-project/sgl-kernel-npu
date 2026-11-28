"""Public verify_ngram_tree_greedy operator and implementation registration."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.verify_ngram_tree_greedy", "auto")
@register_impl("speculative.verify_ngram_tree_greedy", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.verify_ngram_tree_greedy import (
            verify_ngram_tree_greedy,
        )

        _triton_fn = verify_ngram_tree_greedy
    return _triton_fn(*args, **kwargs)


@register_impl("speculative.verify_ngram_tree_greedy", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.verify_ngram_tree_greedy import (
            verify_ngram_tree_greedy,
        )

        _torch_native_fn = verify_ngram_tree_greedy
    return _torch_native_fn(*args, **kwargs)


@register_operator("speculative.verify_ngram_tree_greedy")
def verify_ngram_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    *,
    impl: str = "auto",
) -> None:
    """Follow matching children in an NGRAM tree and append the target token.

    ``candidates`` and ``retrive_index`` are int32/int64 [B, D]; the latter
    maps nodes to global slots in ``target_predict`` (int32/int64 [B, D]) and
    ``predicts`` (int32 [B*D]). ``retrive_next_token`` and
    ``retrive_next_sibling`` are int32/int64 [B, D] local links ending in -1.
    ``accept_index`` (int32 [B, S], S >= 1) receives the root and accepted
    path; ``accept_token_num`` (int32 [B]) counts drafts only. The outputs are
    updated in place; unwritten entries are untouched. All tensors are
    contiguous. ``impl`` selects a backend; ``auto`` selects Triton.
    """
    return get_impl_by_name("speculative.verify_ngram_tree_greedy", impl)(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )

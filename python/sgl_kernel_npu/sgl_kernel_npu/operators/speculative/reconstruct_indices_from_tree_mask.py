"""Public reconstruct_indices_from_tree_mask operator and implementation registration."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.reconstruct_indices_from_tree_mask", "auto")
@register_impl("speculative.reconstruct_indices_from_tree_mask", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.reconstruct_indices_from_tree_mask import (
            reconstruct_indices_from_tree_mask,
        )

        _triton_fn = reconstruct_indices_from_tree_mask
    return _triton_fn(*args, **kwargs)


@register_impl("speculative.reconstruct_indices_from_tree_mask", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.reconstruct_indices_from_tree_mask import (
            reconstruct_indices_from_tree_mask,
        )

        _torch_native_fn = reconstruct_indices_from_tree_mask
    return _torch_native_fn(*args, **kwargs)


@register_operator("speculative.reconstruct_indices_from_tree_mask")
def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    *,
    impl: str = "auto",
) -> None:
    """Reconstruct verify positions and links from a parent-ordered tree mask.

    ``batch_size`` is B and ``draft_token_num`` is D. ``tree_mask`` is contiguous
    bool [B, D, D] (or flat), with each row marking ancestors. ``verified_seq_len``
    is contiguous int32/int64 [B]. The contiguous int64 outputs ``positions``
    ([B*D] or [B, D]), ``retrive_index``, ``retrive_next_token`` and
    ``retrive_next_sibling`` ([B, D]) are written in place. Links are local and
    use -1 when absent. ``impl`` selects a backend; ``auto`` selects Triton.
    """
    return get_impl_by_name("speculative.reconstruct_indices_from_tree_mask", impl)(
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        draft_token_num,
    )

"""Public dispatch for verification-tree construction and five in-place outputs."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("speculative.build_tree", "auto")
@register_impl("speculative.build_tree", "ascendc")
def _ascendc_impl(*args):
    return torch.ops.npu.build_tree(*args)


@register_impl("speculative.build_tree", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.build_tree import build_tree

        _torch_native_fn = build_tree
    return _torch_native_fn(*args)


@register_operator("speculative.build_tree")
def build_tree(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode,
    *,
    impl="auto",
):
    """Write five caller buffers and return None; auto selects ascendc.

    Args:
        parent_list: int64 Tensor [B,topk*(depth-1)+1], parent candidates.
        selected_index: int64 Tensor [B,D-1], ancestor-complete selected nodes,
            excluding the root; D is draft_token_num.
        verified_seq_len: int64 Tensor [B], verified prefix lengths L_b.
        tree_mask: Mutable flat bool Tensor; FULL_MASK needs
            D*sum(L_b)+B*D*D elements and preserves verified-prefix entries.
        positions: Mutable int64 Tensor [B*D] or [B,D], absolute positions.
        retrive_index: Mutable int64 Tensor [B,D], global node indices.
        retrive_next_token: Mutable int64 Tensor [B,D], first-child links or -1.
        retrive_next_sibling: Mutable int64 Tensor [B,D], sibling links or -1.
        topk: Required positive int, candidate count per step.
        depth: Required positive int, candidate tree depth.
        draft_token_num: Required positive int, node count including the root.
        tree_mask_mode: Required int; 0 is FULL_MASK, 1 is QLEN_ONLY, and
            2 (bitpacking) is unsupported. Ascend C retains the legacy
            QLEN_ONLY row-address defect; use FULL_MASK.
        impl: Keyword-only str, "auto" (default), "ascendc" or "torch_native".

    All tensors use contiguous storage on the same device (NPU for ascendc).
    The five output buffers must not overlap. Returns None without replacing
    their storage. Read-only inputs are unchanged.
    """
    return get_impl_by_name("speculative.build_tree", impl)(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )

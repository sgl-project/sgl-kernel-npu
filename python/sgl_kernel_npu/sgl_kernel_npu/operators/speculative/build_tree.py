"""Public dispatch for verification-tree construction and five in-place outputs."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_GENERATIONS = (Generation.ARCH22, Generation.ARCH35)
_torch_native_fn = None


@register_impl("speculative.build_tree", "auto", supported_generations=_GENERATIONS)
@register_impl("speculative.build_tree", "ascendc", supported_generations=_GENERATIONS)
def _ascendc_impl(*args):
    import torch

    return torch.ops.npu.build_tree(*args)


@register_impl(
    "speculative.build_tree", "torch_native", supported_generations=_GENERATIONS
)
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

    Mode 0 is FULL_MASK, mode 1 is QLEN_ONLY, and mode 2 is unsupported.
    Ascend C deliberately retains the legacy QLEN_ONLY row-address defect;
    torch_native constructs the mathematically correct rows and exposes it.
    FULL_MASK leaves each row's verified-prefix entries unchanged.
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

"""Public greedy verification operator and implementation registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.verify_tree_greedy", "auto")
@register_impl("speculative.verify_tree_greedy", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.verify_tree_greedy import (
            verify_tree_greedy,
        )

        _triton_fn = verify_tree_greedy
    return _triton_fn(**kwargs)


@register_impl("speculative.verify_tree_greedy", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.verify_tree_greedy import (
            verify_tree_greedy_torch,
        )

        _torch_native_fn = verify_tree_greedy_torch
    return _torch_native_fn(**kwargs)


@register_operator("speculative.verify_tree_greedy")
def verify_tree_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    *,
    impl="auto",
):
    """Write accepted tokens, indices and draft counts in place; return None.

    The migrated Triton implementation verifies a linear chain with [B,D]
    accepted-index storage and canonical global slots. Tree links are retained
    in its signature but are not traversed by that implementation.
    """
    return get_impl_by_name("speculative.verify_tree_greedy", impl)(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )

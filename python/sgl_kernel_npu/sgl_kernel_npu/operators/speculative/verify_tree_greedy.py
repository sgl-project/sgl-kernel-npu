"""Public greedy verification operator and implementation registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.verify_tree_greedy", "auto")
@register_impl("speculative.verify_tree_greedy", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.verify_tree_greedy import (
            verify_tree_greedy,
        )

        _triton_fn = verify_tree_greedy
    return _triton_fn(*args, **kwargs)


@register_impl("speculative.verify_tree_greedy", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.verify_tree_greedy import (
            verify_tree_greedy_torch,
        )

        _torch_native_fn = verify_tree_greedy_torch
    return _torch_native_fn(*args, **kwargs)


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
    """Verify drafts against the target model's greedy token predictions.

    The default impl="auto" selects Triton. B is the request count, D is the
    candidate width including the root, and S is the accepted path capacity.
    Triton verifies linear chains with S=D and slots retrive_index[b, j]=b*D+j.

    Args:
        predicts (torch.Tensor): One-dimensional int32 output for accepted and
            final token IDs, with space for every slot in retrive_index.
        accept_index (torch.Tensor): int32 [B, S] output for root and accepted
            slots, in path order.
        accept_token_num (torch.Tensor): int32 [B] output counting accepted
            drafts, excluding the final target token.
        candidates (torch.Tensor): int32 or int64 [B, D] root and draft token IDs.
        retrive_index (torch.Tensor): int32 or int64 [B, D] global output slots.
        retrive_next_token (torch.Tensor): int32 or int64 [B, D] local first-child
            indices, with -1 for no child. Used by torch_native.
        retrive_next_sibling (torch.Tensor): int32 or int64 [B, D] local next-sibling
            indices, with -1 at the end of the chain. Used by torch_native.
        target_predict (torch.Tensor): int32 or int64 [B, D] target next-token IDs.
        impl (str): "auto", "triton", or "torch_native". Defaults to "auto".

    Returns:
        None. Outputs are updated in place. Unwritten slots retain their values.
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

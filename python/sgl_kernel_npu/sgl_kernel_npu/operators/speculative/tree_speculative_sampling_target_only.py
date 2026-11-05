"""Public target-only tree sampling operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.tree_speculative_sampling_target_only", "auto")
@register_impl("speculative.tree_speculative_sampling_target_only", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.tree_speculative_sampling_target_only import (
            tree_speculative_sampling_target_only,
        )

        _triton_fn = tree_speculative_sampling_target_only
    return _triton_fn(**kwargs)


@register_impl("speculative.tree_speculative_sampling_target_only", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.tree_speculative_sampling_target_only import (
            tree_speculative_sampling_target_only_torch,
        )

        _torch_native_fn = tree_speculative_sampling_target_only_torch
    return _torch_native_fn(**kwargs)


@register_operator("speculative.tree_speculative_sampling_target_only")
def tree_speculative_sampling_target_only(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,
    *,
    impl="auto",
):
    """Traverse a draft tree using target probabilities and sample a final token.

    Writes predicts, accept_index and accept_token_num in place. draft_probs is
    scratch: it is cleared and filled with rejected sibling probabilities.
    Returns None. Random samples come from the caller; deterministic is ignored.
    """
    return get_impl_by_name("speculative.tree_speculative_sampling_target_only", impl)(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )

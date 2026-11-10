"""Public target-only tree sampling operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.tree_speculative_sampling_target_only", "auto")
@register_impl("speculative.tree_speculative_sampling_target_only", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.tree_speculative_sampling_target_only import (
            tree_speculative_sampling_target_only,
        )

        _triton_fn = tree_speculative_sampling_target_only
    return _triton_fn(*args, **kwargs)


@register_impl("speculative.tree_speculative_sampling_target_only", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.tree_speculative_sampling_target_only import (
            tree_speculative_sampling_target_only_torch,
        )

        _torch_native_fn = tree_speculative_sampling_target_only_torch
    return _torch_native_fn(*args, **kwargs)


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

    The default impl="auto" selects Triton. B is the request count, D is the
    node count including the root, S is the path capacity, and V is the
    vocabulary size. Unwritten integer output slots retain their values.

    Args:
        predicts (torch.Tensor): One-dimensional int32 output for accepted and
            sampled token IDs, with space for every slot in retrive_index.
        accept_index (torch.Tensor): int32 [B, S] output for root and accepted
            slots, in path order.
        accept_token_num (torch.Tensor): int32 [B] output counting accepted
            drafts, excluding the final sampled token.
        candidates (torch.Tensor): int64 [B, D] root and candidate token IDs.
        retrive_index (torch.Tensor): int64 [B, D] global output slots.
        retrive_next_token (torch.Tensor): int64 [B, D] local first-child indices.
            Use -1 when a node has no child.
        retrive_next_sibling (torch.Tensor): int64 [B, D] local next-sibling
            indices. Use -1 at the end of a sibling chain.
        uniform_samples (torch.Tensor): float32 [B, D] acceptance samples in
            [0, 1), indexed by the current accepted node.
        uniform_samples_for_final_sampling (torch.Tensor): float32 [B] final
            sampling values in [0, 1).
        target_probs (torch.Tensor): Read-only float32 [B, D, V] target probabilities.
        draft_probs (torch.Tensor): Writable float32 [B, D, V] scratch. Nonempty
            calls clear it and record rejected siblings' target probabilities.
        threshold_single (float): Target probability for immediate acceptance.
            Defaults to 1.0.
        threshold_acc (float): Divisor for cumulative acceptance probability,
            clamped to at least 1e-9. Defaults to 1.0.
        deterministic (bool): Ignored compatibility flag. Defaults to True.
        impl (str): "auto", "triton", or "torch_native". Defaults to "auto".

    Returns:
        None. The integer outputs and rejection scratch are updated in place.
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

"""Public chain rejection sampling operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.chain_speculative_sampling", "auto")
@register_impl("speculative.chain_speculative_sampling", "triton")
def _triton_impl(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.chain_speculative_sampling import (
            chain_speculative_sampling_triton,
        )

        _triton_fn = chain_speculative_sampling_triton
    return _triton_fn(*args, **kwargs)


@register_impl("speculative.chain_speculative_sampling", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.chain_speculative_sampling import (
            chain_speculative_sampling_torch,
        )

        _torch_native_fn = chain_speculative_sampling_torch
    return _torch_native_fn(*args, **kwargs)


@register_operator("speculative.chain_speculative_sampling")
def chain_speculative_sampling(
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
    """Verify a linear draft chain and sample a final token.

    The default impl="auto" selects Triton. B is the request count, D is the
    candidate width including the root, R is the draft probability row count,
    and V is the vocabulary size. Unwritten output slots retain their values.

    Args:
        predicts (torch.Tensor): One-dimensional int32 output for accepted and
            sampled token IDs, with space for every slot in retrive_index.
        accept_index (torch.Tensor): int32 [B, D] output for root and accepted
            slots, in path order.
        accept_token_num (torch.Tensor): int32 [B] output counting accepted
            drafts, excluding the final sampled token.
        candidates (torch.Tensor): int64 [B, D] root and draft token IDs.
        retrive_index (torch.Tensor): int64 [B, D] global output slots.
        retrive_next_token (torch.Tensor): Compatibility child links, normally
            int64 [B, D]. Ignored.
        retrive_next_sibling (torch.Tensor): Compatibility sibling links,
            normally int64 [B, D]. Ignored.
        uniform_samples (torch.Tensor): float32 [B, D] acceptance samples in
            [0, 1). The last column is unused.
        uniform_samples_for_final_sampling (torch.Tensor): float32 [B] final
            sampling values in [0, 1).
        target_probs (torch.Tensor): Read-only float32 [B, D, V] target probabilities.
        draft_probs (torch.Tensor): Read-only float32 [B, R, V] draft probabilities.
        threshold_single (float): Ignored compatibility threshold. Defaults to 1.0.
        threshold_acc (float): Ignored compatibility threshold. Defaults to 1.0.
        deterministic (bool): Ignored compatibility flag. Defaults to True.
        impl (str): "auto", "triton", or "torch_native". Defaults to "auto".

    Returns:
        None. The three output tensors are updated in place.
    """
    return get_impl_by_name("speculative.chain_speculative_sampling", impl)(
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

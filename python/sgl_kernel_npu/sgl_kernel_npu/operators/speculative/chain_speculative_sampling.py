"""Public chain rejection sampling operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("speculative.chain_speculative_sampling", "auto")
@register_impl("speculative.chain_speculative_sampling", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.speculative.chain_speculative_sampling import (
            chain_speculative_sampling_triton,
        )

        _triton_fn = chain_speculative_sampling_triton
    return _triton_fn(**kwargs)


@register_impl("speculative.chain_speculative_sampling", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.speculative.chain_speculative_sampling import (
            chain_speculative_sampling_torch,
        )

        _torch_native_fn = chain_speculative_sampling_torch
    return _torch_native_fn(**kwargs)


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
    """Accept a draft prefix and sample a final token into three output buffers.

    Candidate rows contain the root followed by a linear draft chain. Outputs
    are updated in place; unwritten slots retain their initial values. Returns
    None. Tree links, thresholds and deterministic are compatibility arguments
    ignored by both implementations; random samples are supplied by the caller.
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

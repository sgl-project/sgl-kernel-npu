"""Build the host and optional device row plan for sparse KV prefill."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "mem_cache.prefill_plan"


@register_impl(_OP, "auto")
@register_impl(_OP, "triton")
def _triton_impl(*args, **kwargs):
    from ...implementation.triton.mem_cache.prefill_plan import prefill_plan

    return prefill_plan(*args, **kwargs)


@register_operator(_OP)
def prefill_plan(
    requests,
    lengths,
    table,
    total,
    max_length,
    request_rows,
    context_len,
    *,
    current_metadata=None,
    current_valid=None,
    max_current_width=0,
    impl="auto"
):
    """Build the host and optional device row plan for sparse KV prefill."""
    return get_impl_by_name(_OP, impl)(
        requests,
        lengths,
        table,
        total,
        max_length,
        request_rows,
        context_len,
        current_metadata=current_metadata,
        current_valid=current_valid,
        max_current_width=max_current_width,
    )

"""Public sparse FIA page-table preparation."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.fia_prep", "auto")
@register_impl("attention.fia_prep", "triton")
def _triton(*args, **kwargs):
    from ...implementation.triton.attention.fia_prep import fia_prep

    return fia_prep(*args, **kwargs)


@register_impl("attention.fia_prep", "torch_native")
def _native(*args, **kwargs):
    from ...implementation.torch_native.attention.fia_prep import fia_prep

    return fia_prep(*args, **kwargs)


@register_operator("attention.fia_prep")
def fia_prep(
    ti,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    block_table_out=None,
    actual_kvlen_out=None,
    *,
    impl="auto",
):
    """Build [T,topk] physical page table and [T] causal KV lengths.

    ti contains logical blocks with -1 padding; non-own blocks keep their
    order and the current block is appended. Optional output buffers are reused.
    auto selects Triton.
    """
    return get_impl_by_name("attention.fia_prep", impl)(
        ti,
        seq_lens,
        per_query_req,
        req_to_token,
        block_size,
        block_table_out,
        actual_kvlen_out,
    )

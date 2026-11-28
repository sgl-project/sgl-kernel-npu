"""Build the descriptors for valid host KV row copies."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "mem_cache.copy_rows_metadata"


@register_impl(_OP, "auto")
@register_impl(_OP, "triton")
def _triton_impl(*args, **kwargs):
    from ...implementation.triton.mem_cache.copy_rows_metadata import copy_rows_metadata

    return copy_rows_metadata(*args, **kwargs)


@register_operator(_OP)
def copy_rows_metadata(
    sources, destinations, source_rows, destination_rows, valid, *, impl="auto"
):
    """Build the descriptors for valid host KV row copies."""
    return get_impl_by_name(_OP, impl)(
        sources, destinations, source_rows, destination_rows, valid
    )

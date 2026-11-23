"""Public split-KV HiCache operators and MemFabric implementation selection."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

# Direction values expected by the native operators.
H2D = 1
D2H = 2

_memfabric_fn = None
_reference_fn = None


@register_impl("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "memfabric_aicpu")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", "memfabric_aicpu")
@register_impl("mem_cache.transfer_kv_per_layer_kv_split", "memfabric_aicpu")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split", "memfabric_aicpu")
def _aicpu_impl(*args, **kwargs):
    global _memfabric_fn
    if _memfabric_fn is None:
        from .kv_transfer_impl import transfer_kv

        _memfabric_fn = transfer_kv
    return _memfabric_fn(*args, backend="memfabric_aicpu", **kwargs)


@register_impl("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "ascendc_aiv")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", "ascendc_aiv")
@register_impl("mem_cache.transfer_kv_per_layer_kv_split", "ascendc_aiv")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split", "ascendc_aiv")
def _aiv_impl(*args, **kwargs):
    global _memfabric_fn
    if _memfabric_fn is None:
        from .kv_transfer_impl import transfer_kv

        _memfabric_fn = transfer_kv
    return _memfabric_fn(*args, backend="ascendc_aiv", **kwargs)


@register_impl("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "auto")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", "auto")
@register_impl("mem_cache.transfer_kv_per_layer_kv_split", "auto")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split", "auto")
def _auto_impl(*args, **kwargs):
    global _memfabric_fn
    if _memfabric_fn is None:
        from .kv_transfer_impl import transfer_kv

        _memfabric_fn = transfer_kv
    return _memfabric_fn(*args, backend="auto", **kwargs)


@register_impl("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "torch_native")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", "torch_native")
@register_impl("mem_cache.transfer_kv_per_layer_kv_split", "torch_native")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _reference_fn
    if _reference_fn is None:
        from ....implementation.torch_native.mem_cache.kv_transfer import transfer_kv

        _reference_fn = transfer_kv
    return _reference_fn(*args, **kwargs)


def _index_layers(device_layer_id, host_layer_id):
    if device_layer_id is None and host_layer_id is None:
        return (0, 0, 0)
    if device_layer_id is None or host_layer_id is None:
        raise ValueError("Device and Host Indexer layer IDs must be supplied together")
    return (device_layer_id, host_layer_id, 1)


@register_operator("mem_cache.transfer_kv_per_layer_kv_split_pf_lf")
def transfer_kv_per_layer_kv_split_pf_lf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    device_layer_id,
    host_layer_id,
    device_index_layer_id=None,
    host_index_layer_id=None,
    impl="auto",
):
    """H2D one KV layer and its optional Indexer slot.

    Device K/V: [L,P,T,H,D]; Host storage: [P,L,T,H,D].
    Indices are int64 complete-page token slots on the same NPU as the KV buffers.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto uses memfabric_aicpu for MF-registered Host buffers and ascendc_aiv otherwise.
    """
    return get_impl_by_name("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="page_first",
        direction=H2D,
        kv_layers=(device_layer_id, host_layer_id, 1),
        index_layers=_index_layers(device_index_layer_id, host_index_layer_id),
    )


@register_operator("mem_cache.transfer_kv_all_layer_kv_split_lf_pf")
def transfer_kv_all_layer_kv_split_lf_pf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layer_start=0,
    impl="auto",
):
    """D2H every Device KV layer into Host layers starting at host_layer_start.

    Device K/V: [L,P,T,H,D]; Host storage: [P,L,T,H,D].
    Indices are int64 complete-page token slots on the same NPU as the KV buffers.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto uses ascendc_aiv with the registered Host device addresses.
    """
    return get_impl_by_name("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="page_first",
        direction=D2H,
        kv_layers=(0, host_layer_start, None),
        index_layers=(0, 0, None),
    )


@register_operator("mem_cache.transfer_kv_per_layer_kv_split")
def transfer_kv_per_layer_kv_split(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    device_layer_id,
    host_layer_id,
    device_index_layer_id=None,
    host_index_layer_id=None,
    impl="auto",
):
    """H2D one KV layer and its optional Indexer slot.

    Device K/V: [L,P,T,H,D]; Host storage: [L,P,T,H,D].
    Indices are int64 complete-page token slots on the same NPU as the KV buffers.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto uses memfabric_aicpu for MF-registered Host buffers and ascendc_aiv otherwise.
    """
    return get_impl_by_name("mem_cache.transfer_kv_per_layer_kv_split", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="layer_first",
        direction=H2D,
        kv_layers=(device_layer_id, host_layer_id, 1),
        index_layers=_index_layers(device_index_layer_id, host_index_layer_id),
    )


@register_operator("mem_cache.transfer_kv_all_layer_kv_split")
def transfer_kv_all_layer_kv_split(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layer_start=0,
    impl="auto",
):
    """D2H every Device KV layer into Host layers starting at host_layer_start.

    Device K/V: [L,P,T,H,D]; Host storage: [L,P,T,H,D].
    Indices are int64 complete-page token slots on the same NPU as the KV buffers.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto uses ascendc_aiv with the registered Host device addresses.
    """
    return get_impl_by_name("mem_cache.transfer_kv_all_layer_kv_split", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="layer_first",
        direction=D2H,
        kv_layers=(0, host_layer_start, None),
        index_layers=(0, 0, None),
    )

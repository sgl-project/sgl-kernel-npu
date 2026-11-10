"""Split-KV public API: exact D2H/all-layer and H2D/per-layer copies.

Run on NPU with a matching MemFabric host-registration installation.
Both --impl=auto and --impl=aiv use the in-tree AIV kv_exchange operator.
Production dimensions below are borrowed from GLM-5.2, rather than asserted to
be a captured service call.
"""

import pytest

torch = pytest.importorskip("torch")

from sgl_kernel_npu.operators.mem_cache import (
    transfer_kv_all_layer_kv_split,
    transfer_kv_all_layer_kv_split_lf_pf,
    transfer_kv_per_layer_kv_split,
    transfer_kv_per_layer_kv_split_pf_lf,
)

from tests.python.utils.common import (
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

OPERATORS = {
    "page_first": (
        transfer_kv_all_layer_kv_split_lf_pf,
        transfer_kv_per_layer_kv_split_pf_lf,
    ),
    "layer_first": (transfer_kv_all_layer_kv_split, transfer_kv_per_layer_kv_split),
}
perf_shape = [(78, 21, 128, 512, 64, 128)]

# One guard slot on each side of the slot/layer axis: the untouched slots
# make out-of-bounds layer writes visible and give indexer views a non-zero
# offset inside the block.
GUARD_SLOTS_PER_SIDE = 1


@pytest.fixture
def offload_runtime():
    from memfabric_hybrid import offload
    from sgl_kernel_npu.host_memory import set_memfabric_pool, uninitialize_host_memory

    pool = offload.Offload()
    config = offload.OffloadConfig()
    config.device_id = torch.npu.current_device()
    config.reserve_size = 8 << 30
    config.alloc_size = 8 << 30
    config.flags = 0
    config.scene = offload.Scene.LOCAL
    ret = pool.initialize(config)
    if ret != 0:
        raise RuntimeError(f"MemFabric initialize failed: {ret}")
    set_memfabric_pool(pool)
    yield pool
    torch.npu.synchronize()
    uninitialize_host_memory()


def token_indices(pages, page_size, device="cpu"):
    return (
        torch.tensor(pages, dtype=torch.int64, device=device)[:, None] * page_size
        + torch.arange(page_size, dtype=torch.int64, device=device)
    ).flatten()


def generate_data(pool, layout, shape, dtype, packed):
    from sgl_kernel_npu.host_memory import register_host_memory

    layers, index_layers, ps, k_width, v_width, index_width = shape
    large = layers == 78
    device_pages, host_pages = (33, 17) if large else (8, 10)
    source = [32, 1, 29, 4, 23, 7, 19] if large else [6, 2]
    saved = [1, 8, 15, 3, 12, 5, 16] if large else [1, 8]
    target = [0, 3, 11, 15, 20, 26, 31] if large else [0, 5]
    components = []
    for name, count, width, component_dtype in (
        ("k", layers, k_width, dtype),
        ("v", layers, v_width, dtype),
        ("index_k", index_layers, index_width, dtype),
        ("index_k_scale", index_layers if packed else 0, 1, torch.float32),
    ):
        if not count:
            continue
        values = (
            torch.arange(count * device_pages * ps * width, device="cpu").reshape(
                count, device_pages, ps, 1, width
            )
            % 127
        ).to(component_dtype)
        device = values.npu()
        slots = count + 2 * GUARD_SLOTS_PER_SIDE
        full_shape = (host_pages, slots, ps, 1, max(width, 1))
        if layout == "layer_first":
            full_shape = (slots, host_pages, ps, 1, max(width, 1))
        host = pool.empty(list(full_shape), dtype=component_dtype).fill_(77)
        ref_device = values.npu()
        ref_host = pool.empty(list(full_shape), dtype=component_dtype).fill_(77)
        register_host_memory(host)
        register_host_memory(ref_host)
        full_host, full_ref_host = host, ref_host
        if name.startswith("index_"):
            # Indexer storage has its own slot space, independent of KV layers.
            first = GUARD_SLOTS_PER_SIDE
            host = (
                host[:, first : first + count]
                if layout == "page_first"
                else host[first : first + count]
            )
            ref_host = (
                ref_host[:, first : first + count]
                if layout == "page_first"
                else ref_host[first : first + count]
            )
        if name == "index_k_scale":
            device, ref_device = device.squeeze(-1), ref_device.squeeze(-1)
        components.append(
            (name, device, host, ref_device, ref_host, values, full_host, full_ref_host)
        )
    actual, reference = {}, {}
    for name, dev, host, rd, rh, *_ in components:
        actual["device_" + name], actual["host_" + name] = dev, host
        reference["device_" + name], reference["host_" + name] = rd, rh
    actual["page_size"] = reference["page_size"] = ps
    index_layer_ids = tuple(
        slot * (layers - 1) // max(index_layers - 1, 1) for slot in range(index_layers)
    )
    return components, actual, reference, source, saved, target, index_layer_ids


def assert_bytes_equal(actual, expected):
    a = actual.cpu().contiguous().view(torch.uint8)
    b = expected.cpu().contiguous().view(torch.uint8)
    assert torch.equal(a, b)


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize(
    "shape,dtype,packed",
    [
        pytest.param(
            (3, 2, 4, 7, 3, 5), torch.bfloat16, False, id="edge-small-separate-kv"
        ),
        pytest.param(
            (3, 2, 4, 7, 0, 5), torch.uint8, True, id="packed-k-empty-v-fp32-scale"
        ),
        pytest.param(
            (78, 21, 128, 512, 64, 128),
            torch.bfloat16,
            False,
            id="glm52-borrowed-kv-and-sparse-indexer-dimensions",
        ),
    ],
)
def test_roundtrip(offload_runtime, layout, shape, dtype, packed):
    backup, load = OPERATORS[layout]
    impl, impl_ref = get_test_impls()
    require_distinct_impls(backup, impl, impl_ref)
    require_distinct_impls(load, impl, impl_ref)
    components, actual, reference, source, saved, target, index_layer_ids = (
        generate_data(offload_runtime, layout, shape, dtype, packed)
    )
    ps, layers = shape[2], shape[0]
    source_indices = token_indices(source, ps, "npu")
    saved_indices = token_indices(saved, ps, "npu")
    target_indices = token_indices(target, ps, "npu")
    ready, written = torch.npu.Event(), torch.npu.Event()
    write, read = torch.npu.Stream(), torch.npu.Stream()
    ready.record()
    with torch.npu.stream(write):
        write.wait_event(ready)
        backup(source_indices, saved_indices, **actual, host_layer_start=1, impl=impl)
        written.record()
    backup(
        source_indices, saved_indices, **reference, host_layer_start=1, impl=impl_ref
    )
    write.synchronize()
    # Compare all Host bytes, including guard layers and untouched pages.
    for _, _, _, _, _, _, full, ref_full in components:
        assert_bytes_equal(full, ref_full)

    for layer in range(layers):
        slot = index_layer_ids.index(layer) if layer in index_layer_ids else None
        # Deliberately use different source/destination layer IDs.
        selection = dict(
            device_layer_id=(layer + 1) % layers,
            host_layer_id=layer + 1,
            device_index_layer_id=(slot + 1) % shape[1] if slot is not None else None,
            host_index_layer_id=slot,
        )
        with torch.npu.stream(read):
            read.wait_event(written)
            load(target_indices, saved_indices, **actual, **selection, impl=impl)
        load(target_indices, saved_indices, **reference, **selection, impl=impl_ref)
    read.synchronize()
    for name, dev, host, ref_dev, ref_host, original, full, ref_full in components:
        assert_bytes_equal(dev, ref_dev)
        assert_bytes_equal(full, ref_full)
        if not dev.numel():
            continue
        # Check an independent CPU snapshot of D2H, beyond implementation agreement.
        payload = host if layout == "page_first" else host.transpose(0, 1)
        first_layer = 0 if name.startswith("index_") else GUARD_SLOTS_PER_SIDE
        expected = original[:, source]
        copied = payload[saved, first_layer : first_layer + dev.shape[0]].transpose(
            0, 1
        )
        if dev.ndim == 4:
            expected = expected.squeeze(-1)
            copied = copied.squeeze(-1)
        assert_bytes_equal(copied, expected)

    # Repeated calls overwrite the same destinations with identical bytes.
    # These in-place APIs require manual accuracy and isolated reference inputs.
    for operator, args, kwargs in (
        (backup, (source_indices, saved_indices), dict(actual, host_layer_start=1)),
        (load, (target_indices, saved_indices), dict(actual, **selection)),
    ):
        executor = make_executor(
            operator,
            args=args,
            kwargs=kwargs,
            impl=impl,
            impl_ref=impl_ref,
            accuracy_checked=True,
        )
        check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize(
    "selection",
    [
        {"device_layer_id": -1, "host_layer_id": 0},
        {"device_layer_id": 0, "host_layer_id": 999},
        {"device_layer_id": 0, "host_layer_id": 0, "device_index_layer_id": 0},
    ],
)
def test_reject_invalid_layer_selection(offload_runtime, layout, selection):
    _, load = OPERATORS[layout]
    _, actual, _, _, _, _, _ = generate_data(
        offload_runtime, layout, (3, 2, 4, 7, 3, 5), torch.bfloat16, False
    )
    impl, _ = get_test_impls()
    with pytest.raises(ValueError):
        load(
            token_indices([0], 4, "npu"),
            token_indices([1], 4, "npu"),
            **actual,
            **selection,
            impl=impl,
        )

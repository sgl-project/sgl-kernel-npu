"""Split-KV public API: exact D2H/all-layer and H2D/per-layer copies.

Run on NPU with a matching MemFabric host-registration installation.
Both --impl=auto and --impl=ascendc_aiv use the in-tree AIV kv_exchange operator.
Production dimensions below are borrowed from GLM-5.2, rather than asserted to
be a captured service call.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

perf_shape = [(78, 21, 128, 512, 64, 128)]

# One guard slot on each side of the slot/layer axis: the untouched slots
# make out-of-bounds layer writes visible and give indexer views a non-zero
# offset inside the block.
GUARD_SLOTS_PER_SIDE = 1


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
def test_roundtrip(offload_runtime, operators, layout, shape, dtype, packed):
    skip_accuracy = bool(get_skip_acc_check())
    data = generate_data(
        offload_runtime, layout, shape, dtype, packed, with_reference=not skip_accuracy
    )
    executors = check_transfer_accuracy(
        operators[layout], data, shape, layout, skip_accuracy=skip_accuracy
    )
    for executor in executors:
        check_performance(executor, enable_perf=shape in perf_shape)


def generate_data(pool, layout, shape, dtype, packed, *, with_reference=True):
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        register_host_memory,
    )

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
        slots = count + 2 * GUARD_SLOTS_PER_SIDE
        full_shape = (host_pages, slots, ps, 1, max(width, 1))
        if layout == "layer_first":
            full_shape = (slots, host_pages, ps, 1, max(width, 1))

        def allocate_buffers():
            device = values.npu()
            full_host = pool.empty(list(full_shape), dtype=component_dtype).fill_(77)
            register_host_memory(full_host)
            host = full_host
            if name.startswith("index_"):
                # Indexer storage has its own slot space, independent of KV layers.
                first = GUARD_SLOTS_PER_SIDE
                layer_axis = 1 if layout == "page_first" else 0
                host = full_host.narrow(layer_axis, first, count)
            if name == "index_k_scale":
                device = device.squeeze(-1)
            return device, host, full_host

        device, host, full_host = allocate_buffers()
        ref_device, ref_host, full_ref_host = (
            allocate_buffers() if with_reference else (None, None, None)
        )
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


@pytest.fixture
def operators(configure_npu):
    from sgl_kernel_npu.operators.mem_cache import (
        transfer_kv_all_layer_kv_split,
        transfer_kv_all_layer_kv_split_lf_pf,
        transfer_kv_per_layer_kv_split,
        transfer_kv_per_layer_kv_split_pf_lf,
    )

    return {
        "page_first": (
            transfer_kv_all_layer_kv_split_lf_pf,
            transfer_kv_per_layer_kv_split_pf_lf,
        ),
        "layer_first": (transfer_kv_all_layer_kv_split, transfer_kv_per_layer_kv_split),
    }


@pytest.fixture
def offload_runtime():
    from memfabric_hybrid import offload
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        set_memfabric_pool,
        uninitialize_host_memory,
    )

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


def assert_bytes_equal(actual, expected):
    a = actual.cpu().contiguous().view(torch.uint8)
    b = expected.cpu().contiguous().view(torch.uint8)
    assert_accuracy(a, b)


def check_transfer_accuracy(operators, data, shape, layout, *, skip_accuracy):
    backup, load = operators
    impl, impl_ref = get_test_impls()
    for operator in operators:
        if skip_accuracy:
            require_test_impl(operator, impl)
        else:
            require_distinct_impls(operator, impl, impl_ref)

    # Each in-place implementation receives independently registered buffers.
    components, actual, reference, source, saved, target, index_layer_ids = data
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
    if not skip_accuracy:
        backup(
            source_indices,
            saved_indices,
            **reference,
            host_layer_start=1,
            impl=impl_ref,
        )
        write.synchronize()
        # Compare all Host bytes, including guard layers and untouched pages.
        for _, _, _, _, _, _, full, ref_full in components:
            assert_bytes_equal(full, ref_full)

    # The performance executor uses the last layer. Accuracy checks all layers.
    for layer in (layers - 1,) if skip_accuracy else range(layers):
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
        if not skip_accuracy:
            load(target_indices, saved_indices, **reference, **selection, impl=impl_ref)
    read.synchronize()
    if not skip_accuracy:
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
    return tuple(
        make_executor(
            operator,
            args=args,
            kwargs=kwargs,
            impl=impl,
            impl_ref=impl_ref,
            accuracy_checked=not skip_accuracy,
        )
        for operator, args, kwargs in (
            (backup, (source_indices, saved_indices), dict(actual, host_layer_start=1)),
            (load, (target_indices, saved_indices), dict(actual, **selection)),
        )
    )


@pytest.mark.extra_checks
def test_layer_first_does_not_merge_strided_host_pages(
    offload_runtime, operators, monkeypatch
):
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        register_host_memory,
    )

    # Give each AIV core two consecutive pages, even on different SoCs.
    monkeypatch.setenv("SGLANG_KV_EXCHANGE_BLOCK_DIM", "2")
    monkeypatch.setenv("SGLANG_KV_EXCHANGE_START_DIM", "0")
    page_count, page_size, width = 4, 4, 8
    values = torch.arange(page_count * page_size * width, dtype=torch.uint8).view(
        1, page_count, page_size, 1, width
    )
    host_shape = (1, page_count * 2, page_size, 1, width)
    host_storage = offload_runtime.empty(list(host_shape), dtype=values.dtype).fill_(
        255
    )
    register_host_memory(host_storage)
    host_k = host_storage[:, ::2]
    device_k = values.npu()
    indices = token_indices(list(range(page_count)), page_size, "npu")
    backup, _ = operators["layer_first"]
    impl, _ = get_test_impls()
    backup(
        indices,
        indices,
        device_k=device_k,
        host_k=host_k,
        device_v=device_k.new_empty((0,)),
        host_v=host_k.new_empty((0,)),
        page_size=page_size,
        impl=impl,
    )
    torch.npu.synchronize()
    expected = torch.full(host_shape, 255, dtype=values.dtype)
    expected[:, ::2].copy_(values)
    assert_bytes_equal(host_storage, expected)


@pytest.mark.extra_checks
@pytest.mark.usefixtures("operators")
@pytest.mark.parametrize(
    "device_dtype,host_dtype",
    [
        pytest.param(torch.int32, torch.int64, id="device-int32"),
        pytest.param(torch.int64, torch.int32, id="host-int32"),
    ],
)
def test_native_submit_rejects_non_int64_indices(device_dtype, host_dtype):
    device_indices = torch.empty(0, dtype=device_dtype, device="npu")
    host_indices = torch.empty(0, dtype=host_dtype, device="npu")
    device_k = torch.empty((1, 1, 1, 1, 1), device="npu")
    host_k = torch.empty_like(device_k, device="cpu")
    # Zero pages exercise the adapter checks without submitting a copy.
    with pytest.raises(RuntimeError, match="int64"):
        torch.ops.npu.kv_exchange(
            device_indices, host_indices, [device_k], [host_k], [0], 1, 2, 1
        )


@pytest.mark.extra_checks
@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(
            {"device_layer_id": -1, "host_layer_id": 0}, id="negative-device-layer"
        ),
        pytest.param(
            {"device_layer_id": 0, "host_layer_id": 999}, id="host-layer-out-of-range"
        ),
        pytest.param(
            {"device_layer_id": 0, "host_layer_id": 0, "device_index_layer_id": 0},
            id="unpaired-indexer-layer",
        ),
    ],
)
def test_reject_invalid_layer_selection(offload_runtime, operators, layout, selection):
    _, load = operators[layout]
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

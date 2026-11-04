"""Split-KV public API: exact D2H/all-layer and H2D/per-layer copies.

Run on NPU with a matching MemFabric installation. AICPU is used by
--impl=auto/aicpu; --impl=aiv initializes LOCAL. AIV and mixed auto tests need
MemFabric host-registration support; the AICPU-only package does not provide it.
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


@pytest.fixture(scope="module")
def offload_runtime(request):
    from memfabric_hybrid import offload

    config = offload.OffloadConfig()
    config.device_id = torch.npu.current_device()
    config.reserve_size = config.alloc_size = 1 << 30
    config.flags = 0
    config.scene = (
        offload.Scene.LOCAL
        if request.config.getoption("impl") == "aiv"
        else offload.Scene.AICPU
    )
    ret = offload.initialize(config)
    if ret != 0:
        raise RuntimeError(f"MemFabric initialize failed: {ret}")
    yield offload
    torch.npu.synchronize()
    offload.uninitialize()


def token_indices(pages, page_size, device="cpu"):
    return (
        torch.tensor(pages, dtype=torch.int64, device=device)[:, None] * page_size
        + torch.arange(page_size, dtype=torch.int64, device=device)
    ).flatten()


def generate_data(offload, layout, shape, dtype, packed):
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
        full_shape = (host_pages, count + 2, ps, 1, max(width, 1))
        if layout == "layer_first":
            full_shape = (count + 2, host_pages, ps, 1, max(width, 1))
        host = offload.empty(list(full_shape), dtype=component_dtype).fill_(77)
        ref_device = values.npu()
        ref_host = offload.empty(list(full_shape), dtype=component_dtype).fill_(77)
        full_host, full_ref_host = host, ref_host
        if name.startswith("index_"):
            # Indexer storage has its own slot space, independent of KV layers.
            host = (
                host[:, 1 : count + 1]
                if layout == "page_first"
                else host[1 : count + 1]
            )
            ref_host = (
                ref_host[:, 1 : count + 1]
                if layout == "page_first"
                else ref_host[1 : count + 1]
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
        first_layer = 0 if name.startswith("index_") else 1
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
def test_aicpu_cross_stream_lifetime(offload_runtime, layout):
    if get_test_impls()[0] == "aiv":
        pytest.skip("AICPU shared-resource ordering test")
    backup, load = OPERATORS[layout]
    shape = (3, 2, 4, 7, 3, 5)
    components, actual, _, source, saved, target, _ = generate_data(
        offload_runtime, layout, shape, torch.bfloat16, False
    )
    ready = torch.npu.Event()
    ready.record()
    write, read = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(write):
        write.wait_event(ready)
        backup(
            token_indices(source, 4, "npu"),
            token_indices(saved, 4, "npu"),
            **actual,
            host_layer_start=1,
            impl="aicpu",
        )
    # No event wait between transfer streams: the AICPU wrapper orders its resource.
    with torch.npu.stream(read):
        load(
            token_indices(target, 4, "npu"),
            token_indices(saved, 4, "npu"),
            **actual,
            device_layer_id=0,
            host_layer_id=1,
            impl="aicpu",
        )
    read.synchronize()
    for name, dev, _, _, _, original, _, _ in components:
        if name in ("k", "v"):
            assert_bytes_equal(dev[0, target], original[0, source])


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
def test_aicpu_temporary_input_lifetime(offload_runtime, layout):
    if get_test_impls()[0] == "aiv":
        pytest.skip("AICPU temporary-input lifetime test")
    backup, _ = OPERATORS[layout]
    shape = (2, 8, 4, 1, 5)
    values = (torch.arange(320, device="cpu") % 251).to(torch.uint8).reshape(shape)
    device = values.npu()
    host_shape = (10, 2, 4, 1, 5) if layout == "page_first" else (2, 10, 4, 1, 5)
    host = offload_runtime.empty(list(host_shape), dtype=torch.uint8).fill_(77)
    di = token_indices([6, 2], 4, "npu")
    hi = token_indices([1, 8], 4, "npu")
    ready, write = torch.npu.Event(), torch.npu.Stream()
    ready.record()
    with torch.npu.stream(write):
        write.wait_event(ready)
        backup(
            di,
            hi,
            device,
            host,
            torch.empty(0, device=device.device),
            torch.empty(0, device="cpu"),
            page_size=4,
            impl="aicpu",
        )
    del device, di, hi
    # Churn the creation stream's allocator while the transfer stream is active.
    for _ in range(32):
        torch.empty(shape, dtype=torch.uint8, device="npu").fill_(199)
        torch.empty(8, dtype=torch.int64, device="npu").fill_(0)
    write.synchronize()
    payload = host if layout == "page_first" else host.transpose(0, 1)
    assert_bytes_equal(payload[[1, 8]].transpose(0, 1), values[:, [6, 2]])


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("packed", [False, True])
def test_aicpu_graph_roundtrip(offload_runtime, layout, packed):
    if get_test_impls()[0] == "aiv":
        pytest.skip("AICPU graph replay test")
    backup, load = OPERATORS[layout]
    shape = (3, 2, 4, 7, 0 if packed else 3, 5)
    components, actual, _, source, saved, target, index_layer_ids = generate_data(
        offload_runtime,
        layout,
        shape,
        torch.uint8 if packed else torch.bfloat16,
        packed,
    )
    ps, layers, sentinel = shape[2], shape[0], 255
    source_indices = token_indices(source, ps, "npu")
    saved_indices = token_indices(saved, ps, "npu")
    target_indices = token_indices(target, ps, "npu")
    snapshots = [torch.empty_like(dev) for _, dev, *_ in components]

    def execute():
        backup(
            source_indices, saved_indices, **actual, host_layer_start=1, impl="aicpu"
        )
        # In-graph consumers exercise D2H completion before reusing its source,
        # then H2D completion before taking the final snapshot.
        for _, dev, *_ in components:
            if dev.numel():
                dev.fill_(sentinel)
        for layer in range(layers):
            slot = index_layer_ids.index(layer) if layer in index_layer_ids else None
            load(
                target_indices,
                saved_indices,
                **actual,
                device_layer_id=layer,
                host_layer_id=layer + 1,
                device_index_layer_id=slot,
                host_index_layer_id=slot,
                impl="aicpu",
            )
        for snapshot, (_, dev, *_) in zip(snapshots, components):
            snapshot.copy_(dev)

    streams = [torch.npu.Stream(), torch.npu.Stream()]
    streams[0].wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.stream(streams[0]):
            for _ in range(3):
                execute()
        streams[0].synchronize()
        with torch.npu.graph(graph, stream=streams[0], capture_error_mode="global"):
            execute()

        previous_done = None
        for step, (source, saved, target) in enumerate(
            (
                ([6, 2], [1, 8], [0, 5]),
                ([1, 5], [7, 2], [6, 3]),
                ([4, 0], [9, 0], [2, 7]),
                ([2, 7], [0, 6], [5, 1]),
            ),
            start=1,
        ):
            expected = []
            stream = streams[(step - 1) % len(streams)]
            with torch.npu.stream(stream):
                if previous_done is not None:
                    stream.wait_event(previous_done)
                source_indices.copy_(token_indices(source, ps))
                saved_indices.copy_(token_indices(saved, ps))
                target_indices.copy_(token_indices(target, ps))
                for _, dev, _, _, _, original, full, _ in components:
                    values = ((original.to(torch.int32) + 17 * step) % 127).to(
                        dev.dtype
                    )
                    source_values = values.squeeze(-1) if dev.ndim == 4 else values
                    dev.copy_(source_values)
                    full.fill_(sentinel)
                    expected_dev = torch.full_like(source_values, sentinel)
                    expected_host = torch.full_like(full, sentinel)
                    if dev.numel():
                        expected_dev[:, target] = source_values[:, source]
                        host_pf = (
                            expected_host
                            if layout == "page_first"
                            else expected_host.transpose(0, 1)
                        )
                        host_pf[saved, 1 : 1 + dev.shape[0]] = values[
                            :, source
                        ].transpose(0, 1)
                    expected.append((expected_dev, expected_host))
                # These eager allocations must not overwrite graph-private tables.
                for _ in range(8):
                    torch.empty(32, dtype=torch.int64, device="npu").fill_(0)
                graph.replay()
                previous_done = stream.record_event()
                stream.synchronize()
            for snapshot, component, (expected_dev, expected_host) in zip(
                snapshots, components, expected
            ):
                assert_bytes_equal(snapshot, expected_dev)
                assert_bytes_equal(component[1], expected_dev)
                assert_bytes_equal(component[6], expected_host)
    finally:
        for stream in streams:
            stream.synchronize()
        graph.reset()


@pytest.mark.extra_checks
@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("direction", [1, 2])
def test_metadata_addresses(offload_runtime, layout, direction):
    from sgl_kernel_npu.implementation.memfabric.mem_cache.kv_transfer import (
        build_kv_sparse_metadata,
        select_components,
    )

    dev = torch.empty((3, 8, 4, 1, 5), dtype=torch.uint8, device="npu")
    full_shape = [10, 6, 4, 1, 5] if layout == "page_first" else [6, 10, 4, 1, 5]
    host = offload_runtime.empty(full_shape, dtype=torch.uint8)
    empty = torch.empty(0, device="npu")
    components = select_components(
        dev,
        host,
        empty,
        torch.empty(0, device="cpu"),
        None,
        None,
        None,
        None,
        layout,
        (0, 2, None),
        (0, 0, 0),
        preserve_missing=True,
    )
    di, hi = token_indices([6, 2], 4, "npu"), token_indices([1, 8], 4, "npu")
    src, dst, lengths, count = build_kv_sparse_metadata(
        components, di, hi, 4, direction
    )
    expected_d = [dev[layer, page].data_ptr() for page in (6, 2) for layer in range(3)]
    expected_h = [
        (
            host[page, layer + 2] if layout == "page_first" else host[layer + 2, page]
        ).data_ptr()
        for page in (1, 8)
        for layer in range(3)
    ]
    assert src.cpu().tolist() == (expected_h if direction == 1 else expected_d)
    assert dst.cpu().tolist() == (expected_d if direction == 1 else expected_h)
    assert lengths.cpu().tolist() == [20] * 6
    assert count.cpu().item() == 6
    invalid = token_indices([8], 4, "npu")
    _, _, bad_lengths, _ = build_kv_sparse_metadata(
        components, invalid, hi[:4], 4, direction
    )
    assert bad_lengths.cpu().tolist() == [-1] * 3


@pytest.mark.extra_checks
@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("direction", [1, 2])
@pytest.mark.parametrize("component_count", [1, 2, 3, 4])
def test_fused_metadata_components(layout, direction, component_count):
    from sgl_kernel_npu.implementation.memfabric.mem_cache.kv_transfer import (
        build_kv_sparse_metadata,
    )

    # Unequal layer counts cross block boundaries and leave component tails.
    # Nonzero base offsets and PF/LF strides must survive without repacking.
    ps = 4
    components = []
    for layers, width, dtype in (
        (43, 5, torch.bfloat16),
        (2, 2, torch.bfloat16),
        (65, 3, torch.uint8),
        (1, 1, torch.float32),
    )[:component_count]:
        dev = torch.empty((layers + 2, 8, ps, 1, width), dtype=dtype, device="npu")
        dev = dev[1 : layers + 1]
        shape = (10, layers + 2, ps, 1, width)
        if layout == "layer_first":
            shape = (layers + 2, 10, ps, 1, width)
        host = torch.empty(shape, dtype=dtype, device="cpu")
        if layout == "layer_first":
            host = host[1 : layers + 1].transpose(0, 1)
        else:
            host = host[:, 1 : layers + 1]
        components.append((dev, host))

    components.extend([None] * (4 - component_count))
    di = token_indices([6, 2, 4], ps, "npu")
    hi = token_indices([1, 8, 3], ps, "npu")
    # Reuse the same tensors with changed contents, as with graph replay.
    for device_tokens, host_tokens in (
        ([24, 8, 16], [4, 32, 12]),
        ([-4, 1, 32], [4, 32, 12]),
        ([24, 8, 16], [-4, 3, 40]),
    ):
        di[::ps] = torch.tensor(device_tokens, dtype=torch.int64, device="npu")
        hi[::ps] = torch.tensor(host_tokens, dtype=torch.int64, device="npu")
        expected_src, expected_dst, expected_lengths = [], [], []
        for pair in components:
            if pair is None:
                continue
            dev, host = pair
            item = dev.element_size()
            for dt, ht in zip(device_tokens, host_tokens):
                dp, hp = dt // ps, ht // ps
                valid = (
                    dt >= 0
                    and ht >= 0
                    and dt % ps == 0
                    and ht % ps == 0
                    and dp < dev.shape[1]
                    and hp < host.shape[0]
                )
                for layer in range(dev.shape[0]):
                    da, ha = dev.data_ptr(), host.data_ptr()
                    if valid:
                        da += (dp * dev.stride(1) + layer * dev.stride(0)) * item
                        ha += (hp * host.stride(0) + layer * host.stride(1)) * item
                    expected_src.append(ha if direction == 1 else da)
                    expected_dst.append(da if direction == 1 else ha)
                    expected_lengths.append(
                        ps * dev.shape[3] * dev.shape[4] * item if valid else -1
                    )
        src, dst, lengths, count = build_kv_sparse_metadata(
            components, di, hi, ps, direction
        )
        assert src.cpu().tolist() == expected_src
        assert dst.cpu().tolist() == expected_dst
        assert lengths.cpu().tolist() == expected_lengths
        assert count.cpu().tolist() == [len(expected_lengths)]


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

"""Public MF H2D accuracy/performance, followed by supplementary metadata checks.

The main test measures the public transfer, including metadata and MF sparse copy.
Supplementary cases exercise internal tables and failure paths without an MF pool.
"""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("sgl_kernel_npu")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

build = getattr(torch.ops.npu, "build_kv_sparse_metadata", None)
pytestmark = pytest.mark.skipif(
    build is None, reason="build a wheel with build_kv_sparse_metadata"
)


# GLM-5.2: borrowed per-layer K/V/Indexer widths, page size and seven loaded pages.
# Synthetic pools include guard layers and untouched pages.
perf_shape = [(7, 128, 512, 64, 128)]


def generate_data(pool, shape, dtype, layout, *, with_reference):
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        register_host_memory,
    )

    pages, page_size, k_width, v_width, index_width = shape
    actual = {"page_size": page_size}
    reference = {"page_size": page_size}
    host_snapshots = []
    for name, width in (("k", k_width), ("v", v_width), ("index_k", index_width)):
        device_shape = (3, pages + 3, page_size, 1, width)
        device = torch.full(device_shape, -1, dtype=dtype, device="npu")
        host_shape = (pages + 2, 3, page_size, 1, width)
        if layout == "layer_first":
            host_shape = (3, pages + 2, page_size, 1, width)
        values = (
            torch.arange((pages + 2) * 3 * page_size * width, device="cpu")
            .remainder(127)
            .to(dtype)
            .reshape(host_shape)
        )
        host = pool.empty(list(host_shape), dtype=dtype)
        host.copy_(values)
        register_host_memory(host)
        actual["device_" + name], actual["host_" + name] = device, host
        if with_reference:
            # H2D sources are read-only; isolate both mutable destinations.
            reference["device_" + name] = device.clone()
            reference["host_" + name] = host
            host_snapshots.append((host, values))

    device_indices = indices([p * page_size for p in range(2, pages + 2)], page_size)
    host_indices = indices(
        [p * page_size for p in reversed(range(1, pages + 1))], page_size
    )
    selection = dict(
        device_layer_id=1,
        host_layer_id=2,
        device_index_layer_id=2,
        host_index_layer_id=1,
    )
    actual.update(selection)
    reference.update(selection)
    return (device_indices, host_indices), actual, reference, host_snapshots


def check_accuracy(operator, data, *, skip_accuracy):
    args, actual, reference, host_snapshots = data
    impl, impl_ref = get_test_impls()
    if skip_accuracy:
        require_test_impl(operator, impl)
    else:
        require_distinct_impls(operator, impl, impl_ref)
        # The public H2D operator mutates its Device tensors and returns None.
        operator(*args, **reference, impl=impl_ref)
        operator(*args, **actual, impl=impl)
        torch.npu.current_stream().synchronize()
        for name in ("device_k", "device_v", "device_index_k"):
            assert_accuracy(
                actual[name].cpu().view(torch.uint8),
                reference[name].cpu().view(torch.uint8),
            )
        for host, snapshot in host_snapshots:
            assert_accuracy(host.view(torch.uint8), snapshot.view(torch.uint8))
    return make_executor(
        operator,
        args=args,
        kwargs=actual,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((7, 128, 512, 64, 128), id="glm52-borrowed-h2d-dimensions"),
        pytest.param((2, 4, 7, 3, 5), id="edge-small-unaligned-widths"),
    ],
)
def test_memfabric_aicpu_h2d(offload_runtime, shape, dtype, layout):
    from sgl_kernel_npu.operators.mem_cache import (
        transfer_kv_per_layer_kv_split,
        transfer_kv_per_layer_kv_split_pf_lf,
    )

    operator = (
        transfer_kv_per_layer_kv_split_pf_lf
        if layout == "page_first"
        else transfer_kv_per_layer_kv_split
    )
    skip_accuracy = bool(get_skip_acc_check())
    data = generate_data(
        offload_runtime, shape, dtype, layout, with_reference=not skip_accuracy
    )
    executor = check_accuracy(operator, data, skip_accuracy=skip_accuracy)
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.fixture
def offload_runtime(mf_local_device):
    pytest.importorskip("memfabric_hybrid")
    from memfabric_hybrid import offload
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        set_memfabric_pool,
        uninitialize_host_memory,
    )

    pool = offload.Offload()
    config = offload.OffloadConfig()
    config.device_id = mf_local_device
    config.reserve_size = 8 << 30
    config.alloc_size = 8 << 30
    config.flags = 0
    config.scene = offload.Scene.LOCAL
    status = pool.initialize(config)
    if status != 0:
        raise RuntimeError(f"MemFabric initialize failed: {status}")
    set_memfabric_pool(pool)
    try:
        yield pool
    finally:
        torch.npu.synchronize()
        uninitialize_host_memory()


def indices(starts, page_size):
    return torch.tensor(
        [start + offset for start in starts for offset in range(page_size)],
        dtype=torch.int64,
        device="npu",
    )


@pytest.mark.extra_checks
def test_memfabric_aicpu_h2d_graph_replay(offload_runtime):
    impl, _ = get_test_impls()
    if impl not in ("auto", "memfabric_aicpu"):
        pytest.skip("requires the MemFabric AICPU transfer path")

    from sgl_kernel_npu.operators.mem_cache import (
        transfer_kv_per_layer_kv_split_pf_lf as load,
    )
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        register_host_memory,
    )

    page_size = 128
    host_k = offload_runtime.empty([2, 1, page_size, 1, 64], dtype=torch.bfloat16)
    host_k[0].fill_(3)
    host_k[1].fill_(7)
    register_host_memory(host_k)
    device_k = torch.empty((1, 2, page_size, 1, 64), device="npu", dtype=host_k.dtype)
    device_v, host_v = device_k.new_empty((0,)), host_k.new_empty((0,))
    device_indices = indices([0], page_size)
    host_indices = indices([0], page_size)

    def submit():
        load(
            device_indices,
            host_indices,
            device_k,
            host_k,
            device_v,
            host_v,
            device_layer_id=0,
            host_layer_id=0,
            page_size=page_size,
            impl=impl,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        submit()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        submit()

    try:
        with torch.npu.stream(stream):
            device_k.zero_()
            graph.replay()
        stream.synchronize()
        snapshot = device_k.cpu()[0]
        assert_accuracy(snapshot[0].view(torch.uint8), host_k[0, 0].view(torch.uint8))
        assert torch.count_nonzero(snapshot[1]).item() == 0

        with torch.npu.stream(stream):
            device_indices.add_(page_size)
            host_indices.add_(page_size)
            graph.replay()
        stream.synchronize()
        snapshot = device_k.cpu()[0]
        assert_accuracy(snapshot[0].view(torch.uint8), host_k[0, 0].view(torch.uint8))
        assert_accuracy(snapshot[1].view(torch.uint8), host_k[1, 0].view(torch.uint8))
    finally:
        stream.synchronize()
        graph.reset()


@pytest.mark.extra_checks
@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("direction", [1, 2])
@pytest.mark.parametrize("component_count", [1, 2, 3, 4])
def test_addresses_and_updated_indices(layout, direction, component_count):
    ps = 4
    devices, hosts = [], []
    for layers, width, dtype in (
        (43, 5, torch.bfloat16),
        (2, 2, torch.bfloat16),
        (65, 3, torch.uint8),
        (1, 1, torch.float32),
    )[:component_count]:
        device = torch.empty((layers + 2, 8, ps, 1, width), dtype=dtype, device="npu")
        device = device[1 : layers + 1]
        shape = (10, layers + 2, ps, 1, width)
        if layout == "layer_first":
            shape = (layers + 2, 10, ps, 1, width)
        host = torch.empty(shape, dtype=dtype, device="cpu")
        host = (
            host[1 : layers + 1].transpose(0, 1)
            if layout == "layer_first"
            else host[:, 1 : layers + 1]
        )
        devices.append(device)
        hosts.append(host)

    hi = indices([4, 32, 12], ps)
    di = torch.empty(2 * hi.numel(), dtype=hi.dtype, device=hi.device)[::2]
    for dstarts, hstarts in (
        ([24, 8, 16], [4, 32, 12]),
        ([0, 28, 4], [12, 0, 36]),
    ):
        # Keep addresses and shapes fixed; the kernel must read new NPU values.
        di.copy_(indices(dstarts, ps))
        hi.copy_(indices(hstarts, ps))
        src, dst, lengths, count = build(di, hi, devices, hosts, ps, direction)
        expected_src, expected_dst, expected_lengths = [], [], []
        for dev, host in zip(devices, hosts):
            item = dev.element_size()
            for dt, ht in zip(dstarts, hstarts):
                for layer in range(dev.shape[0]):
                    da = (
                        dev.data_ptr()
                        + (layer * dev.stride(0) + dt // ps * dev.stride(1)) * item
                    )
                    ha = (
                        host.data_ptr()
                        + (layer * host.stride(1) + ht // ps * host.stride(0)) * item
                    )
                    expected_src.append(ha if direction == 1 else da)
                    expected_dst.append(da if direction == 1 else ha)
                    expected_lengths.append(ps * dev.shape[3] * dev.shape[4] * item)
        # Synchronize only in the test, never in the implementation.
        assert src.cpu().tolist() == expected_src
        assert dst.cpu().tolist() == expected_dst
        assert lengths.cpu().tolist() == expected_lengths
        assert count.cpu().item() == len(expected_lengths)


@pytest.mark.extra_checks
@pytest.mark.parametrize("direction", [1, 2])
def test_native_submit_callback_and_errors(direction, monkeypatch):
    from sgl_kernel_npu.operators.mem_cache.memfabric import kv_transfer_impl

    monkeypatch.delenv("SGLANG_HICACHE_HOST_MEM_BACKEND", raising=False)
    kv_transfer_impl._get_aicpu_submit.cache_clear()
    submit = kv_transfer_impl._get_aicpu_submit()
    ps = 4
    dev = torch.empty((2, 3, ps, 1, 8), dtype=torch.uint8, device="npu")
    host = torch.empty((4, 2, ps, 1, 8), dtype=torch.uint8, device="cpu")
    di, hi = indices([8, 0], ps), indices([4, 12], ps)
    device_addresses = [
        dev[layer, page].data_ptr() for page in (2, 0) for layer in range(2)
    ]
    host_addresses = [
        host[page, layer].data_ptr() for page in (1, 3) for layer in range(2)
    ]
    expected_src, expected_dst = (
        (host_addresses, device_addresses)
        if direction == 1
        else (device_addresses, host_addresses)
    )

    class FakePool:
        calls = 0
        result = 0
        error = None

        def sparse_copy(self, src, dst, lengths, count, device):
            self.calls += 1
            assert device == dev.device
            assert all(t.device == device for t in (src, dst, lengths, count))
            # Readback is test-only: the real MF callback consumes these NPU tables.
            assert src.cpu().tolist() == expected_src
            assert dst.cpu().tolist() == expected_dst
            assert lengths.cpu().tolist() == [ps * 8] * 4
            assert count.cpu().item() == 4
            if self.error is not None:
                raise self.error
            return self.result

    pool = FakePool()
    args = (pool.sparse_copy, di, hi, [dev], [host], ps, direction, dev.device)
    try:
        submit(*args)
        pool.result = 7
        with pytest.raises(RuntimeError, match="MF sparse_copy failed.*7"):
            submit(*args)
        pool.result = 0
        submit(*args)

        pool.error = ValueError("MF callback sentinel")
        with pytest.raises(ValueError, match="MF callback sentinel") as raised:
            submit(*args)
        assert raised.value is pool.error
        pool.error = None
        submit(*args)
        assert pool.calls == 5
    finally:
        torch.npu.current_stream().synchronize()
        kv_transfer_impl._get_aicpu_submit.cache_clear()


def _check_invalid_page_aborts_batch():
    ps = 4
    dev = torch.empty((1, 2, ps, 1, 1), dtype=torch.uint8, device="npu")
    host = torch.empty((2, 1, ps, 1, 1), dtype=torch.uint8, device="cpu")
    di, hi = indices([0, 1], ps), indices([0, 4], ps)
    stream = torch.npu.current_stream()
    stream.synchronize()
    with pytest.raises(RuntimeError):
        build(di, hi, [dev], [host], ps, 2)
        stream.synchronize()


@pytest.mark.extra_checks
def test_invalid_page_aborts_batch():
    # An AICPU task error can leave the stream in an error state.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])"
            "['_check_invalid_page_aborts_batch']()",
            __file__,
        ],
        check=True,
        timeout=120,
    )


@pytest.mark.extra_checks
def test_record_limit():
    dev = torch.empty((1, 8, 1, 1, 1), dtype=torch.uint8, device="npu")
    host = torch.empty((10, 1, 1, 1, 1), dtype=torch.uint8, device="cpu")
    di = torch.zeros(65537, dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="65536"):
        build(di, di, [dev], [host], 1, 1)


@pytest.mark.extra_checks
def test_empty_tables():
    dev = torch.empty((1, 8, 4, 1, 1), dtype=torch.uint8, device="npu")
    host = torch.empty((10, 1, 4, 1, 1), dtype=torch.uint8, device="cpu")
    di = torch.empty(0, dtype=torch.int64, device="npu")
    src, dst, lengths, count = build(di, di, [dev], [host], 4, 1)
    assert src.numel() == dst.numel() == lengths.numel() == 0
    assert count.cpu().item() == 0


_SPLIT_BYTES = 32 * 1024


def _assert_copy_coverage(src, dst, lengths, originals, split_first):
    """Check byte coverage against actual tensor views, independent of chunk math."""
    intervals = [[] for _ in originals]
    owners = []
    for source, destination, size in zip(src, dst, lengths):
        assert 0 < size <= _SPLIT_BYTES
        matches = [
            (i, source - original_src)
            for i, (original_src, original_dst, original_size) in enumerate(originals)
            if 0 <= source - original_src <= original_size - size
            and destination - original_dst == source - original_src
        ]
        assert len(matches) == 1
        owner, offset = matches[0]
        owners.append(owner)
        intervals[owner].append((offset, size))

    for (_, _, original_size), pieces in zip(originals, intervals):
        end = 0
        for offset, size in sorted(pieces):
            assert offset == end, "copy records leave a gap or overlap"
            end += size
        assert end == original_size

    ordered = owners[:-1] if split_first else owners
    assert ordered == sorted(ordered), "component/page/layer order changed"
    if split_first:
        split_owner = next(
            i for i, (_, _, size) in enumerate(originals) if size > _SPLIT_BYTES
        )
        split_row = owners.index(split_owner)
        assert owners[-1] == split_owner
        assert lengths[split_row] == lengths[-1] == _SPLIT_BYTES // 2
        assert src[-1] == src[split_row] + lengths[split_row]
        assert dst[-1] == dst[split_row] + lengths[split_row]
    if len(lengths) >= 48:
        assert len(lengths) % 2 == 0
        half = len(lengths) // 2
        # MF pairs the first and second halves into an 88 KiB UB allocation.
        for first, second in zip(lengths[:half], lengths[half:]):
            assert first % 32 == 0
            assert (first + 31) // 32 * 32 + (second + 31) // 32 * 32 <= 88 * 1024


@pytest.mark.extra_checks
@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("direction", [1, 2])
@pytest.mark.parametrize(
    "page_bytes,layers,pages,page_size,dtype,expected_count,split_first",
    [
        pytest.param(
            (128 * 1024, 16 * 1024, 32 * 1024),
            (4, 4, 2),
            3,
            128,
            torch.bfloat16,
            66,
            False,
            id="glm-k-v-index",
        ),
        pytest.param(
            (_SPLIT_BYTES + 64,),
            (2,),
            2,
            1,
            torch.uint8,
            8,
            False,
            id="partial-last-chunk",
        ),
        pytest.param(
            (3 * _SPLIT_BYTES,),
            (17,),
            1,
            1,
            torch.uint8,
            52,
            True,
            id="odd-paired-batch",
        ),
        pytest.param(
            (16 * 1024, 3 * _SPLIT_BYTES),
            (1, 16),
            1,
            1,
            torch.uint8,
            50,
            True,
            id="odd-first-large-component-is-second",
        ),
        pytest.param(
            (2 * _SPLIT_BYTES, 16 * 1024),
            (23, 1),
            1,
            1,
            torch.uint8,
            47,
            False,
            id="47-no-pairing",
        ),
        pytest.param(
            (3 * _SPLIT_BYTES,),
            (16,),
            1,
            1,
            torch.uint8,
            48,
            False,
            id="48-pairing-boundary",
        ),
        pytest.param(
            (_SPLIT_BYTES + 1,),
            (23,),
            1,
            1,
            torch.uint8,
            46,
            False,
            id="unaligned-below-pairing-boundary",
        ),
    ],
)
def test_large_page_copy_coverage(
    layout,
    direction,
    page_bytes,
    layers,
    pages,
    page_size,
    dtype,
    expected_count,
    split_first,
):
    devices, hosts = [], []
    for size, layer_count in zip(page_bytes, layers):
        width = size // (page_size * dtype.itemsize)
        dev = torch.empty(
            (layer_count + 2, 4, page_size, 1, width), dtype=dtype, device="npu"
        )[1 : layer_count + 1]
        shape = (5, layer_count + 2, page_size, 1, width)
        if layout == "layer_first":
            shape = (layer_count + 2, 5, page_size, 1, width)
        host = torch.empty(shape, dtype=dtype, device="cpu")
        host = (
            host[1 : layer_count + 1].transpose(0, 1)
            if layout == "layer_first"
            else host[:, 1 : layer_count + 1]
        )
        devices.append(dev)
        hosts.append(host)

    device_pages, host_pages = [2, 0, 1][:pages], [1, 3, 0][:pages]
    di = indices([p * page_size for p in device_pages], page_size)
    hi = indices([p * page_size for p in host_pages], page_size)
    originals = []
    for dev, host in zip(devices, hosts):
        for dp, hp in zip(device_pages, host_pages):
            for layer in range(dev.shape[0]):
                da, ha = dev[layer, dp].data_ptr(), host[hp, layer].data_ptr()
                source, destination = (ha, da) if direction == 1 else (da, ha)
                originals.append((source, destination, dev[layer, dp].nbytes))

    src, dst, lengths, count = build(di, hi, devices, hosts, page_size, direction)
    assert count.cpu().item() == expected_count
    assert src.numel() == dst.numel() == lengths.numel() == expected_count
    _assert_copy_coverage(
        src.cpu().tolist(),
        dst.cpu().tolist(),
        lengths.cpu().tolist(),
        originals,
        split_first,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "page_bytes,layers",
    [
        ((_SPLIT_BYTES + 1,), (24,)),
        ((2 * _SPLIT_BYTES, 1), (24, 1)),
    ],
)
def test_split_paired_batch_rejects_unaligned_payload(page_bytes, layers):
    devices = [
        torch.empty((n, 1, 1, 1, size), dtype=torch.uint8, device="npu")
        for size, n in zip(page_bytes, layers)
    ]
    hosts = [
        torch.empty((1, n, 1, 1, size), dtype=torch.uint8, device="cpu")
        for size, n in zip(page_bytes, layers)
    ]
    di = indices([0], 1)
    # This must fail in the host adapter before enqueueing an AICPU task.
    with pytest.raises(RuntimeError, match="32"):
        build(di, di, devices, hosts, 1, 2)
    torch.npu.current_stream().synchronize()


@pytest.mark.extra_checks
@pytest.mark.parametrize("chunks,accepted_pages", [(2, 32768), (3, 21845)])
@pytest.mark.parametrize("exceeds_limit", [False, True])
def test_split_record_limit(chunks, accepted_pages, exceeds_limit):
    size = chunks * _SPLIT_BYTES
    dev = torch.empty((1, 1, 1, 1, size), dtype=torch.uint8, device="npu")
    host = torch.empty((1, 1, 1, 1, size), dtype=torch.uint8, device="cpu")
    # Reuse one valid page to exercise the table limit without a multi-GB KV pool.
    pages = accepted_pages + int(exceeds_limit)
    di = torch.zeros(pages, dtype=torch.int64, device="npu")
    if exceeds_limit:
        with pytest.raises(RuntimeError, match="65536"):
            build(di, di, [dev], [host], 1, 1)
        return
    src, dst, lengths, count = build(di, di, [dev], [host], 1, 1)
    assert count.cpu().item() == 65536
    assert src.numel() == dst.numel() == lengths.numel() == 65536
    sizes = lengths.cpu().tolist()
    assert all(0 < length <= _SPLIT_BYTES for length in sizes)
    assert sum(sizes) == pages * size
    if chunks == 3:
        assert sizes[0] == sizes[-1] == _SPLIT_BYTES // 2
        sources, destinations = src.cpu().tolist(), dst.cpu().tolist()
        assert sources[-1] == sources[0] + _SPLIT_BYTES // 2
        assert destinations[-1] == destinations[0] + _SPLIT_BYTES // 2

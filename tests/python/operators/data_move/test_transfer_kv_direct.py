import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

# LFM2 convolution state, shared by all-layer D2H and per-layer H2D:
# (layers, device slots, convolution width, channels).
perf_shape = [(2, 4, 128, 3)]


def generate_data(flow, layout, d, h):
    n, slots, host_slots = (2, 4, 5) if layout == "conv" else (3, 8, 10)
    shapes = [(128, 3)] if layout == "conv" else [(2, 4)]
    if layout in ("multi", "multi-transpose"):
        shapes.append((3,))
    devices, hosts, storage_views = [], [], []
    for shape, dtype in zip(shapes, (torch.float32, torch.float16)):
        dev = torch.randn(n, slots, *shape, dtype=dtype, device="npu")
        storage_views.append(dev)
        if layout in ("transpose", "multi-transpose") and len(shape) > 1:
            dev = dev.transpose(-1, -2)
        host = torch.empty(
            host_slots, n, 1, *dev.shape[2:], dtype=dtype, pin_memory=True, device="cpu"
        )
        if flow == "H2D":
            host.normal_()
        else:
            host.zero_()
        devices.append(dev)
        hosts.append(host)
    d, h = torch.tensor(d, dtype=torch.long, device="cpu"), torch.tensor(
        h, dtype=torch.long, device="cpu"
    )
    return devices, hosts, storage_views, d, h


def generate_rejected_data(case):
    dst = torch.zeros(8, 2, 4, device="npu")
    if case == "strided":
        dst = dst[..., ::2]
    host = torch.zeros(
        10, 3, 1, *dst.shape[1:], pin_memory=case != "pageable", device="cpu"
    )
    source_indices = torch.tensor([0], device="cpu")
    target_indices = torch.tensor([8 if case == "bounds" else 0], device="cpu")
    return host, dst, source_indices, target_indices


def generate_repeated_destination_data():
    host = (
        torch.arange(5 * 2 * 4, dtype=torch.float32, device="cpu")
        .reshape(5, 2, 1, 4)
        .pin_memory()
    )
    device = torch.arange(2 * 5 * 4, dtype=torch.float32, device="npu").reshape(2, 5, 4)
    source_indices = torch.tensor([0, 2, 4], device="cpu")
    target_indices = torch.tensor([1, 1, 3], device="cpu")
    return host, device, source_indices, target_indices


@pytest.mark.parametrize(
    "flow,layout,d,h,layers",
    [
        pytest.param(
            "D2H", "multi", [1, 4, 6], [2, 5, 7], [], id="all-layers-scattered"
        ),
        pytest.param(
            "H2D",
            "multi",
            [0, 2, 5],
            [3, 3, 4],
            [1],
            id="single-layer-duplicate-source",
        ),
        pytest.param("H2D", "multi", [1, 6, 3], [7, 2, 5], [2], id="mapping-order"),
        pytest.param("D2H", "transpose", [1, 4, 6], [2, 5, 7], [], id="transpose-d2h"),
        pytest.param("H2D", "transpose", [1, 4, 6], [2, 5, 7], [1], id="transpose-h2d"),
        pytest.param(
            "D2H", "multi-transpose", [1, 4, 6], [2, 5, 7], [], id="multi-transpose-d2h"
        ),
        pytest.param(
            "H2D",
            "multi-transpose",
            [0, 2, 5],
            [3, 3, 4],
            [1],
            id="multi-transpose-h2d",
        ),
        pytest.param("H2D", "multi", [], [], [0], id="empty"),
        pytest.param(
            "roundtrip", "multi", [2, 3, 4], [6, 7, 8], range(3), id="contiguous-runs"
        ),
        pytest.param(
            "roundtrip", "single", [1, 4, 6], [2, 5, 7], range(3), id="single-component"
        ),
        pytest.param(
            "roundtrip",
            "transpose",
            [1, 4, 6],
            [2, 5, 7],
            range(3),
            id="dense-permuted",
        ),
        pytest.param("roundtrip", "conv", [2, 1], [1, 4], [1], id="lfm2-conv-only"),
    ],
)
def test_state_transfer(flow, layout, d, h, layers):
    from sgl_kernel_npu.operators.data_move import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

    devices, hosts, storage_views, d, h = generate_data(flow, layout, d, h)
    # These cases assert aclrt physical-copy semantics: host sources are
    # plain pinned (unregistered) buffers and destinations may repeat with
    # ordered writes. The default auto (AIV) backend requires DVA-registered
    # host memory and rejects repeated destinations, so pin the aclrt impl.
    impl, impl_ref = "aclrt", get_test_impls()[1]
    checked = not get_skip_acc_check()
    stream = torch.npu.Stream()
    event = torch.npu.Event()

    def submit(op, args, ref_args, outputs, expected):
        if checked:
            require_distinct_impls(op, impl, impl_ref)
            op(*ref_args, impl=impl_ref)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            assert op(*args, impl=impl) is None
            event.record(stream)
        event.synchronize()
        if checked:
            for actual, ref in zip(outputs, expected):
                assert_accuracy(
                    actual.contiguous().view(torch.uint8),
                    ref.contiguous().view(torch.uint8),
                )
        executor = make_executor(
            op, args=args, impl=impl, impl_ref=impl_ref, accuracy_checked=checked
        )
        return executor

    unchanged = [v.clone() for v in devices] if checked and d.numel() == 0 else []
    original = [v.clone() for v in devices] if checked and flow == "roundtrip" else []
    if flow != "H2D":
        ref_hosts = (
            [torch.zeros_like(v, pin_memory=True) for v in hosts] if checked else []
        )
        executor = submit(
            transfer_kv_all_layer_direct_lf_pf,
            (devices, hosts, d, h),
            (devices, ref_hosts, d, h),
            hosts,
            ref_hosts,
        )
        if checked and flow == "D2H":
            for storage, host in zip(storage_views, hosts):
                # ACL copies physical payloads; use the original contiguous
                # allocation as an oracle, before any payload transpose.
                selected = storage.cpu()[:, d].transpose(0, 1).contiguous()
                assert_accuracy(
                    host[h, :, 0].flatten(2).view(torch.uint8),
                    selected.flatten(2).view(torch.uint8),
                )
        check_performance(executor, enable_perf=tuple(devices[0].shape) in perf_shape)
    if flow != "D2H":
        for layer in layers:
            for i, (dev, host) in enumerate(zip(devices, hosts)):
                if flow == "roundtrip":
                    dev[layer, d] = 0
                ref = dev[layer].clone() if checked else None
                executor = submit(
                    transfer_kv_per_layer_direct_pf_lf,
                    (host, dev[layer], h, d, layer),
                    (host, ref, h, d, layer),
                    [dev[layer]],
                    [ref],
                )
                if checked and flow == "H2D":
                    assert_accuracy(
                        storage_views[i][layer, d].cpu().flatten(1).view(torch.uint8),
                        host[h, layer, 0].flatten(1).view(torch.uint8),
                    )
                if original:
                    assert_accuracy(
                        dev[layer].contiguous().view(torch.uint8),
                        original[i][layer].contiguous().view(torch.uint8),
                    )

        # Empty indices must preserve every layer, including layers not passed to H2D.
        for actual, expected in zip(devices, unchanged):
            assert_accuracy(actual.view(torch.uint8), expected.view(torch.uint8))

        # Profile a single layer/component call after checking every requested write.
        check_performance(executor, enable_perf=tuple(devices[0].shape) in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        ("pageable", "pinned memory"),
        ("strided", "physically dense"),
        ("bounds", "exceeds component slot count"),
    ],
)
def test_state_transfer_rejected(case, message):
    from sgl_kernel_npu.operators.data_move import transfer_kv_per_layer_direct_pf_lf

    host, dst, source_indices, target_indices = generate_rejected_data(case)
    # The pageable/strided/bounds rejections below are aclrt input checks;
    # pin it because the default auto (AIV) path rejects unregistered host
    # memory with a different message.
    with pytest.raises(RuntimeError, match=message):
        transfer_kv_per_layer_direct_pf_lf(
            host,
            dst,
            source_indices,
            target_indices,
            0,
            impl="aclrt",
        )


@pytest.mark.parametrize("direction", ["D2H", "H2D"])
def test_state_transfer_repeated_destination(direction):
    from sgl_kernel_npu.operators.data_move import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

    # aclrt applies repeated-destination writes in order; the AIV backend
    # (auto) rejects duplicates outright.
    impl, impl_ref = "aclrt", get_test_impls()[1]
    host, device, source_indices, target_indices = generate_repeated_destination_data()
    if direction == "D2H":
        op = transfer_kv_all_layer_direct_lf_pf
        expected = host.clone()
        for source, target in zip(source_indices.tolist(), target_indices.tolist()):
            expected[target, :, 0] = device[:, source].cpu()
        args = ([device], [host], source_indices, target_indices)
        ref = torch.empty_like(host, pin_memory=True).copy_(host)
        ref_args = ([device], [ref], source_indices, target_indices)
        actual = host
    else:
        op = transfer_kv_per_layer_direct_pf_lf
        expected = device[1].cpu().clone()
        for source, target in zip(source_indices.tolist(), target_indices.tolist()):
            expected[target] = host[source, 1, 0]
        args = (host, device[1], source_indices, target_indices, 1)
        ref = device[1].clone()
        ref_args = (host, ref, source_indices, target_indices, 1)
        actual = device[1]
    if not get_skip_acc_check():
        require_distinct_impls(op, impl, impl_ref)
        op(*ref_args, impl=impl_ref)
    op(*args, impl=impl)
    torch.npu.synchronize()
    if not get_skip_acc_check():
        assert_accuracy(actual.cpu().view(torch.uint8), expected.view(torch.uint8))
        assert_accuracy(ref.cpu().view(torch.uint8), expected.view(torch.uint8))


def test_auto_routes_to_fused_aiv():
    from sgl_kernel_npu.operators.utils.registry import get_impl_by_name

    for operator_id in (
        "data_move.transfer_kv_per_layer_direct_pf_lf",
        "data_move.transfer_kv_all_layer_direct_lf_pf",
    ):
        assert get_impl_by_name(operator_id, "auto") is get_impl_by_name(
            operator_id, "ascendc_aiv"
        )


# (layers, slots, payload) exercised by the perf gate.

aiv_perf_shape = [(4, 8, (128,))]


@pytest.fixture
def acl_host():
    """Register host pools with the acl backend so the AIV path resolves DVAs."""
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        register_acl_host_memory,
        uninitialize_host_memory,
    )

    def register(tensor):
        register_acl_host_memory(tensor)
        return tensor

    yield register
    torch.npu.synchronize()
    uninitialize_host_memory()


def _indices(values):
    return torch.tensor(values, dtype=torch.int64, device="npu")


@pytest.mark.parametrize(
    "layers,slots,host_slots,payload,dtype,device_ids,host_ids",
    [
        pytest.param(4, 8, 8, (128,), torch.bfloat16, range(8), range(8), id="bf16"),
        pytest.param(
            3, 6, 6, (8, 64), torch.float16, [1, 3, 5], [5, 0, 2], id="subset"
        ),
        pytest.param(
            2,
            10,
            12,
            (32,),
            torch.float32,
            [7, 2, 9],
            [11, 1, 4],
            id="fp32-pools-differ",
        ),
        pytest.param(3, 5, 5, (16,), torch.float32, [2], [3], id="single-slot"),
        # The AIV kernel stages each slot through an 88KB UB buffer and
        # splits larger entries, so cover the boundary from both sides:
        # 45056 * 2B = 88KB exactly, 45057/45055 straddle it by one element.
        pytest.param(
            2,
            4,
            4,
            (45056,),
            torch.float16,
            range(4),
            range(4),
            id="ub-88kb-exact-fp16",
        ),
        pytest.param(
            2, 4, 4, (45057,), torch.float16, [0, 2], [1, 3], id="ub-88kb-over-fp16"
        ),
        pytest.param(
            2,
            4,
            4,
            (45055,),
            torch.float16,
            range(4),
            range(4),
            id="ub-88kb-under-fp16",
        ),
        # Real GDN state shape (~2MB per slot, fp32): several splits per entry.
        pytest.param(
            2,
            4,
            4,
            (32, 128, 128),
            torch.float32,
            range(4),
            range(4),
            id="gdn-shaped-fp32",
        ),
    ],
)
def test_state_aiv_all_layer_d2h(
    acl_host, layers, slots, host_slots, payload, dtype, device_ids, host_ids
):
    from sgl_kernel_npu.operators.data_move import transfer_kv_all_layer_direct_lf_pf

    impl, impl_ref = "ascendc_aiv", get_test_impls()[1]
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_kv_all_layer_direct_lf_pf, impl, impl_ref)
    torch.manual_seed(0)
    dev = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = acl_host(
        torch.randn(host_slots, layers, 1, *payload, dtype=dtype, device="cpu")
    )
    d, h = _indices(device_ids), _indices(host_ids)

    ref_dev = dev.clone()
    ref_host = host.clone()
    if checked:
        transfer_kv_all_layer_direct_lf_pf([ref_dev], [ref_host], d, h, impl=impl_ref)
    host.zero_()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert (
            transfer_kv_all_layer_direct_lf_pf([dev], [host], d, h, impl=impl)
            is None
        )
    stream.synchronize()

    if checked:
        expected = torch.zeros_like(ref_host)
        for device_slot, host_slot in zip(device_ids, host_ids):
            expected[host_slot, :, 0] = ref_dev[:, device_slot]
        assert_accuracy(host.view(torch.uint8), expected.view(torch.uint8))
    executor = make_executor(
        transfer_kv_all_layer_direct_lf_pf,
        args=([dev], [host], d, h),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=(layers, slots, payload) in aiv_perf_shape)


def test_state_aiv_all_layer_draft_mixed(acl_host):
    # Main layers live in one layer-first tensor while each draft layer is its
    # own single-layer allocation; the host pool is narrowed per component so
    # one submission covers independent layer counts.

    from sgl_kernel_npu.operators.data_move import transfer_kv_all_layer_direct_lf_pf

    impl, impl_ref = "ascendc_aiv", get_test_impls()[1]
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_kv_all_layer_direct_lf_pf, impl, impl_ref)
    layers_main, drafts, slots, host_slots, payload = 3, 2, 8, 9, (48,)
    total_layers = layers_main + drafts
    torch.manual_seed(0)
    dev_main = torch.randn(
        layers_main, slots, *payload, dtype=torch.float32, device="npu"
    )
    dev_drafts = [
        torch.randn(1, slots, *payload, dtype=torch.float32, device="npu")
        for _ in range(drafts)
    ]
    host = acl_host(
        torch.randn(
            host_slots, total_layers, 1, *payload, dtype=torch.float32, device="cpu"
        )
    )
    d, h = _indices([4, 1, 6]), _indices([7, 0, 3])
    device_states = [dev_main, *dev_drafts]
    host_states = [host.narrow(1, 0, layers_main)] + [
        host.narrow(1, layers_main + i, 1) for i in range(drafts)
    ]

    ref_device_states = [dev.clone() for dev in device_states]
    ref_host = host.clone()
    if checked:
        ref_host_states = [ref_host.narrow(1, 0, layers_main)] + [
            ref_host.narrow(1, layers_main + i, 1) for i in range(drafts)
        ]
        transfer_kv_all_layer_direct_lf_pf(
            ref_device_states, ref_host_states, d, h, impl=impl_ref
        )
    host.zero_()
    transfer_kv_all_layer_direct_lf_pf(device_states, host_states, d, h, impl=impl)
    torch.npu.synchronize()

    if checked:
        expected = torch.zeros_like(ref_host)
        for device_slot, host_slot in zip(d.tolist(), h.tolist()):
            expected[host_slot, :layers_main, 0] = ref_device_states[0][:, device_slot]
            for i, draft in enumerate(ref_device_states[1:]):
                expected[host_slot, layers_main + i, 0] = draft[0, device_slot]
        assert_accuracy(host.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "layers,slots,host_slots,payload,dtype,layer_id,host_ids,device_ids",
    [
        pytest.param(4, 8, 8, (128,), torch.bfloat16, 2, range(8), range(8), id="bf16"),
        pytest.param(3, 6, 6, (8, 64), torch.float16, 0, [2, 5], [1, 4], id="subset"),
        pytest.param(
            2, 10, 12, (32,), torch.float32, 1, [9, 3], [5, 0], id="fp32-pools-differ"
        ),
        # UB-staging boundary (88KB per slot) in both directions, plus a
        # multi-split GDN-shaped payload.
        pytest.param(
            2,
            4,
            4,
            (45056,),
            torch.float16,
            1,
            range(4),
            range(4),
            id="ub-88kb-exact-fp16",
        ),
        pytest.param(
            2, 4, 4, (45057,), torch.float16, 0, [1, 3], [0, 2], id="ub-88kb-over-fp16"
        ),
        pytest.param(
            2,
            4,
            4,
            (32, 128, 128),
            torch.float32,
            1,
            range(4),
            range(4),
            id="gdn-shaped-fp32",
        ),
    ],
)
def test_state_aiv_per_layer_h2d(
    acl_host, layers, slots, host_slots, payload, dtype, layer_id, host_ids, device_ids
):
    from sgl_kernel_npu.operators.data_move import transfer_kv_per_layer_direct_pf_lf

    impl, impl_ref = "ascendc_aiv", get_test_impls()[1]
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_kv_per_layer_direct_pf_lf, impl, impl_ref)
    torch.manual_seed(0)
    dev = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = acl_host(
        torch.randn(host_slots, layers, 1, *payload, dtype=dtype, device="cpu")
    )
    h, d = _indices(host_ids), _indices(device_ids)

    ref_dev = torch.zeros_like(dev)
    # The torch_native reference requires a pinned host source; host itself
    # stays a pageable acl-registered pool for the AIV path under test.
    ref_host = torch.empty_like(host, pin_memory=True).copy_(host)
    if checked:
        transfer_kv_per_layer_direct_pf_lf(
            ref_host, ref_dev[layer_id], h, d, layer_id, impl=impl_ref
        )
    dev.zero_()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert (
            transfer_kv_per_layer_direct_pf_lf(
                host, dev[layer_id], h, d, layer_id, impl=impl
            )
            is None
        )
    stream.synchronize()

    if checked:
        assert_accuracy(dev.view(torch.uint8), ref_dev.view(torch.uint8))
    executor = make_executor(
        transfer_kv_per_layer_direct_pf_lf,
        args=(host, dev[layer_id], h, d, layer_id),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=(layers, slots, payload) in aiv_perf_shape)


def test_state_aiv_per_layer_h2d_duplicate_host_source(acl_host):
    # Migrated from the retired unittest suite: one host source slot may back
    # several device slots, and the (src, dst) pairs map in input order even
    # when both sides are unsorted. Expected values are constructed directly
    # so the mapping order is verified without relying on the reference impl.

    from sgl_kernel_npu.operators.data_move import transfer_kv_per_layer_direct_pf_lf

    impl = "ascendc_aiv"
    layers, slots, host_slots, payload = 2, 6, 8, (16,)
    torch.manual_seed(0)
    dev = torch.zeros(layers, slots, *payload, dtype=torch.float32, device="npu")
    host = acl_host(
        torch.randn(host_slots, layers, 1, *payload, dtype=torch.float32, device="cpu")
    )
    # host slot 3 backs device slots 0 and 2; neither side is sorted.
    h, d = _indices([3, 3, 5]), _indices([0, 2, 5])
    layer_id = 1
    transfer_kv_per_layer_direct_pf_lf(host, dev[layer_id], h, d, layer_id, impl=impl)
    torch.npu.synchronize()

    expected = torch.zeros_like(dev)
    for host_slot, device_slot in zip(h.tolist(), d.tolist()):
        expected[layer_id, device_slot] = host[host_slot, layer_id, 0]
    assert_accuracy(dev.view(torch.uint8), expected.view(torch.uint8))


def test_state_aiv_roundtrip(acl_host):
    # Backup every layer D2H, restore them per-layer H2D, and compare.

    from sgl_kernel_npu.operators.data_move import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

    impl, impl_ref = "ascendc_aiv", get_test_impls()[1]
    checked = not get_skip_acc_check()
    layers, slots, payload, dtype = 4, 8, (64,), torch.float16
    torch.manual_seed(0)
    original = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = acl_host(torch.zeros(slots, layers, 1, *payload, dtype=dtype, device="cpu"))
    ids = _indices(range(slots))

    transfer_kv_all_layer_direct_lf_pf([original], [host], ids, ids, impl=impl)
    restored = torch.zeros_like(original)
    for layer_id in range(layers):
        transfer_kv_per_layer_direct_pf_lf(
            host, restored[layer_id], ids, ids, layer_id, impl=impl
        )
    torch.npu.synchronize()

    if checked:
        assert_accuracy(restored.view(torch.uint8), original.view(torch.uint8))


def test_state_aiv_empty_indices_noop(acl_host):
    # Empty indices return before any submission; buffers stay untouched.

    from sgl_kernel_npu.operators.data_move import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

    dev = torch.randn(2, 4, 16, dtype=torch.float32, device="npu")
    host = acl_host(torch.randn(4, 2, 1, 16, dtype=torch.float32, device="cpu"))
    before_host, before_dev = host.clone(), dev.clone()
    empty = _indices([])
    transfer_kv_all_layer_direct_lf_pf([dev], [host], empty, empty, impl="ascendc_aiv")
    transfer_kv_per_layer_direct_pf_lf(
        host, dev[0], empty, empty, 0, impl="ascendc_aiv"
    )
    assert_accuracy(host.view(torch.uint8), before_host.view(torch.uint8))
    assert_accuracy(dev.view(torch.uint8), before_dev.view(torch.uint8))


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,error_type,message",
    [
        ("unregistered", RuntimeError, "not registered"),
        ("too-many", ValueError, "at most 16 components"),
        ("dtype", RuntimeError, "dtypes must match"),
        ("host-layout", RuntimeError, "buffers must be device \\(layers, slots"),
        ("device-layout", RuntimeError, "buffers must be device \\(layers, slots"),
        # Duplicate host destinations race across the round-robin cores.
        ("duplicate-host-dst", RuntimeError, "destination slot indices must be unique"),
    ],
)
def test_state_aiv_all_layer_rejected(acl_host, case, error_type, message):
    from sgl_kernel_npu.operators.data_move import transfer_kv_all_layer_direct_lf_pf

    torch.manual_seed(0)
    dev = torch.randn(2, 4, 16, dtype=torch.float32, device="npu")
    host = torch.randn(4, 2, 1, 16, dtype=torch.float32, device="cpu")
    if case != "unregistered":
        host = acl_host(host)
    if case == "too-many":
        dev_states = [dev] * 17
        host_states = [host] * 17
    else:
        dev_states, host_states = [dev], [host]
        if case == "dtype":
            host_states = [acl_host(host.to(torch.bfloat16))]
        elif case == "host-layout":
            host_states = [host.squeeze(2)]
        elif case == "device-layout":
            dev_states = [dev[0]]
    d, h = _indices([1]), _indices([2])
    if case == "duplicate-host-dst":
        d, h = _indices([1, 3]), _indices([2, 2])
    with pytest.raises(error_type, match=message):
        transfer_kv_all_layer_direct_lf_pf(
            dev_states, host_states, d, h, impl="ascendc_aiv"
        )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,error_type,message",
    [
        ("unregistered", RuntimeError, "not registered"),
        ("layer-id", ValueError, "layer_id is outside"),
        ("host-layout", RuntimeError, "buffers must be device \\(layers, slots"),
        # Slot payloads must be flat-packed: a dense-but-permuted trailing
        # block is rejected (migrated from the retired unittest suite's
        # non-dense rejection; the AIV path copies raw (slot, layer) rows).
        ("non-packed", RuntimeError, "trailing payload dims must be packed"),
        # The AIV kernel dereferences indices unchecked, so out-of-range
        # slots and duplicate destinations are rejected up front (mirrors
        # the retired unittest suite's out-of-range rejection).
        ("out-of-range", RuntimeError, "exceeds component slot count"),
        ("duplicate-dst", RuntimeError, "destination slot indices must be unique"),
    ],
)
def test_state_aiv_per_layer_rejected(acl_host, case, error_type, message):
    from sgl_kernel_npu.operators.data_move import transfer_kv_per_layer_direct_pf_lf

    torch.manual_seed(0)
    dev = torch.randn(2, 4, 16, dtype=torch.float32, device="npu")
    host = torch.randn(4, 2, 1, 16, dtype=torch.float32, device="cpu")
    if case != "unregistered":
        host = acl_host(host)
    layer_id = 0
    dst = dev[0]
    if case == "layer-id":
        layer_id = 2
    elif case == "host-layout":
        host = host.squeeze(2)
    elif case == "non-packed":
        # Same element count as a [slots, 16] layer, but the trailing block
        # is permuted so it is not flat-contiguous inside a slot row.
        base = torch.randn(2, 4, 2, 8, dtype=torch.float32, device="npu")
        dst = base[0].transpose(-1, -2)
        host = acl_host(torch.randn(4, 2, 1, 8, 2, dtype=torch.float32, device="cpu"))
    h, d = _indices([2]), _indices([1])
    if case == "out-of-range":
        d = _indices([4])
    elif case == "duplicate-dst":
        d = _indices([1, 1])
    with pytest.raises(error_type, match=message):
        transfer_kv_per_layer_direct_pf_lf(
            host, dst, h, d, layer_id, impl="ascendc_aiv"
        )

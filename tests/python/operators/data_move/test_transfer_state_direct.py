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
    impl, impl_ref = get_test_impls()
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
    with pytest.raises(RuntimeError, match=message):
        transfer_kv_per_layer_direct_pf_lf(
            host,
            dst,
            source_indices,
            target_indices,
            0,
            impl=get_test_impls()[0],
        )


@pytest.mark.parametrize("direction", ["D2H", "H2D"])
def test_state_transfer_repeated_destination(direction):
    from sgl_kernel_npu.operators.data_move import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

    impl, impl_ref = get_test_impls()
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

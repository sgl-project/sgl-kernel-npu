import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance


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

    n, slots, host_slots = (2, 4, 5) if layout == "conv" else (3, 8, 10)
    shapes = [(128, 3)] if layout == "conv" else [(2, 4)]
    if layout == "multi":
        shapes.append((3,))
    devices, hosts = [], []
    for shape, dtype in zip(shapes, (torch.float32, torch.float16)):
        dev = torch.randn(n, slots, *shape, dtype=dtype, device="npu")
        if layout == "transpose":
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
        check_performance(executor, enable_perf=layout == "conv")
        if checked and flow == "D2H":
            for dev, host in zip(devices, hosts):
                selected = dev.cpu()[:, d].transpose(0, 1).contiguous()
                assert_accuracy(
                    host[h, :, 0].view(torch.uint8), selected.view(torch.uint8)
                )
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
                        dev[layer, d].cpu().view(torch.uint8),
                        host[h, layer, 0].view(torch.uint8),
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
        check_performance(executor, enable_perf=layout == "conv")


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

    dst = torch.zeros(8, 2, 4, device="npu")
    if case == "strided":
        dst = dst[..., ::2]
    host = torch.zeros(
        10, 3, 1, *dst.shape[1:], pin_memory=case != "pageable", device="cpu"
    )
    with pytest.raises(RuntimeError, match=message):
        transfer_kv_per_layer_direct_pf_lf(
            host,
            dst,
            torch.tensor([0], device="cpu"),
            torch.tensor([8 if case == "bounds" else 0], device="cpu"),
            0,
            impl=get_test_impls()[0],
        )

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

# LFM2 convolution state: (layers, device slots, convolution width, channels).
perf_shape = [(2, 4, 128, 3)]


def generate_data(layers, slots, host_slots, payload, dtype, d, h):
    dev = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = torch.randn(
        host_slots, layers, 1, *payload, dtype=dtype, pin_memory=True, device="cpu"
    )
    d, h = torch.tensor(d, device="cpu"), torch.tensor(h, device="cpu")
    return dev, host, d, h


def generate_dense_transpose_data():
    storage = torch.arange(2 * 4 * 2 * 3, device="npu", dtype=torch.float32).reshape(
        2, 4, 2, 3
    )
    device = storage.transpose(-1, -2)
    host = (
        torch.arange(5 * 2 * 3 * 2, dtype=torch.float32, device="cpu")
        .reshape(5, 2, 1, 3, 2)
        .pin_memory()
    )
    d, h = torch.tensor([2, 0], device="cpu"), torch.tensor([1, 4], device="cpu")
    return storage, device, host, d, h


def generate_rejected_data(case):
    device = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32, device="npu").reshape(
        2, 4, 2, 3
    )
    host = torch.full(
        (5, 2, 1, 2, 3), -7, dtype=device.dtype, pin_memory=True, device="cpu"
    )
    if case == "dtype":
        host = host.to(torch.int32).pin_memory()
    elif case in ("slot-gap", "layer-gap"):
        shape = (2, 8, 2, 3) if case == "slot-gap" else (4, 4, 2, 3)
        storage = torch.zeros(shape, device="npu")
        device = storage[:, ::2] if case == "slot-gap" else storage[::2]
    elif case == "payload-gap":
        device = device[..., ::2]
        host = torch.zeros(5, 2, 1, 2, 2, pin_memory=True, device="cpu")
    elif case == "host-layout":
        host = host.transpose(-1, -2)
    elif case == "pageable":
        host = torch.zeros_like(host, pin_memory=False)
    elif case == "device":
        device = device.cpu()
    elif case == "host-device":
        host = host.to(device.device)
    elif case == "shape":
        host = torch.zeros(5, 2, 1, 5, pin_memory=True, device="cpu")
    d, h = torch.tensor([1], device="cpu"), torch.tensor([2], device="cpu")
    if case == "negative":
        d[0] = -1
    elif case == "late-index":
        d, h = torch.tensor([1, -1], device="cpu"), torch.tensor([2, 3], device="cpu")
    elif case == "bounds":
        d[0] = 4
    elif case == "index-rank":
        d = d.reshape(1, 1)
    elif case == "index-dtype":
        d = d.to(torch.int32)
    return device, host, d, h


@pytest.mark.parametrize(
    "direction,layers,slots,host_slots,payload,dtype,d,h",
    [
        pytest.param(
            "D2H", 4, 8, 8, (128,), torch.bfloat16, range(8), range(8), id="d2h-1d"
        ),
        pytest.param(
            "D2H", 3, 6, 6, (8, 64), torch.float16, range(6), range(6), id="d2h-2d"
        ),
        pytest.param(
            "D2H",
            2,
            10,
            10,
            (32,),
            torch.float32,
            [1, 3, 5],
            [7, 2, 9],
            id="d2h-subset",
        ),
        pytest.param(
            "H2D", 4, 8, 8, (128,), torch.bfloat16, range(8), range(8), id="h2d-1d"
        ),
        pytest.param(
            "H2D", 3, 6, 6, (8, 64), torch.float16, range(6), range(6), id="h2d-2d"
        ),
        pytest.param(
            "H2D",
            2,
            10,
            10,
            (32,),
            torch.float32,
            [3, 7, 1],
            [0, 4, 8],
            id="h2d-subset",
        ),
        pytest.param(
            "roundtrip",
            4,
            8,
            8,
            (64,),
            torch.float16,
            range(8),
            range(8),
            id="roundtrip",
        ),
        pytest.param(
            "D2H", 2, 4, 4, (32,), torch.float32, range(4), range(4), id="d2h-fp32"
        ),
        pytest.param("D2H", 3, 5, 5, (16,), torch.float32, [2], [3], id="single-slot"),
        pytest.param(
            "H2D", 2, 4, 5, (128, 3), torch.float16, [2, 0], [3, 1], id="lfm2-conv-only"
        ),
    ],
)
def test_transfer_mamba_state(
    direction, layers, slots, host_slots, payload, dtype, d, h
):
    from sgl_kernel_npu.operators.data_move import (
        TransferDirection,
        transfer_mamba_state,
    )

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_mamba_state, impl, impl_ref)
    dev, host, d, h = generate_data(layers, slots, host_slots, payload, dtype, d, h)
    stream = torch.npu.Stream()
    original = dev.clone() if direction == "roundtrip" and checked else None
    for step in ("D2H", "H2D") if direction == "roundtrip" else (direction,):
        to_host = step == "D2H"
        dst = host if to_host else dev
        dst.zero_()
        args = (dev, host, d, h, TransferDirection[step])
        if checked:
            expected = (
                torch.zeros_like(dst, pin_memory=True)
                if to_host
                else torch.zeros_like(dst)
            )
            reference = (dev, expected) if to_host else (expected, host)
            transfer_mamba_state(*reference, *args[2:], impl=impl_ref)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            assert transfer_mamba_state(*args, impl=impl) is None
        stream.synchronize()
        if checked:
            assert_accuracy(dst.view(torch.uint8), expected.view(torch.uint8))
            selected = dev.cpu()[:, d].transpose(0, 1)
            assert_accuracy(
                host[h, :, 0].view(torch.uint8), selected.contiguous().view(torch.uint8)
            )
            if original is not None and not to_host:
                assert_accuracy(dev.view(torch.uint8), original.view(torch.uint8))
        executor = make_executor(
            transfer_mamba_state,
            args=args,
            impl=impl,
            impl_ref=impl_ref,
            accuracy_checked=checked,
        )
        check_performance(executor, enable_perf=tuple(dev.shape) in perf_shape)


@pytest.mark.parametrize("direction", ["D2H", "H2D"])
def test_transfer_mamba_state_dense_transpose(direction):
    from sgl_kernel_npu.operators.data_move import (
        TransferDirection,
        transfer_mamba_state,
    )

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    storage, device, host, d, h = generate_dense_transpose_data()
    args = (d, h, TransferDirection[direction])
    if checked:
        require_distinct_impls(transfer_mamba_state, impl, impl_ref)
        expected = host.clone() if direction == "D2H" else storage.cpu().clone()
        for device_slot, host_slot in zip(d.tolist(), h.tolist()):
            for layer in range(2):
                if direction == "D2H":
                    expected[host_slot, layer, 0].copy_(
                        storage[layer, device_slot].cpu().reshape(3, 2)
                    )
                else:
                    expected[layer, device_slot].copy_(
                        host[host_slot, layer, 0].reshape(2, 3)
                    )
        ref_storage = storage.clone()
        ref_device = ref_storage.transpose(-1, -2)
        ref_host = torch.empty_like(host, pin_memory=True).copy_(host)
        transfer_mamba_state(ref_device, ref_host, *args, impl=impl_ref)
    transfer_mamba_state(device, host, *args, impl=impl)
    torch.npu.synchronize()
    if checked:
        actual = host if direction == "D2H" else storage.cpu()
        reference = ref_host if direction == "D2H" else ref_storage.cpu()
        assert_accuracy(actual.view(torch.uint8), expected.view(torch.uint8))
        assert_accuracy(reference.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        ("dtype", "same dtype"),
        ("slot-gap", "slot pitch"),
        ("layer-gap", "layer pitch"),
        ("payload-gap", "physically dense"),
        ("host-layout", "contiguous"),
        ("pageable", "pinned memory"),
        ("device", "NPU"),
        ("host-device", "CPU"),
        ("shape", "shape|buffers"),
        ("negative", "non-negative|out of range"),
        ("late-index", "non-negative|out of range"),
        ("bounds", "index|out of range"),
        ("index-rank", "one-dimensional int64"),
        ("index-dtype", "one-dimensional int64"),
    ],
)
def test_transfer_mamba_state_rejected(case, message):
    from sgl_kernel_npu.operators.data_move import (
        TransferDirection,
        transfer_mamba_state,
    )

    device, host, d, h = generate_rejected_data(case)
    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_mamba_state, impl, impl_ref)
    before = host.clone()
    for backend in (impl, impl_ref) if checked else (impl,):
        error_type, error_message = RuntimeError, message
        if case == "device" and backend in ("auto", "aclrt"):
            # All-CPU inputs fail dispatch before the NPU kernel's validation.
            error_type = NotImplementedError
            error_message = (
                "Could not run 'npu::transfer_mamba_state' "
                "with arguments from the 'CPU' backend"
            )
        with pytest.raises(error_type, match=error_message):
            transfer_mamba_state(
                device, host, d, h, TransferDirection.D2H, impl=backend
            )
        torch.npu.synchronize()
        if checked:
            assert_accuracy(
                host.contiguous().view(torch.uint8),
                before.contiguous().view(torch.uint8),
            )

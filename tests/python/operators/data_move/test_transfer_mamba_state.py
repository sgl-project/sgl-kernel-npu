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
    dev = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = torch.randn(host_slots, layers, 1, *payload, dtype=dtype, pin_memory=True)
    d, h = torch.tensor(d), torch.tensor(h)
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
        check_performance(executor, enable_perf=payload == (128, 3))

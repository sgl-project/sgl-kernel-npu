import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(4, 1024, 17, 4)]


def generate_data(shape, initial):
    batch, dim, tokens, width = shape
    torch.manual_seed(42)
    lengths = [tokens - i for i in range(batch)]
    starts = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
    x = torch.randn(dim, sum(lengths), dtype=torch.float16)
    weight = torch.randn(dim, width, dtype=torch.float16)
    bias = torch.randn(dim, dtype=torch.float16)
    states = torch.randn(batch + 1, dim, width - 1, dtype=torch.float16)
    slots = torch.arange(batch, dtype=torch.int32).flip(0)
    flags = torch.full((batch,), initial, dtype=torch.bool)
    return (x, weight, bias, starts, slots, flags, states)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 1024, 17, 4), id="mamba-varlen-prefill"),
        pytest.param((1, 128, 1, 3), id="single-token"),
    ],
)
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "swish"])
def test_causal_conv1d_fn(shape, initial, activation):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_fn

    args = generate_data(shape, initial)
    impl, impl_ref = get_test_impls()
    require_test_impl(causal_conv1d_fn, impl)
    executor = make_executor(
        causal_conv1d_fn, args=args, kwargs={"activation": activation}
    )
    if not get_skip_acc_check():
        # This API is already PyTorch-only. Compare explicit temporal windows, not the same registered function twice.
        x, weight, bias, starts, slots, flags, states = args
        expected = torch.empty_like(x)
        final = states.clone()
        for request in range(shape[0]):
            begin, end = int(starts[request]), int(starts[request + 1])
            slot = int(slots[request])
            history = states[slot] if initial else torch.zeros_like(states[slot])
            signal = torch.cat((history, x[:, begin:end]), -1)
            windows = signal.unfold(-1, shape[-1], 1)
            value = (windows.float() * weight[:, None].float()).sum(-1) + bias[
                :, None
            ].float()
            value = value.to(x.dtype)
            expected[:, begin:end] = value if activation is None else F.silu(value)
            final[slot] = signal[:, -(shape[-1] - 1) :]
        actual = executor()
        assert_accuracy(actual, expected)
        assert_accuracy(
            states, final, standard={str(states.dtype): AccuracyStandard(0, 0)}
        )
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)

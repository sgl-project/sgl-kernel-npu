import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(128, 6144)]
quant_standard = {
    "torch.int8": AccuracyStandard(1, 0),
    "torch.float32": AccuracyStandard(0, 5e-3),
}


def generate_data(shape):
    torch.manual_seed(42)
    counts = [shape[0]] + [0] * 15
    return torch.randn(shape, dtype=torch.bfloat16), torch.tensor(
        counts, dtype=torch.int64
    )


@pytest.mark.parametrize(
    "shape,need_quant",
    [
        pytest.param((1, 6144), False, id="kimi-k3.1-pro-moe-decode"),
        pytest.param((128, 6144), True, id="kimi-k3.1-pro-moe-int8-prefill"),
    ],
)
def test_situ(shape, need_quant):
    from sgl_kernel_npu.operators.activation import situ

    x, counts = generate_data(shape)
    executor = check_accuracy(
        situ,
        args=(x, counts, 1),
        kwargs={"need_quant": need_quant, "linear_beta": 25.0},
        standard=quant_standard,
    )
    check_performance(executor, enable_perf=shape in perf_shape)

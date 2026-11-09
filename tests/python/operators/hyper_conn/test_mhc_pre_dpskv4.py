"""mHC pre accuracy and selected-shape performance through public APIs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# DeepSeek-V4-Flash: hidden_size=4096, hc_mult=4.
perf_shape = [(1, 4, 4096), (16, 4, 4096)]
mhc_shapes = [
    pytest.param((1, 4, 4096), id="deepseek-v4-flash-decode-t1-hc4-h4096"),
    pytest.param((16, 4, 4096), id="deepseek-v4-flash-decode-t16-hc4-h4096"),
]


def generate_data(shape):
    torch.manual_seed(11)
    _, streams, hidden = shape
    width = streams * (streams + 2)
    x = torch.randn(shape, dtype=torch.bfloat16)
    fn = torch.randn(width, streams * hidden, dtype=torch.float32) * 0.01
    scale = torch.ones(3, dtype=torch.float32)
    base = torch.zeros(width, dtype=torch.float32)
    return (x, fn, scale, base, streams, 20, 1e-6, 1e-6)


@pytest.mark.parametrize("shape", mhc_shapes)
def test_mhc_pre_dpskv4(shape):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    args = generate_data(shape)
    executor = check_accuracy(
        mhc_pre_dpskv4,
        args=args,
        standard={
            "torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03),
            "torch.float32": AccuracyStandard(atol=0.03, rtol=0.03),
        },
    )
    check_performance(executor, enable_perf=shape in perf_shape)

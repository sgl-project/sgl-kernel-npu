"""mHC post accuracy and selected-shape performance through public APIs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# DeepSeek-V4-Flash: hidden_size=4096, hc_mult=4; see ../deepseek_v4_shapes.md.
perf_shape = [(1, 4, 4096), (16, 4, 4096)]
mhc_shapes = [
    pytest.param((1, 4, 4096), id="deepseek-v4-flash-decode-t1-hc4-h4096"),
    pytest.param((16, 4, 4096), id="deepseek-v4-flash-decode-t16-hc4-h4096"),
]


def generate_data(shape):
    torch.manual_seed(14)
    tokens, streams, hidden = shape
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    residual = torch.randn(shape, dtype=torch.bfloat16)
    # Fixed independent inputs isolate post accuracy from pre's implementation.
    post = 2 * torch.randn(tokens, streams, dtype=torch.float32).sigmoid()
    comb = torch.randn(tokens, streams, streams, dtype=torch.float32).softmax(-1)
    return x, residual, post, comb


@pytest.mark.parametrize("shape", mhc_shapes)
def test_mhc_post(shape):
    from sgl_kernel_npu.operators.norm import mhc_post

    args = generate_data(shape)
    executor = check_accuracy(
        mhc_post,
        args=args,
        standard={"torch.bfloat16": AccuracyStandard(atol=0.06, rtol=0.03)},
    )
    check_performance(executor, enable_perf=shape in perf_shape)

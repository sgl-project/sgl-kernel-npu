"""mHC post accuracy and selected-shape performance through public APIs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Synthetic single-token and batched decode shapes, not captured model inputs.
perf_shape = [(1, 4, 4096), (16, 4, 4096)]
mhc_shapes = [
    pytest.param((1, 4, 4096), id="synthetic-decode-1x4x4096"),
    pytest.param((16, 4, 4096), id="synthetic-batch-16x4x4096"),
]


@pytest.mark.parametrize("shape", mhc_shapes)
def test_mhc_post(shape):
    from sgl_kernel_npu.operators.norm import mhc_post

    torch.manual_seed(14)
    tokens, streams, hidden = shape
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    residual = torch.randn(shape, dtype=torch.bfloat16)
    # Fixed independent inputs isolate post accuracy from pre's implementation.
    post = 2 * torch.randn(tokens, streams, dtype=torch.float32).sigmoid()
    comb = torch.randn(tokens, streams, streams, dtype=torch.float32).softmax(-1)
    executor = check_accuracy(
        mhc_post,
        args=(x, residual, post, comb),
        standard={"torch.bfloat16": AccuracyStandard(atol=0.06, rtol=0.03)},
    )
    check_performance(executor, enable_perf=shape in perf_shape)

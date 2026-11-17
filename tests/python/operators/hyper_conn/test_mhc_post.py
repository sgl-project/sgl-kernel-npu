import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Source: sglang/srt/configs/glm5_next.py: hidden_size=4096, hc_mult=4.
# GLM-5.3-Flash cases borrow these dimensions; token counts are illustrative,
# not traced production inputs. Use model widths supported on CANN 9.1/9.2.
perf_shape = [(1, 4096), (128, 4096)]

# Avoid the shared BF16 atol=0.05 masking errors in inputs scaled by 0.1.
POST_STANDARD = {
    "torch.float16": AccuracyStandard(atol=1e-5, rtol=2e-3),
    "torch.bfloat16": AccuracyStandard(atol=1e-5, rtol=1e-2),
}


def generate_data(tokens, hidden, dtype):
    torch.manual_seed(43)
    x = torch.randn(tokens, 4, hidden, dtype=dtype) * 0.1
    h_out = torch.randn(tokens, hidden, dtype=dtype) * 0.1
    h_post = torch.rand(tokens, 4, dtype=torch.float32)
    # Deliberately asymmetric: catches a transposed residual mapping.
    h_res = torch.rand(tokens, 4, 4, dtype=torch.float32).softmax(-1)
    # Mimic pre's doubly stochastic residual map without calling pre.
    for _ in range(20):
        h_res = h_res / h_res.sum(-2, keepdim=True)
        h_res = h_res / h_res.sum(-1, keepdim=True)
    return x, h_res, h_out, h_post


@pytest.mark.parametrize(
    "tokens,hidden",
    [
        pytest.param(1, 4096, id="glm53-flash-decode-1x4x4096"),
        pytest.param(128, 4096, id="glm53-flash-extend-128x4x4096"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_post(tokens, hidden, dtype):
    from sgl_kernel_npu.operators.hyper_conn import mhc_post

    inputs = generate_data(tokens, hidden, dtype)
    executor = check_accuracy(mhc_post, args=inputs, standard=POST_STANDARD)
    check_performance(
        executor,
        enable_perf=(tokens, hidden) in perf_shape and dtype == torch.bfloat16,
    )

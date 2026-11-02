import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Source: sglang/srt/configs/glm5_next.py: hidden_size=4096, hc_mult=4.
# GLM-5.3-Flash cases borrow these dimensions; token counts are illustrative,
# not traced production inputs. Small-width cases are synthetic edge tests.
perf_shape = [(1, 4096), (128, 4096)]


def generate_data(tokens, hidden, dtype, mode):
    torch.manual_seed(43)
    x = torch.randn(tokens, 4, hidden, dtype=dtype) * 0.1
    h_out = torch.randn(tokens, hidden, dtype=dtype) * 0.1
    h_post = torch.rand(tokens, 4, dtype=torch.float32)
    # Deliberately asymmetric: catches a transposed residual mapping.
    h_res = torch.rand(tokens, 4, 4, dtype=torch.float32).softmax(-1)
    if mode == "identity":
        h_res = torch.eye(4, dtype=torch.float32).repeat(tokens, 1, 1)
        h_post.zero_()
    elif mode == "post_only":
        h_res.zero_()
    return x, h_res, h_out, h_post


@pytest.mark.parametrize(
    "tokens,hidden,mode",
    [
        pytest.param(0, 128, "normal", id="edge-empty-0x4x128"),
        pytest.param(1, 128, "normal", id="edge-minimum-1x4x128"),
        pytest.param(2, 128, "normal", id="edge-two-rows-2x4x128"),
        pytest.param(5, 256, "normal", id="edge-odd-rows-5x4x256"),
        pytest.param(3, 128, "identity", id="edge-identity-3x4x128"),
        pytest.param(3, 128, "post_only", id="edge-post-only-3x4x128"),
        pytest.param(1, 4096, "normal", id="glm53-flash-decode-1x4x4096"),
        pytest.param(128, 4096, "normal", id="glm53-flash-extend-128x4x4096"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_post(tokens, hidden, mode, dtype):
    from sgl_kernel_npu.operators.hyper_conn import mhc_post

    inputs = generate_data(tokens, hidden, dtype, mode)
    executor = check_accuracy(mhc_post, args=inputs)
    check_performance(
        executor,
        enable_perf=(tokens, hidden) in perf_shape and dtype == torch.bfloat16,
    )

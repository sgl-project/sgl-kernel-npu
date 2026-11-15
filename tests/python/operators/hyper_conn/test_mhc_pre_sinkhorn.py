import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Source: sglang/srt/configs/glm5_next.py: hidden_size=4096, hc_mult=4.
# GLM-5.3-Flash cases borrow these dimensions; token counts are illustrative,
# not traced production inputs. Use model widths supported on CANN 9.1/9.2.
perf_shape = [(1, 4096), (128, 4096)]

# CANN projects with HF32; post/comb remain FP32 but differ from the FP32
# reference GEMM. This operator-specific budget is calibrated on Ascend910_9362
# with CANN 9.1 over nine seeds, both activation dtypes and both eps values.
# These budgets check synthetic operator accuracy, not model-level accuracy.
# BF16/FP16 activation budgets allow dtype rounding and near-zero cancellation.
# The same thresholds also passed on Ascend950DT_9582 with CANN 9.1.
PRE_STANDARD = {
    "torch.float32": AccuracyStandard(atol=5e-5, rtol=5e-5),
    "torch.float16": AccuracyStandard(atol=1e-5, rtol=2e-3),
    "torch.bfloat16": AccuracyStandard(atol=1e-5, rtol=1e-2),
}


def generate_data(tokens, hidden, dtype):
    torch.manual_seed(42)
    x = torch.randn(tokens, 4, hidden, dtype=dtype) * 0.1
    phi = torch.randn(24, 4 * hidden, dtype=torch.float32) / (4 * hidden) ** 0.5
    alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    bias = torch.randn(24, dtype=torch.float32) * 0.1
    return x, phi, alpha, bias


@pytest.mark.parametrize(
    "tokens,hidden",
    [
        pytest.param(1, 4096, id="glm53-flash-decode-1x4x4096"),
        pytest.param(128, 4096, id="glm53-flash-extend-128x4x4096"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hc_eps,norm_eps", [(1e-6, 1e-6), (1e-5, 1e-5)])
def test_mhc_pre_sinkhorn(tokens, hidden, dtype, hc_eps, norm_eps):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

    inputs = generate_data(tokens, hidden, dtype)
    executor = check_accuracy(
        mhc_pre_sinkhorn,
        args=inputs,
        kwargs={"hc_eps": hc_eps, "norm_eps": norm_eps},
        standard=PRE_STANDARD,
    )
    check_performance(
        executor,
        enable_perf=(tokens, hidden) in perf_shape
        and dtype == torch.bfloat16
        and hc_eps == 1e-6,
    )

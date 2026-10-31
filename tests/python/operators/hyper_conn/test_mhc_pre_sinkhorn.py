import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# GLM-5.3-Flash config width; synthetic tokens, not a captured workload.
perf_shape = [(1, 4096), (128, 4096)]


def generate_data(tokens, hidden, dtype, mode):
    torch.manual_seed(42)
    x = torch.randn(tokens, 4, hidden, dtype=dtype) * 0.1
    phi = torch.randn(24, 4 * hidden, dtype=torch.float32) / (4 * hidden) ** 0.5
    alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    bias = torch.randn(24, dtype=torch.float32) * 0.1
    if mode == "zero":
        x.zero_()
    return x, phi, alpha, bias


@pytest.mark.parametrize(
    "tokens,hidden,mode",
    [
        (0, 128, "normal"),
        (1, 128, "normal"),
        (2, 128, "normal"),
        (3, 256, "normal"),
        (5, 128, "zero"),
        (1, 4096, "normal"),
        (128, 4096, "normal"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hc_eps,norm_eps", [(1e-6, 1e-6), (1e-5, 1e-5)])
def test_mhc_pre_sinkhorn(tokens, hidden, mode, dtype, hc_eps, norm_eps):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

    inputs = generate_data(tokens, hidden, dtype, mode)
    executor = check_accuracy(
        mhc_pre_sinkhorn,
        args=inputs,
        kwargs={"hc_eps": hc_eps, "norm_eps": norm_eps},
    )
    check_performance(
        executor,
        enable_perf=(tokens, hidden) in perf_shape
        and dtype == torch.bfloat16
        and hc_eps == 1e-6,
    )

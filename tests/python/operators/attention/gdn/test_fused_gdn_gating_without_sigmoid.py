import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_skip_acc_check
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(16, 8), (32, 32)]
standard = {"torch.float32": AccuracyStandard(5e-2, 1e-2)}


def generate_data(batch, heads, dtype):
    torch.manual_seed(20260326)
    return (
        torch.randn(heads, dtype=dtype),
        torch.randn(batch, heads, dtype=dtype),
        torch.randn(batch, heads, dtype=dtype),
        torch.randn(heads, dtype=dtype),
    )


@pytest.mark.parametrize(
    "batch,heads,dtype,beta,threshold",
    [
        pytest.param(1, 8, torch.float32, 1.0, 20.0, id="minimum-fp32"),
        pytest.param(16, 8, torch.bfloat16, 1.0, 20.0, id="gdn-verify-bf16"),
        pytest.param(64, 16, torch.float16, 0.5, 15.0, id="large-batch-fp16"),
        pytest.param(32, 32, torch.bfloat16, 2.0, 25.0, id="gdn-multi-head"),
    ],
)
def test_gdn_gating(batch, heads, dtype, beta, threshold):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating_without_sigmoid

    args = generate_data(batch, heads, dtype) + (beta, threshold)
    original_beta = args[2].clone()
    executor = check_accuracy(
        fused_gdn_gating_without_sigmoid, args=args, standard=standard
    )
    if not get_skip_acc_check():
        _, raw_beta = executor()
        assert_accuracy(
            raw_beta, original_beta, standard={str(dtype): AccuracyStandard(0, 0)}
        )
        assert raw_beta is args[2]
    check_performance(executor, enable_perf=(batch, heads) in perf_shape)

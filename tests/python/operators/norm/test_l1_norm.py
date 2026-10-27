import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Preserve the legacy L1 tolerance for the FP32 result, including low-precision inputs.
ACCURACY_STANDARD = {
    "torch.float32": AccuracyStandard(atol=0, rtol=5e-3),
}


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize("shape", [(1, 8), (17, 16), (0, 8)])
# Check signed-sum normalization and its FP32 output without near-zero random divisors.
def test_l1_norm(shape, dtype):
    from sgl_kernel_npu.operators.norm import l1_norm

    torch.npu.manual_seed(42)
    x = 0.5 + torch.rand(shape, dtype=dtype)
    x[:, 1::2] *= -0.25
    executor = check_accuracy(
        l1_norm,
        args=(x,),
        impl="triton",
        impl_ref="torch_native",
        standard=ACCURACY_STANDARD,
    )
    result = executor()
    assert result.shape == x.shape and result.dtype == torch.float32
    check_performance(executor)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
# Compare NaN/Inf masks and Inf signs for the original zero-sum behavior explicitly.
def test_l1_norm_zero_sum_classification(dtype):
    from sgl_kernel_npu.operators.norm import l1_norm

    x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]], dtype=dtype)
    result = l1_norm(x, impl="triton")
    reference = l1_norm(x, impl="torch_native")
    assert_accuracy(torch.isnan(result), torch.isnan(reference))
    assert_accuracy(torch.isinf(result), torch.isinf(reference))
    assert_accuracy(
        torch.isinf(result) & torch.signbit(result),
        torch.isinf(reference) & torch.signbit(reference),
    )

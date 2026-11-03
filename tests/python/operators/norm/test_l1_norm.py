import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard, get_standard

INPUT_STANDARDS = {
    torch.float32: get_standard("torch.float32"),
    torch.float16: AccuracyStandard(atol=0, rtol=5e-3),
    torch.bfloat16: AccuracyStandard(atol=0, rtol=5e-3),
}


def generate_test_data(shape, router_experts, dtype):
    torch.npu.manual_seed(42)
    if router_experts is None:
        x = 0.5 + torch.rand(shape, dtype=dtype)
        x[:, 1::2] *= -0.25
    else:
        # Model routing weights are positive top-k softmax scores before renormalization.
        logits = torch.randn(shape[0], router_experts, dtype=torch.float32)
        x = logits.softmax(dim=-1).topk(shape[1], dim=-1).values.to(dtype)
    return x


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "shape,router_experts",
    [
        pytest.param((1, 8), None, id="shape0"),
        pytest.param((17, 16), None, id="shape1"),
        pytest.param((0, 8), None, id="shape2"),
        # Qwen3-30B-A3B routes each token to 8 of 128 experts.
        # Token counts below represent selected decode/prefill workloads.
        pytest.param((1, 8), 128, id="qwen3-30b-a3b-decode-t1-top8"),
        pytest.param((16, 8), 128, id="qwen3-30b-a3b-decode-t16-top8"),
        pytest.param((256, 8), 128, id="qwen3-30b-a3b-prefill-t256-top8"),
    ],
)
# Compare signed inputs and positive MoE routing weights while preserving FP32 output.
def test_l1_norm(shape, router_experts, dtype):
    from sgl_kernel_npu.operators.norm import l1_norm

    x = generate_test_data(shape, router_experts, dtype)
    executor = check_accuracy(
        l1_norm,
        args=(x,),
        # Select by input dtype; the comparison key describes the FP32 output.
        standard={"torch.float32": INPUT_STANDARDS[dtype]},
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

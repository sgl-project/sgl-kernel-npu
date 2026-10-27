import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "shape,zero",
    [
        pytest.param((1, 1, 1), False, id="minimum"),
        pytest.param((2, 17, 128), False, id="row-tail"),
        pytest.param((1, 1024, 512), False, id="baseline-width512"),
        pytest.param((2, 7, 256), True, id="zero"),
        pytest.param((0, 4, 128), False, id="empty-batch"),
    ],
)
# Compare row mean squares, output shape/dtype and zero/empty inputs across backends.
def test_fused_variance(shape, zero, dtype):
    from sgl_kernel_npu.operators.norm import fused_variance

    torch.npu.manual_seed(42)
    x = torch.zeros(shape, dtype=dtype) if zero else torch.randn(shape, dtype=dtype)
    executor = check_accuracy(
        fused_variance, args=(x,), impl="triton", impl_ref="torch_native"
    )
    result = executor()
    assert result.shape == (*shape[:-1], 1) and result.dtype == dtype
    check_performance(executor)

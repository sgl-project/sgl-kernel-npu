"""Public AICPU sample: confirmed inputs, independent reference and rejection cases."""

import pytest

torch = pytest.importorskip("torch")
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_add_example_accuracy(dtype):
    from sgl_kernel_npu.operators.examples import add_example

    x1 = torch.tensor([2, 5], dtype=dtype, device="npu")
    x2 = torch.tensor([3, 3], dtype=dtype, device=x1.device)
    before1, before2 = x1.clone(), x2.clone()
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(add_example, impl)
    else:
        require_distinct_impls(add_example, impl, impl_ref)
        reference = add_example(x1.clone(), x2.clone(), impl=impl_ref)
    result = add_example(x1, x2, impl=impl)
    torch.npu.synchronize()
    import torch_npu

    assert result.is_contiguous() and torch_npu.get_npu_format(result) == 2
    if not skip_accuracy:
        assert_accuracy(result, reference)
        expected = torch.tensor([5, 8], dtype=dtype, device=x1.device)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert result.data_ptr() not in (x1.data_ptr(), x2.data_ptr())
    assert torch.equal(x1, before1)
    assert torch.equal(x2, before2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize(
    "case,message",
    [
        ("broadcast", "identical shapes"),
        ("dtype", "same dtype"),
        ("empty", "empty inputs"),
        ("noncontiguous_x1", "contiguous"),
        ("noncontiguous_x2", "contiguous"),
        ("cpu", "same NPU"),
    ],
)
def test_add_example_rejects_invalid_inputs(dtype, case, message):
    from sgl_kernel_npu.operators.examples import add_example

    x1 = torch.tensor([2, 5], dtype=dtype, device="npu")
    x2 = torch.tensor([3, 3], dtype=dtype, device=x1.device)
    if case == "broadcast":
        x2 = x2[:1]
    elif case == "dtype":
        x2 = x2.to(torch.int32 if dtype == torch.float32 else torch.float32)
    elif case == "empty":
        x1, x2 = x1[:0], x2[:0]
    elif case == "noncontiguous_x1":
        x1 = x1.repeat_interleave(2)[::2]
        assert not x1.is_contiguous()
    elif case == "noncontiguous_x2":
        x2 = x2.repeat_interleave(2)[::2]
        assert not x2.is_contiguous()
    elif case == "cpu":
        x2 = x2.cpu()
    impl, impl_ref = get_test_impls()
    for backend in dict.fromkeys((impl, impl_ref)):
        with pytest.raises(ValueError, match=message):
            add_example(x1, x2, impl=backend)
    # The native adapter also protects direct torch.ops callers.
    with pytest.raises(RuntimeError, match=message):
        torch.ops.npu.sgl_add_example(x1, x2)


def test_add_example_rejects_unsupported_dtype():
    from sgl_kernel_npu.operators.examples import add_example

    x1 = torch.tensor([2, 5], dtype=torch.float16, device="npu")
    x2 = torch.tensor([3, 3], dtype=torch.float16, device=x1.device)
    impl, impl_ref = get_test_impls()
    for backend in dict.fromkeys((impl, impl_ref)):
        with pytest.raises(ValueError, match="float32 or int32"):
            add_example(x1, x2, impl=backend)
    with pytest.raises(RuntimeError, match="float32 or int32"):
        torch.ops.npu.sgl_add_example(x1, x2)

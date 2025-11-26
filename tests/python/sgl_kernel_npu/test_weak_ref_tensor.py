import pytest
import torch
from sgl_kernel_npu.memory import weak_ref_tensor


def test_weak_ref_tensor():
    # Test with NPU tensor
    x = torch.arange(12, device="npu").reshape(3, 4)
    y = weak_ref_tensor(x)

    assert x.data_ptr() == y.data_ptr()
    assert x.shape == y.shape
    assert x.stride() == y.stride()
    assert x.dtype == y.dtype
    assert x.device == y.device

    # Test with CPU tensor (should raise an error)
    with pytest.raises(RuntimeError):
        x_cpu = torch.rand(2, 2, device="cpu")
        weak_ref_tensor(x_cpu)

    # Test with non-tensor inputs
    assert weak_ref_tensor(None) is None
    assert weak_ref_tensor(123) == 123
    s = "a string"
    assert weak_ref_tensor(s) is s


if __name__ == "__main__":
    pytest.main([__file__])

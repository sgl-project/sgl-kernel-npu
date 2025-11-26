import pytest
import torch
from sgl_kernel_npu.memory import weak_ref_tensor


def test_weak_ref_tensor():
    x = torch.arange(12, device="npu").reshape(3, 4)
    y = weak_ref_tensor(x)

    assert x.data_ptr() == y.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__])

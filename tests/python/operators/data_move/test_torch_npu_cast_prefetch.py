"""Numerical and smoke tests for the public TorchNPU data-move API."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("shape", [(1,), (7, 33)], ids=["single-value", "odd-tail"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_dtype_cast(shape, dtype):
    from sgl_kernel_npu.operators.data_move import dtype_cast

    x = torch.randn(shape).mul(4)
    result = dtype_cast(x, dtype, impl="torch_npu")
    torch.testing.assert_close(result.cpu(), x.cpu().to(dtype), rtol=0, atol=0)


def test_format_cast_to_nd():
    import torch_npu
    from sgl_kernel_npu.operators.data_move import format_cast

    x = torch.arange(32, dtype=torch.float16).view(2, 16)
    result = format_cast(x, 2)
    assert torch_npu.get_npu_format(result) == 2
    torch.testing.assert_close(result, x, rtol=0, atol=0)


def test_prefetch_arch22():
    from sgl_kernel_npu.operators.data_move import prefetch
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH22:
        pytest.skip("prefetch is registered for arch22")
    x = torch.ones(2048, dtype=torch.float16)
    assert prefetch(x, x, x.numel() * x.element_size()) is None
    torch.npu.synchronize()
    torch.testing.assert_close(x.cpu(), torch.ones_like(x, device="cpu"))

"""On-device production/reference checks for rope operators."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_partial_rope(impl_pair):
    from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

    impl, reference = impl_pair
    x = torch.randn(16, 1, 64, 512, device="npu", dtype=torch.bfloat16)
    angles = torch.randn(16, 1, 1, 32, device="npu").repeat_interleave(2, -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    a, e = x.clone(), x.clone()
    kwargs = dict(rotary_mode="interleave", partial_slice=[448, 512])
    inplace_partial_rotary_mul(a, cos, sin, impl=impl, **kwargs)
    inplace_partial_rotary_mul(e, cos, sin, impl=reference, **kwargs)
    torch.testing.assert_close(a, e, atol=0.03, rtol=0.03)

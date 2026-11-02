"""On-device production/reference checks for norm operators."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_mhc_pre_post(impl_pair):
    from sgl_kernel_npu.operators.norm import mhc_post, mhc_pre

    impl, reference = impl_pair
    torch.manual_seed(11)
    x = torch.randn(16, 4, 4096, device="npu", dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * 4096, device="npu") * 0.01
    scale = torch.ones(3, device="npu")
    base = torch.zeros(24, device="npu")
    actual = mhc_pre(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl=impl)
    expected = mhc_pre(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl=reference)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.03, rtol=0.03)
    y, post, comb = expected
    a = mhc_post(y, x, post, comb, impl=impl)
    e = mhc_post(y, x, post, comb, impl=reference)
    torch.testing.assert_close(a, e, atol=0.06, rtol=0.03)

"""mHC facades preserve vendor results and delegate support checks."""

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")


def load_mhc(package_factory, monkeypatch, generation):
    package = package_factory()
    hardware = package.generation.Hardware(
        2201 if generation == "ARCH22" else 3510,
        getattr(package.generation.Generation, generation),
    )
    monkeypatch.setattr(package.generation, "get_hardware", lambda: hardware)
    modules = package.load()
    api = modules.import_module("operators.hyper_conn")
    vendor = ModuleType("cann_ops_transformer.ops")
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", vendor)
    return api, vendor


@pytest.mark.parametrize("generation", ["ARCH22", "ARCH35"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pre_preserves_vendor_dtype_and_delegates_constraints(
    package_factory, monkeypatch, generation, dtype
):
    api, vendor = load_mhc(package_factory, monkeypatch, generation)
    # An external-backend stub accepts these values; this does not declare
    # that a real CANN kernel supports the shape, stride or attribute values.
    x = torch.full((2, 3, 258), 0.1001, dtype=torch.bfloat16)[..., ::2]
    phi = torch.zeros(15, 387)
    alpha = torch.ones(6)[::2]
    bias = torch.zeros(15)

    def pre(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
        y = (x.float().sum(2) + hc_eps).to(dtype)
        post = torch.full((2, 1, hc_mult), num_iters, dtype=torch.float32)
        comb = torch.full((2, 1, hc_mult, hc_mult), norm_eps)
        return y, post, comb

    vendor.mhc_pre_sinkhorn = pre
    y, post, comb = api.mhc_pre_sinkhorn(
        x, phi, alpha, bias, hc_mult=3, num_iters=7, hc_eps=0.1, norm_eps=-1.0
    )
    expected = (x.float().sum(1) + 0.1).to(dtype)
    assert y.dtype == dtype
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    torch.testing.assert_close(post, torch.full((2, 3), 7.0))
    torch.testing.assert_close(comb, torch.full((2, 3, 3), -1.0))


@pytest.mark.parametrize("generation", ["ARCH22", "ARCH35"])
def test_post_delegates_constraints_and_preserves_output(
    package_factory, monkeypatch, generation
):
    api, vendor = load_mhc(package_factory, monkeypatch, generation)
    x = torch.full((2, 3, 258), 0.1001, dtype=torch.bfloat16)[..., ::2]
    h_res = torch.eye(3).repeat(2, 1, 1)
    h_out = torch.ones(2, 129)
    h_post = torch.full((2, 3), 2.0)

    def post(x, h_res, h_out, h_post):
        return torch.bmm(h_res, x.float()) + h_post[..., None] * h_out[:, None]

    vendor.mhc_post = post
    result = api.mhc_post(x, h_res, h_out, h_post)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, x.float() + 2, atol=0, rtol=0)

"""Forwarding and backend selection for the activation, norm, RoPE and sampling APIs."""

import sys
from unittest.mock import Mock, sentinel

import pytest


@pytest.mark.parametrize("explicit_impl", [False, True], ids=["auto", "explicit"])
@pytest.mark.parametrize(
    "category, name, args, kwargs, forwarded_args, forwarded_kwargs",
    [
        ("activation", "fast_gelu", (sentinel.x,), {}, (sentinel.x,), {}),
        (
            "activation",
            "geglu",
            (sentinel.x,),
            {"dim": 1, "approximate": 0, "activate_left": True},
            (sentinel.x, 1, 0, True),
            {},
        ),
        ("activation", "swiglu", (sentinel.x,), {}, (sentinel.x, -1), {}),
        (
            "norm",
            "rms_norm",
            (sentinel.x, sentinel.gamma),
            {"epsilon": 1e-5},
            (sentinel.x, sentinel.gamma, 1e-5),
            {},
        ),
        (
            "norm",
            "add_rms_norm",
            (sentinel.x, sentinel.residual, sentinel.gamma),
            {},
            (sentinel.x, sentinel.residual, sentinel.gamma, 1e-6),
            {},
        ),
        (
            "rope",
            "interleave_rope",
            (sentinel.x, sentinel.cos, sentinel.sin),
            {},
            (sentinel.x, sentinel.cos, sentinel.sin),
            {},
        ),
        (
            "rope",
            "rotary_mul",
            (sentinel.x, sentinel.cos, sentinel.sin),
            {},
            (sentinel.x, sentinel.cos, sentinel.sin),
            {},
        ),
        (
            "rope",
            "rotary_mul",
            (sentinel.x, sentinel.cos, sentinel.sin),
            {"rotary_mode": "interleave", "rotate": sentinel.matrix},
            (sentinel.x, sentinel.cos, sentinel.sin),
            {"rotary_mode": "interleave", "rotate": sentinel.matrix},
        ),
        (
            "rope",
            "mrope",
            (sentinel.positions, sentinel.q, sentinel.k, sentinel.cache, 64),
            {"mrope_section": [0, 0, 0], "rotary_mode": "interleave"},
            (sentinel.positions, sentinel.q, sentinel.k, sentinel.cache, 64),
            {"mrope_section": [0, 0, 0], "rotary_mode": "interleave"},
        ),
        (
            "rope",
            "mrope",
            (sentinel.positions, sentinel.q, sentinel.k, sentinel.cache, 128),
            {"mrope_section": [16, 24, 24], "cache_mode": "interleave"},
            (sentinel.positions, sentinel.q, sentinel.k, sentinel.cache, 128),
            {
                "mrope_section": [16, 24, 24],
                "rotary_mode": "half",
                "cache_mode": "interleave",
            },
        ),
        (
            "sampler",
            "top_k_top_p",
            (sentinel.logits,),
            {"k": sentinel.k},
            (sentinel.logits, None, sentinel.k),
            {},
        ),
    ],
)
def test_torch_npu_migration_forwarding(
    registry_for_generation,
    monkeypatch,
    explicit_impl,
    category,
    name,
    args,
    kwargs,
    forwarded_args,
    forwarded_kwargs,
):
    vendor_impl = "ops-transformer" if category == "rope" else "ops-nn"
    impl = vendor_impl if explicit_impl else "auto"
    modules = registry_for_generation("ARCH22")
    operator = getattr(modules.import_module(f"operators.{category}"), name)
    npu_fn = Mock(return_value=sentinel.result)
    monkeypatch.setattr(sys.modules["torch_npu"], f"npu_{name}", npu_fn, raising=False)

    assert modules.public_api.list_impls(operator) == [
        "auto",
        vendor_impl,
        "torch_native",
    ]
    assert modules.public_api.get_impl(operator, "auto") is modules.public_api.get_impl(
        operator, vendor_impl
    )
    assert operator(*args, impl=impl, **kwargs) is sentinel.result
    npu_fn.assert_called_once_with(*forwarded_args, **forwarded_kwargs)


def test_mrope_reports_unsupported_vendor_generation(registry_for_generation):
    modules = registry_for_generation("ARCH35")
    operator = modules.import_module("operators.rope").mrope

    with pytest.raises(modules.registry.UnsupportedGenerationError):
        operator(None, None, None, None, 128)
    with pytest.raises(modules.registry.UnsupportedGenerationError):
        modules.public_api.get_impl(operator, "ops-transformer")
    assert callable(modules.public_api.get_impl(operator, "torch_native"))


def test_torch_npu_backend_error_propagates(registry_for_generation, monkeypatch):
    modules = registry_for_generation("ARCH22")
    operator = modules.import_module("operators.activation").fast_gelu
    error = RuntimeError("FastGELU backend failed")
    backend = Mock(side_effect=error)
    monkeypatch.setattr(
        sys.modules["torch_npu"], "npu_fast_gelu", backend, raising=False
    )

    with pytest.raises(RuntimeError) as caught:
        operator(sentinel.x)
    assert caught.value is error
    backend.assert_called_once_with(sentinel.x)


@pytest.mark.parametrize("impl", ["auto", "ops-nn", "torch_native"])
def test_top_k_top_p_requires_at_least_one_threshold(registry_for_generation, impl):
    modules = registry_for_generation("ARCH22")
    operator = modules.import_module("operators.sampler").top_k_top_p
    with pytest.raises(ValueError, match="at least one of p or k"):
        operator(sentinel.logits, impl=impl)

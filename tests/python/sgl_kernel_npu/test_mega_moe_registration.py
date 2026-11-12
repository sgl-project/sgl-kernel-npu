"""MegaMoE registration contracts using isolated source imports and fake providers."""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "python/sgl_kernel_npu/sgl_kernel_npu"


@pytest.fixture(params=[2201, 3510], ids=["arch22", "arch35"])
def moe_modules(request, monkeypatch):
    namespace = "_mega_moe_registration_test"
    package = ModuleType(namespace)
    package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, namespace, package)
    # Load generation before registry so no native runtime or hardware is needed.
    name = f"{namespace}.operators.utils.generation"
    spec = importlib.util.spec_from_file_location(
        name, SOURCE / "operators/utils/generation.py"
    )
    generation = importlib.util.module_from_spec(spec)
    sys.modules[name] = generation
    spec.loader.exec_module(generation)
    hardware = generation.Hardware(
        request.param, generation.generation_from_arch(request.param)
    )
    monkeypatch.setattr(generation, "get_hardware", lambda: hardware)
    # The upstream MoE package also exports a torch-backed gating operator.
    # Isolate that sibling so registration contracts still run without torch.
    gating = ModuleType(f"{namespace}.operators.moe.moe_gating_top_k")
    gating.moe_gating_top_k = Mock()
    monkeypatch.setitem(sys.modules, gating.__name__, gating)
    try:
        module = importlib.import_module(f"{namespace}.operators.moe.mega_moe")
        registry = importlib.import_module(f"{namespace}.operators.utils.registry")
        yield module, registry
    finally:
        for name in list(sys.modules):
            if name.startswith(namespace + "."):
                del sys.modules[name]


@pytest.mark.parametrize("operator_name", ["mega_moe", "get_symm_buffer_for_mega_moe"])
def test_registration_and_aliases(moe_modules, operator_name):
    module, registry = moe_modules
    operator = getattr(module, operator_name)
    assert registry.get_op_name(operator) == f"moe.{operator_name}"
    assert registry.list_impls(operator) == ["auto", "cann_ops_transformer"]
    assert registry.get_impl(operator, "auto") is registry.get_impl(
        operator, "cann_ops_transformer"
    )


@pytest.mark.parametrize("operator_name", ["mega_moe", "get_symm_buffer_for_mega_moe"])
@pytest.mark.parametrize("impl", [None, "auto", "cann_ops_transformer"])
def test_lazy_loading_and_argument_forwarding(
    moe_modules, monkeypatch, operator_name, impl
):
    module, _ = moe_modules
    assert module._mega_moe_fn is None
    assert module._get_symm_buffer_fn is None
    backend = ModuleType("cann_ops_transformer.ops")
    backend.mega_moe = Mock(return_value=object())
    backend.get_symm_buffer_for_mega_moe = Mock(return_value=object())
    parent = ModuleType("cann_ops_transformer")
    parent.ops = backend
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", parent)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", backend)
    provider = getattr(backend, operator_name)
    operator = getattr(module, operator_name)
    argument = object()
    selection = {} if impl is None else {"impl": impl}
    assert operator(argument, option=3, **selection) is provider.return_value
    provider.assert_called_once_with(argument, option=3)
    other = (
        "mega_moe" if operator_name != "mega_moe" else "get_symm_buffer_for_mega_moe"
    )
    getattr(backend, other).assert_not_called()
    # The loaded functions remain usable without reimporting the optional package.
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", None)
    assert operator(argument, option=3, **selection) is provider.return_value
    assert provider.call_count == 2


@pytest.mark.parametrize("operator_name", ["mega_moe", "get_symm_buffer_for_mega_moe"])
@pytest.mark.parametrize("impl", ["missing", "cann"])
def test_unknown_impl_fails_before_backend_load(
    moe_modules, monkeypatch, operator_name, impl
):
    module, registry = moe_modules
    loader = Mock(side_effect=AssertionError("backend must stay unloaded"))
    monkeypatch.setattr(module, "_load_backend", loader)
    with pytest.raises(registry.ImplementationNotFoundError):
        getattr(module, operator_name)(impl=impl)
    loader.assert_not_called()


@pytest.mark.parametrize("operator_name", ["mega_moe", "get_symm_buffer_for_mega_moe"])
def test_missing_provider_can_retry(moe_modules, monkeypatch, operator_name):
    module, _ = moe_modules
    operator = getattr(module, operator_name)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    with pytest.raises(ModuleNotFoundError):
        operator()
    assert module._mega_moe_fn is None
    assert module._get_symm_buffer_fn is None

    backend = ModuleType("cann_ops_transformer.ops")
    backend.mega_moe = Mock(return_value=object())
    backend.get_symm_buffer_for_mega_moe = Mock(return_value=object())
    parent = ModuleType("cann_ops_transformer")
    parent.ops = backend
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", parent)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", backend)
    assert operator() is getattr(backend, operator_name).return_value
    getattr(backend, operator_name).assert_called_once_with()


@pytest.mark.parametrize("operator_name", ["mega_moe", "get_symm_buffer_for_mega_moe"])
def test_public_exports_and_keyword_only_impl(moe_modules, operator_name):
    module, _ = moe_modules
    public = importlib.import_module(module.__package__)
    operator = getattr(module, operator_name)
    assert getattr(public, operator_name) is operator
    assert operator_name in public.__all__
    selection = inspect.signature(operator).parameters["impl"]
    assert selection.kind is inspect.Parameter.KEYWORD_ONLY
    assert selection.default == "auto"


@pytest.mark.parametrize("operator_name", ["mega_moe", "get_symm_buffer_for_mega_moe"])
@pytest.mark.parametrize("impl", ["auto", "cann_ops_transformer"])
def test_provider_failure_propagates_without_fallback(
    moe_modules, monkeypatch, operator_name, impl
):
    module, _ = moe_modules
    error = RuntimeError("provider rejected shape or dtype")
    compute, create_buffer = Mock(), Mock()
    provider = compute if operator_name == "mega_moe" else create_buffer
    provider.side_effect = error
    monkeypatch.setattr(module, "_load_backend", lambda: (compute, create_buffer))
    with pytest.raises(RuntimeError) as caught:
        getattr(module, operator_name)(impl=impl)
    assert caught.value is error
    provider.assert_called_once_with()
    (create_buffer if operator_name == "mega_moe" else compute).assert_not_called()


def test_buffer_lifetime_and_optional_arguments_are_owned_by_caller(
    moe_modules, monkeypatch
):
    module, _ = moe_modules
    buffer = Mock()
    compute, create_buffer = Mock(return_value=(object(), object())), Mock(
        return_value=buffer
    )
    monkeypatch.setattr(module, "_load_backend", lambda: (compute, create_buffer))
    assert module.get_symm_buffer_for_mega_moe(object()) is buffer
    options = dict(
        x=object(),
        topk_ids=object(),
        topk_weights=object(),
        l1_weights=[object()],
        l2_weights=[object()],
        sym_buffer=buffer,
        l1_weights_sf=[object()],
        l2_weights_sf=[object()],
        l1_bias=None,
        l2_bias=None,
        x_active_mask=None,
        activation="situglu",
        activation_params={"beta": 4.0, "linear_beta": 25.0},
        weight1_type=object(),
        weight2_type=object(),
    )
    assert module.mega_moe(**options) is compute.return_value
    compute.assert_called_once_with(**options)
    buffer.destroy.assert_not_called()

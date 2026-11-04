"""CPU checks for the public MegaMoe routing and context contract."""

import importlib.util
import sys
from pathlib import Path
from threading import Event, Thread
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def api(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    path = root / "python/sgl_kernel_npu/sgl_kernel_npu/operators/moe/zerc_moe.py"
    for name in (
        "_megamoe_test",
        "_megamoe_test.operators",
        "_megamoe_test.operators.moe",
        "_megamoe_test.operators.utils",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    generation = ModuleType("_megamoe_test.operators.utils.generation")
    generation.Generation = SimpleNamespace(ARCH35=object())
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    torch_npu = ModuleType("torch_npu")
    torch_npu.get_npu_format = lambda tensor: tensor.format
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    registry = ModuleType("_megamoe_test.operators.utils.registry")
    implementations = {}

    def register_impl(name, impl, **_kwargs):
        def register(function):
            implementations[(name, impl)] = function
            return function

        return register

    registry.register_impl = register_impl
    registry.register_operator = lambda _name: lambda function: function
    registry.get_impl_by_name = lambda name, impl: implementations[(name, impl)]
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    name = "_megamoe_test.operators.moe.zerc_moe"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    backend = SimpleNamespace(
        get_symm_buffer_for_mega_moe=Mock(
            return_value=SimpleNamespace(
                destroy=Mock(),
                context=SimpleNamespace(device=torch.device("privateuseone:0")),
            )
        ),
        mega_moe=Mock(return_value=("output", "counts")),
    )
    monkeypatch.setattr(module, "_load_backend", lambda: backend)
    return module, backend


def _tensor(shape, dtype, device="privateuseone:0"):
    return SimpleNamespace(
        shape=torch.Size(shape),
        ndim=len(shape),
        dtype=dtype,
        device=torch.device(device),
        is_contiguous=lambda: True,
        format=2,
    )


def _inputs():
    return (
        _tensor((2, 16), torch.bfloat16),
        _tensor((2, 1), torch.int32),
        _tensor((2, 1), torch.bfloat16),
        [_tensor((1, 8, 16), torch.bfloat16)],
        [_tensor((1, 16, 4), torch.bfloat16)],
    )


def test_group_context_reuse_and_retirement(api):
    module, backend = api
    group = object()
    buffer = module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4)
    assert module.zerc_moe(*_inputs(), buffer) == ("output", "counts")
    assert module.zerc_moe(*_inputs(), buffer) == ("output", "counts")
    assert backend.get_symm_buffer_for_mega_moe.call_count == 1
    assert backend.mega_moe.call_count == 2
    buffer.destroy()
    buffer.destroy()
    buffer._buffer.destroy.assert_called_once()
    with pytest.raises(ValueError, match="live"):
        module.zerc_moe(*_inputs(), buffer)
    with pytest.raises(RuntimeError, match="already created"):
        module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4)


def test_failed_context_creation_is_not_retried_for_same_group(api):
    module, backend = api
    group = object()
    backend.get_symm_buffer_for_mega_moe.side_effect = RuntimeError("context error")
    with pytest.raises(RuntimeError, match="context error"):
        module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4)
    with pytest.raises(RuntimeError, match="already created"):
        module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4)
    assert backend.get_symm_buffer_for_mega_moe.call_count == 1


def test_invalid_capacity_and_shapes_fail_before_kernel(api):
    module, backend = api
    with pytest.raises(ValueError, match="divisible by 16"):
        module.create_zerc_moe_buffer(object(), 2, 8, 1, 15, 4)
    backend.get_symm_buffer_for_mega_moe.assert_not_called()
    buffer = module.create_zerc_moe_buffer(object(), 2, 1, 1, 16, 4)
    with pytest.raises(ValueError, match="capacity"):
        module.zerc_moe(*_inputs(), buffer)
    backend.mega_moe.assert_not_called()


def test_separate_groups_do_not_share_buffer(api):
    module, backend = api
    first = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    second = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    assert first is not second
    assert backend.get_symm_buffer_for_mega_moe.call_count == 2


def test_invalid_shared_expert_fails_before_kernel(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    bad_first = [_tensor((1, 8, 16), torch.bfloat16)]
    bad_second = [_tensor((1, 15, 4), torch.bfloat16)]
    with pytest.raises(ValueError, match="second shared"):
        module.zerc_moe(
            *_inputs(),
            buffer,
            l1_weights_shared=bad_first,
            l2_weights_shared=bad_second,
        )
    backend.mega_moe.assert_not_called()


def test_explicit_group_is_required(api):
    module, backend = api
    with pytest.raises(ValueError, match="explicit process group"):
        module.create_zerc_moe_buffer(None, 2, 8, 1, 16, 4)
    backend.get_symm_buffer_for_mega_moe.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "privateuseone:1"])
def test_mixed_or_cpu_tensors_fail_before_kernel(api, device):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    x, ids, probabilities, first, second = _inputs()
    if device == "cpu":
        x.device = torch.device(device)
    else:
        ids.device = torch.device(device)
    with pytest.raises(ValueError, match="NPU device"):
        module.zerc_moe(x, ids, probabilities, first, second, buffer)
    backend.mega_moe.assert_not_called()


def test_nz_storage_is_rejected_before_kernel(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    x, ids, probabilities, first, second = _inputs()
    first[0].format = 29
    with pytest.raises(ValueError, match="ND storage"):
        module.zerc_moe(x, ids, probabilities, first, second, buffer)
    backend.mega_moe.assert_not_called()


def test_buffer_and_input_devices_must_match(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    buffer._buffer.context.device = torch.device("privateuseone:1")
    with pytest.raises(ValueError, match="context device"):
        module.zerc_moe(*_inputs(), buffer)
    backend.mega_moe.assert_not_called()


def test_mxfp8_requires_matching_weight_and_scale_types(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(
        object(), 2, 8, 1, 16, 64, dispatch_quant_mode=4, dispatch_quant_out_dtype=24
    )
    x, ids, probabilities, _, _ = _inputs()
    first = [_tensor((1, 128, 16), torch.float8_e4m3fn)]
    second = [_tensor((1, 16, 64), torch.float8_e4m3fn)]
    scales1 = [_tensor((1, 128, 1, 2), torch.float8_e8m0fnu)]
    scales2 = [_tensor((1, 16, 1, 2), torch.float8_e8m0fnu)]
    assert module.zerc_moe(
        x,
        ids,
        probabilities,
        first,
        second,
        buffer,
        l1_weights_sf=scales1,
        l2_weights_sf=scales2,
    ) == ("output", "counts")
    scales2[0].dtype = torch.float32
    with pytest.raises(ValueError, match="E8M0"):
        module.zerc_moe(
            x,
            ids,
            probabilities,
            first,
            second,
            buffer,
            l1_weights_sf=scales1,
            l2_weights_sf=scales2,
        )
    assert backend.mega_moe.call_count == 1


def test_destroy_waits_for_an_inflight_host_launch(api):
    module, backend = api
    entered = Event()
    release = Event()
    destroy_started = Event()
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)

    def launch(*_args, **_kwargs):
        entered.set()
        assert release.wait(3)
        return "output", "counts"

    backend.mega_moe.side_effect = launch
    caller = Thread(target=lambda: module.zerc_moe(*_inputs(), buffer))
    caller.start()
    assert entered.wait(3)

    def destroy():
        destroy_started.set()
        buffer.destroy()

    closer = Thread(target=destroy)
    closer.start()
    assert destroy_started.wait(3)
    buffer._buffer.destroy.assert_not_called()
    release.set()
    caller.join(3)
    closer.join(3)
    assert not caller.is_alive()
    assert not closer.is_alive()
    buffer._buffer.destroy.assert_called_once()

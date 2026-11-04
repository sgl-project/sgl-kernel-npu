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


def test_original_megamoe_signatures_are_preserved(api):
    import ast
    import inspect

    module, _ = api
    root = Path(__file__).resolve().parents[3]
    original = ast.parse(
        (
            root
            / "csrc/zerc_moe/source/torch_extension/npu_ops_transformer/ops/mega_moe.py"
        ).read_text()
    )
    pairs = (
        ("mega_moe", module.zerc_moe),
        ("get_symm_buffer_for_mega_moe", module.create_zerc_moe_buffer),
    )
    for original_name, public in pairs:
        function = next(
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name == original_name
        )
        params = inspect.signature(public).parameters
        names = [arg.arg for arg in function.args.args]
        assert list(params)[: len(names)] == names
        assert all(
            params[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            for name in names
        )
        required = len(names) - len(function.args.defaults)
        for name in names[:required]:
            assert params[name].default is inspect.Parameter.empty
        for name, default in zip(names[required:], function.args.defaults):
            assert params[name].default == ast.literal_eval(default)
        if original_name == "mega_moe":
            assert list(params)[len(names) :] == ["impl"]
            assert params["impl"].kind == inspect.Parameter.KEYWORD_ONLY
            assert params["impl"].default == "auto"
        else:
            assert len(params) == len(names)


@pytest.mark.parametrize("by_keyword", [False, True])
def test_original_arguments_reach_the_backend_without_shifting(api, by_keyword):
    module, backend = api
    group = object()
    if by_keyword:
        buffer = module.create_zerc_moe_buffer(
            group=group,
            num_experts=2,
            num_max_tokens_per_rank=8,
            num_topk=1,
            hidden=16,
            intermediate_hidden=4,
            max_recv_token_num=0,
            dispatch_quant_mode=0,
            dispatch_quant_out_dtype=28,
            combine_quant_mode=0,
            comm_alg="",
        )
    else:
        buffer = module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4, 0, 0, 28, 0, "")
    backend.get_symm_buffer_for_mega_moe.assert_called_once_with(
        group,
        num_experts=2,
        num_max_tokens_per_rank=8,
        num_topk=1,
        hidden=16,
        intermediate_hidden=4,
        max_recv_token_num=0,
        dispatch_quant_mode=0,
        dispatch_quant_out_dtype=28,
        combine_quant_mode=0,
        comm_alg="",
    )
    x, ids, weights, l1, l2 = _inputs()
    shared1 = [_tensor((1, 32, 16), torch.bfloat16)]
    shared2 = [_tensor((1, 16, 16), torch.bfloat16)]
    # Arbitrary mock size: the real host validates the version-specific length.
    timer = _tensor((10,), torch.int64)
    optional = dict(
        scales=None,
        l1_weights_sf=None,
        l2_weights_sf=None,
        x_active_mask=None,
        weight1_type=28,
        weight2_type=28,
        timer=timer,
        l1_weights_shared=shared1,
        l2_weights_shared=shared2,
        l1_weights_shared_sf=None,
        l2_weights_shared_sf=None,
    )
    if by_keyword:
        result = module.zerc_moe(
            x=x,
            topk_ids=ids,
            topk_weights=weights,
            l1_weights=l1,
            l2_weights=l2,
            sym_buffer=buffer,
            **optional,
        )
    else:
        result = module.zerc_moe(x, ids, weights, l1, l2, buffer, *optional.values())
    assert result == ("output", "counts")
    backend.mega_moe.assert_called_once_with(
        x, ids, weights, l1, l2, buffer._buffer, **optional
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("scales", object()),
        ("x_active_mask", object()),
        ("weight1_type", 1),
        ("weight2_type", 1),
    ],
)
def test_unsupported_original_call_options_fail_before_launch(api, name, value):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    with pytest.raises(ValueError):
        module.zerc_moe(*_inputs(), buffer, **{name: value})
    backend.mega_moe.assert_not_called()


@pytest.mark.parametrize(
    "name,value", [("combine_quant_mode", 1), ("comm_alg", "other")]
)
def test_unsupported_original_buffer_options_do_not_reserve_group(api, name, value):
    module, backend = api
    group = object()
    with pytest.raises(ValueError):
        module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4, **{name: value})
    backend.get_symm_buffer_for_mega_moe.assert_not_called()
    module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 4)
    backend.get_symm_buffer_for_mega_moe.assert_called_once()


def test_original_buffer_metadata_and_context_are_exposed(api):
    module, backend = api
    underlying = backend.get_symm_buffer_for_mega_moe.return_value
    names = (
        "rank_id",
        "group_name",
        "ep_world_size",
        "ccl_buffer_size",
        "num_experts",
        "max_recv_token_num",
        "num_max_tokens_per_rank",
        "num_topk",
        "hidden",
        "intermediate_hidden",
        "dispatch_quant_mode",
        "dispatch_quant_out_dtype",
        "combine_quant_mode",
        "comm_alg",
        "global_bs",
    )
    for index, name in enumerate(names):
        setattr(underlying, name, index)
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    for index, name in enumerate(names):
        assert getattr(buffer, name) == index
        if name in (
            "rank_id",
            "group_name",
            "num_max_tokens_per_rank",
            "num_topk",
            "hidden",
            "intermediate_hidden",
        ):
            setattr(buffer, name, -1)
            assert getattr(underlying, name) == -1
        else:
            with pytest.raises(AttributeError):
                setattr(buffer, name, -1)
    assert buffer.context is underlying.context
    buffer.destroy()
    with pytest.raises(ValueError, match="destroyed"):
        _ = buffer.context
    with pytest.raises(ValueError, match="destroyed"):
        buffer.intermediate_hidden = 3


def test_writable_sizing_metadata_does_not_enlarge_allocation(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 1, 1, 16, 4)
    buffer.num_max_tokens_per_rank = 8
    buffer.num_topk = 2
    buffer.hidden = 32
    assert buffer.config["num_max_tokens_per_rank"] == 8
    assert buffer.config["num_topk"] == 2
    assert buffer.config["hidden"] == 32
    with pytest.raises(ValueError, match="capacity"):
        module.zerc_moe(*_inputs(), buffer)
    backend.mega_moe.assert_not_called()


def test_original_positional_quantized_scales_are_not_shifted(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 64, 0, 4, 24, 0, "")
    x, ids, weights, _, _ = _inputs()
    l1 = [_tensor((1, 128, 16), torch.float8_e4m3fn)]
    l2 = [_tensor((1, 16, 64), torch.float8_e4m3fn)]
    sf1 = [_tensor((1, 128, 1, 2), torch.float8_e8m0fnu)]
    sf2 = [_tensor((1, 16, 1, 2), torch.float8_e8m0fnu)]
    shared1 = [_tensor((1, 128, 16), torch.float8_e4m3fn)]
    shared2 = [_tensor((1, 16, 64), torch.float8_e4m3fn)]
    shared_sf1 = [_tensor((1, 128, 1, 2), torch.float8_e8m0fnu)]
    shared_sf2 = [_tensor((1, 16, 1, 2), torch.float8_e8m0fnu)]
    assert module.zerc_moe(
        x,
        ids,
        weights,
        l1,
        l2,
        buffer,
        None,
        sf1,
        sf2,
        None,
        28,
        28,
        None,
        shared1,
        shared2,
        shared_sf1,
        shared_sf2,
    ) == ("output", "counts")
    assert backend.mega_moe.call_args.kwargs == dict(
        scales=None,
        l1_weights_sf=sf1,
        l2_weights_sf=sf2,
        x_active_mask=None,
        weight1_type=28,
        weight2_type=28,
        timer=None,
        l1_weights_shared=shared1,
        l2_weights_shared=shared2,
        l1_weights_shared_sf=shared_sf1,
        l2_weights_shared_sf=shared_sf2,
    )


def test_timer_must_share_device_and_storage_format(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 4)
    timer = _tensor((10,), torch.int64, device="cpu")
    with pytest.raises(ValueError, match="NPU device"):
        module.zerc_moe(*_inputs(), buffer, timer=timer)
    timer.device = torch.device("privateuseone:0")
    timer.format = 29
    with pytest.raises(ValueError, match="ND storage"):
        module.zerc_moe(*_inputs(), buffer, timer=timer)
    backend.mega_moe.assert_not_called()


def test_buffer_intermediate_metadata_does_not_constrain_weight_shapes(api):
    module, backend = api
    group = object()
    buffer = module.create_zerc_moe_buffer(group, 2, 8, 1, 16, 0)
    x, ids, probs, w1, w2 = _inputs()
    assert module.zerc_moe(x, ids, probs, w1, w2, buffer) == ("output", "counts")
    buffer.intermediate_hidden = 99
    assert buffer.config["intermediate_hidden"] == 99
    w1_second = [_tensor((1, 12, 16), torch.bfloat16)]
    w2_second = [_tensor((1, 16, 6), torch.bfloat16)]
    assert module.zerc_moe(x, ids, probs, w1_second, w2_second, buffer) == (
        "output",
        "counts",
    )
    assert backend.get_symm_buffer_for_mega_moe.call_count == 1
    assert backend.mega_moe.call_count == 2
    with pytest.raises(ValueError, match="first expert weight shape"):
        module.zerc_moe(x, ids, probs, w1, w2_second, buffer)
    with pytest.raises(ValueError, match="three-dimensional"):
        module.zerc_moe(x, ids, probs, w1, [_tensor((1, 16), torch.bfloat16)], buffer)
    assert backend.mega_moe.call_count == 2


def test_mxfp8_scales_follow_each_calls_actual_weight_shape(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(
        object(), 2, 8, 1, 16, 0, dispatch_quant_mode=4, dispatch_quant_out_dtype=24
    )
    x, ids, probs, _, _ = _inputs()
    for size in (64, 128):
        w1 = [_tensor((1, 2 * size, 16), torch.float8_e4m3fn)]
        w2 = [_tensor((1, 16, size), torch.float8_e4m3fn)]
        sf1 = [_tensor((1, 2 * size, 1, 2), torch.float8_e8m0fnu)]
        sf2 = [_tensor((1, 16, size // 64, 2), torch.float8_e8m0fnu)]
        assert module.zerc_moe(
            x, ids, probs, w1, w2, buffer, l1_weights_sf=sf1, l2_weights_sf=sf2
        ) == ("output", "counts")
    assert backend.mega_moe.call_count == 2
    with pytest.raises(ValueError, match="scale shape"):
        module.zerc_moe(
            x,
            ids,
            probs,
            w1,
            w2,
            buffer,
            l1_weights_sf=sf1,
            l2_weights_sf=[_tensor((1, 16, 1, 2), torch.float8_e8m0fnu)],
        )
    assert backend.mega_moe.call_count == 2


@pytest.mark.parametrize("empty", [[], ()])
@pytest.mark.parametrize("shared", [False, True])
def test_empty_bf16_scale_and_shared_lists_match_none(api, empty, shared):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 0)
    kwargs = dict(
        l1_weights_sf=empty,
        l2_weights_sf=empty,
        l1_weights_shared_sf=empty,
        l2_weights_shared_sf=empty,
    )
    if shared:
        kwargs.update(
            l1_weights_shared=[_tensor((1, 32, 16), torch.bfloat16)],
            l2_weights_shared=[_tensor((1, 16, 16), torch.bfloat16)],
        )
    else:
        kwargs.update(l1_weights_shared=empty, l2_weights_shared=empty)
    assert module.zerc_moe(*_inputs(), buffer, **kwargs) == ("output", "counts")
    forwarded = backend.mega_moe.call_args.kwargs
    assert forwarded["l1_weights_sf"] is None
    assert forwarded["l2_weights_sf"] is None
    assert forwarded["l1_weights_shared_sf"] is None
    assert forwarded["l2_weights_shared_sf"] is None
    if not shared:
        assert forwarded["l1_weights_shared"] is None
        assert forwarded["l2_weights_shared"] is None


def test_empty_mxfp8_scales_and_orphan_shared_weights_are_rejected(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(
        object(), 2, 8, 1, 16, 0, dispatch_quant_mode=4, dispatch_quant_out_dtype=24
    )
    x, ids, probs, _, _ = _inputs()
    w1 = [_tensor((1, 128, 16), torch.float8_e4m3fn)]
    w2 = [_tensor((1, 16, 64), torch.float8_e4m3fn)]
    with pytest.raises(ValueError, match="requires one scale"):
        module.zerc_moe(
            x, ids, probs, w1, w2, buffer, l1_weights_sf=[], l2_weights_sf=[]
        )
    bf16 = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 0)
    with pytest.raises(ValueError, match="supplied together"):
        module.zerc_moe(
            *_inputs(),
            bf16,
            l1_weights_shared=[],
            l2_weights_shared=[_tensor((1, 16, 16), torch.bfloat16)],
        )
    backend.mega_moe.assert_not_called()


def test_buffer_allows_smaller_hidden_and_topk_within_capacity(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 2, 32, 0)
    x, ids, probs, w1, w2 = _inputs()
    assert module.zerc_moe(x, ids, probs, w1, w2, buffer) == ("output", "counts")
    bigger = (
        _tensor((2, 32), torch.bfloat16),
        _tensor((2, 2), torch.int32),
        _tensor((2, 2), torch.bfloat16),
        [_tensor((1, 8, 32), torch.bfloat16)],
        [_tensor((1, 32, 4), torch.bfloat16)],
    )
    assert module.zerc_moe(*bigger, buffer) == ("output", "counts")
    assert backend.mega_moe.call_count == 2


def test_buffer_capacity_and_actual_route_constraints_remain_checked(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 2, 32, 0)
    x, ids, probs, w1, w2 = _inputs()
    for hidden in (0, 15):
        with pytest.raises(ValueError, match="hidden dimension"):
            module.zerc_moe(
                _tensor((2, hidden), torch.bfloat16), ids, probs, w1, w2, buffer
            )
    for count in (0, 3):
        with pytest.raises(ValueError, match="routing tensors"):
            module.zerc_moe(
                x,
                _tensor((2, count), torch.int32),
                _tensor((2, count), torch.bfloat16),
                w1,
                w2,
                buffer,
            )
    capped = module.create_zerc_moe_buffer(
        object(), 2, 8, 2, 32, 0, max_recv_token_num=5
    )
    with pytest.raises(ValueError, match="routed token count"):
        module.zerc_moe(x, ids, probs, w1, w2, capped)
    backend.mega_moe.assert_not_called()


def test_larger_actual_dimensions_are_forwarded_for_host_capacity_check(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 4, 8, 1, 16, 0)
    actual = (
        _tensor((2, 32), torch.bfloat16),
        _tensor((2, 2), torch.int32),
        _tensor((2, 2), torch.bfloat16),
        [_tensor((2, 8, 32), torch.bfloat16)],
        [_tensor((2, 32, 4), torch.bfloat16)],
    )
    assert module.zerc_moe(*actual, buffer) == ("output", "counts")
    backend.mega_moe.assert_called_once()
    # The mock checks forwarding only. The actual host enforces the allocated
    # CCL buffer capacity, which is not exercised by this CPU test.
    root = Path(__file__).resolve().parents[3]
    tiling = (
        root / "csrc/zerc_moe/source/mc2/mega_moe/op_host/op_tiling/mega_moe_tiling.cpp"
    ).read_text()
    assert "OP_TILING_CHECK(cclBufferSize < leastCclBufferSize" in tiling


def test_mixed_none_and_empty_optional_values_are_absent(api):
    module, backend = api
    buffer = module.create_zerc_moe_buffer(object(), 2, 8, 1, 16, 0)
    assert module.zerc_moe(
        *_inputs(),
        buffer,
        l1_weights_sf=[],
        l2_weights_sf=None,
        l1_weights_shared=None,
        l2_weights_shared=[],
        l1_weights_shared_sf=[],
        l2_weights_shared_sf=None,
    ) == ("output", "counts")
    for name in (
        "l1_weights_sf",
        "l2_weights_sf",
        "l1_weights_shared",
        "l2_weights_shared",
        "l1_weights_shared_sf",
        "l2_weights_shared_sf",
    ):
        assert backend.mega_moe.call_args.kwargs[name] is None

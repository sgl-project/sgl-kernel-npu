# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2

"""EQBSA metadata guards; run with pytest, without MindIE-SD or an NPU."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def entry(monkeypatch):
    # Substitute only the native boundary; execute the real Python entry.
    name = "_eqbsa_test"
    loader = ModuleType(f"{name}._mindiesd")
    raw = Mock(return_value=(object(), object()))
    loader.load_eqbsa = lambda: raw
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    source = (
        Path(__file__).resolve().parents[3]
        / "python/attentions/attentions/eqbsa/__init__.py"
    )
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.eagle_quant_block_sparse_attention, raw


@pytest.fixture
def inputs():
    # Tail blocks, unequal sequence lengths and GQA in one small contract case.
    def tensor(shape, dtype=torch.int8):
        return torch.empty(shape, dtype=dtype, device="meta")

    return dict(
        query=tensor((2, 4, 257, 128)),
        key=tensor((2, 2, 385, 128)),
        value=tensor((2, 2, 385, 128)),
        num_key_value_heads=2,
        query_scale=tensor((2, 4, 5), torch.float32),
        key_scale=tensor((2, 2, 7), torch.float32),
        value_scale=tensor((2, 2, 128), torch.float32),
        block_sparse_mask=tensor((2, 4, 3, 4)),
    )


@pytest.mark.parametrize(
    "blocks,mask_shape,dtype",
    [
        (None, (2, 4, 3, 4), torch.int8),
        ([128, 256], (2, 4, 3, 2), torch.bool),
        ([256, 128], (2, 4, 2, 4), torch.int8),
    ],
)
def test_valid_inputs_are_forwarded_unchanged(entry, inputs, blocks, mask_shape, dtype):
    function, raw = entry
    inputs.update(
        block_shape=blocks,
        block_sparse_mask=torch.empty(mask_shape, dtype=dtype, device="meta"),
    )
    assert function(**inputs) is raw.return_value
    raw.assert_called_once()
    args, kwargs = raw.call_args
    assert all(
        actual is inputs[name] for actual, name in zip(args, ("query", "key", "value"))
    )
    assert kwargs["block_shape"] == ([128, 128] if blocks is None else blocks)
    for name in ("query_scale", "key_scale", "value_scale", "block_sparse_mask"):
        assert kwargs[name] is inputs[name]


@pytest.mark.parametrize("name", ["query_scale", "key_scale", "value_scale"])
@pytest.mark.parametrize(
    "invalid", ["missing", "shape", torch.float16, torch.bfloat16, torch.int32]
)
def test_invalid_scales_stop_before_dispatch(entry, inputs, name, invalid):
    function, raw = entry
    scale = inputs[name]
    if invalid == "missing":
        inputs[name] = None
        message = "requires query_scale, key_scale and value_scale"
    elif invalid == "shape":
        inputs[name] = scale.transpose(1, 2)
        message = f"{name} must have shape"
    else:
        inputs[name] = scale.to(invalid)
        message = f"{name} must have dtype"
    with pytest.raises(ValueError, match=message):
        function(**inputs)
    raw.assert_not_called()


@pytest.mark.parametrize(
    "shape", [(2, 4, 2, 4), (2, 4, 4, 4), (2, 4, 3, 3), (2, 4, 3, 5)]
)
def test_wrong_mask_block_counts_stop_before_dispatch(entry, inputs, shape):
    function, raw = entry
    inputs["block_sparse_mask"] = torch.empty(shape, dtype=torch.int8, device="meta")
    with pytest.raises(ValueError, match="block_sparse_mask must have block counts"):
        function(**inputs)
    raw.assert_not_called()


@pytest.mark.parametrize("invalid", ["mask_rank", "block_shape"])
def test_existing_native_validation_is_not_replaced(entry, inputs, invalid):
    function, raw = entry
    if invalid == "mask_rank":
        inputs["block_sparse_mask"] = inputs["block_sparse_mask"].flatten(2)
    else:
        inputs["block_shape"] = [128, 0]
    error = RuntimeError("official parameter error")
    raw.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        function(**inputs)
    assert caught.value is error
    raw.assert_called_once()

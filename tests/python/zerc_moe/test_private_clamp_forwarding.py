"""Exercise the real private Python call chain without loading a device extension.

These tests check argument forwarding only; they do not validate the ACLNN ABI
or run the Ascend kernel.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import Mock, sentinel

import pytest
import torch


@pytest.fixture
def private_chain():
    root = Path(__file__).resolve().parents[3]
    source = (
        root
        / "csrc/zerc_moe/source/torch_extension/npu_ops_transformer/ops/mega_moe.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"mega_moe", "_npu_mega_moe"}
    ]
    assert {node.name for node in selected} == {"mega_moe", "_npu_mega_moe"}
    # The PrivateUse1 decorator normally loads/registers the NPU extension.
    # Remove only decorators; execute the actual two function bodies unchanged.
    for function in selected:
        function.decorator_list = []
    binding = Mock(return_value=(sentinel.output, sentinel.counts))
    scope = {
        "torch": torch,
        "List": List,
        "Optional": Optional,
        "SymmBuffer": object,
        "_op_module": SimpleNamespace(npu_zerc_moe=binding),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope
    )
    return scope["mega_moe"], binding


def _buffer():
    return SimpleNamespace(
        context=sentinel.context,
        num_experts=16,
        ep_world_size=4,
        ccl_buffer_size=8192,
        max_recv_token_num=23,
        dispatch_quant_mode=4,
        combine_quant_mode=0,
        comm_alg="",
        global_bs=128,
        dispatch_quant_out_dtype=24,
    )


@pytest.mark.parametrize("clamp", [0.0, 10.0])
def test_private_eager_chain_preserves_all_arguments(private_chain, clamp):
    run, binding = private_chain
    buffer = _buffer()
    l1, l2 = [sentinel.l1], [sentinel.l2]
    sf1, sf2 = [sentinel.sf1], [sentinel.sf2]
    shared1, shared2 = [sentinel.shared1], [sentinel.shared2]
    shared_sf1, shared_sf2 = [sentinel.shared_sf1], [sentinel.shared_sf2]

    # Values like the mask/scales sentinels exercise slot order in this mock;
    # the public wrapper still rejects non-default reserved arguments.
    result = run(
        sentinel.x,
        sentinel.ids,
        sentinel.weights,
        l1,
        l2,
        buffer,
        scales=sentinel.scales,
        l1_weights_sf=sf1,
        l2_weights_sf=sf2,
        x_active_mask=sentinel.mask,
        weight1_type=31,
        weight2_type=32,
        timer=sentinel.timer,
        l1_weights_shared=shared1,
        l2_weights_shared=shared2,
        l1_weights_shared_sf=shared_sf1,
        l2_weights_shared_sf=shared_sf2,
        activation_clamp=clamp,
    )
    assert result == (sentinel.output, sentinel.counts)
    binding.assert_called_once_with(
        sentinel.context,
        sentinel.x,
        sentinel.ids,
        sentinel.weights,
        l1,
        l2,
        16,
        4,
        8192,
        sf1,
        sf2,
        sentinel.mask,
        sentinel.scales,
        23,
        4,
        0,
        "",
        128,
        24,
        31,
        32,
        sentinel.timer,
        shared1,
        shared2,
        shared_sf1,
        shared_sf2,
        clamp,
    )


def test_private_eager_chain_defaults_to_unclamped(private_chain):
    run, binding = private_chain
    assert run(sentinel.x, sentinel.ids, sentinel.weights, [], [], _buffer()) == (
        sentinel.output,
        sentinel.counts,
    )
    args = binding.call_args.args
    assert args[-5:] == (None, None, None, None, 0.0)

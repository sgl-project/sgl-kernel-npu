"""CPU checks for the independent BF16 oracle and worker argument validation."""

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKER = Path(__file__).resolve().parents[1] / "operators/moe/mega_moe_worker.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("_mega_moe_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_matches_scalar_swiglu(worker, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        worker,
        "make_weights",
        lambda torch, expert, hidden, intermediate: (
            torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            torch.tensor([[expert + 1.0]], dtype=torch.bfloat16),
        ),
    )
    x = torch.tensor([[1.0], [-1.0], [0.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    routing = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
    expected = torch.tensor(
        [
            [2 / (1 + math.exp(-1)) * 1.75],
            [2 / (1 + math.exp(1)) * 1.5],
            [0.0],
        ]
    )
    actual = worker.reference_moe(torch, x, ids, routing, 1, 1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("routing", ["balanced", "skewed"])
@pytest.mark.parametrize("input_mode", ["normal", "zero"])
def test_input_matrix_is_deterministic_and_routes_valid_experts(
    worker, routing, input_mode
):
    torch = pytest.importorskip("torch")
    args = SimpleNamespace(
        hidden=8, topk=4, num_experts=8, routing=routing, input_mode=input_mode
    )
    for rank in range(4):
        x, ids, weights = worker.generate_data(torch, args, rank, 17)
        again = worker.generate_data(torch, args, rank, 17)
        for actual, expected in zip((x, ids, weights), again):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert ids.min() >= 0 and ids.max() < args.num_experts
        assert all(row.unique().numel() == args.topk for row in ids)
        torch.testing.assert_close(weights.sum(-1), torch.ones(17))
        if input_mode == "zero":
            assert torch.count_nonzero(x) == 0
        if routing == "skewed":
            assert ids.unique().numel() < args.num_experts


@pytest.mark.parametrize(
    "options,message",
    [
        (["--tokens", "33", "--capacity", "32"], "0 < tokens <= capacity"),
        (["--num-experts", "7"], "divisible by 4"),
        (["--topk", "9"], "invalid topk"),
        (["--perf-iters", "-1"], "invalid topk or perf-iters"),
        (["--hidden", "513"], "positive multiples of 512"),
    ],
)
def test_worker_rejects_invalid_config_before_runtime_import(options, message):
    result = subprocess.run(
        [sys.executable, str(WORKER), *options],
        env={**os.environ, "WORLD_SIZE": "4"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_eight_rank_routing_reaches_all_experts(worker):
    torch = pytest.importorskip("torch")
    args = SimpleNamespace(
        hidden=8,
        topk=8,
        num_experts=16,
        world_size=8,
        routing="all-ranks",
        input_mode="normal",
    )
    for rank in range(8):
        _, ids, _ = worker.generate_data(torch, args, rank, 17)
        # Two experts belong to each destination rank. Every row visits all
        # eight destinations, and each source rank exercises all 16 experts.
        torch.testing.assert_close(ids // 2, torch.arange(8).expand(17, 8))
        torch.testing.assert_close(ids.unique(), torch.arange(16))


@pytest.fixture
def mxfp4_reference():
    path = WORKER.with_name("mega_moe_mxfp4_accuracy.py")
    spec = importlib.util.spec_from_file_location("_mxfp4_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fp4_decode_values_and_block_scales(mxfp4_reference):
    torch = pytest.importorskip("torch")
    # Low nibble precedes high nibble. Cover all signed E2M1 encodings.
    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    ).repeat(1, 4)
    exponents = torch.tensor([[[127, 128]]], dtype=torch.uint8)
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    ).repeat(2)
    expected = torch.cat((values, values * 2)).reshape(1, 64)
    torch.testing.assert_close(
        mxfp4_reference.decode_weight(torch, packed, exponents), expected
    )


def test_mxfp8_rounding_and_saturation(mxfp4_reference):
    torch = pytest.importorskip("torch")
    values = torch.zeros(1, 64)
    values[0, :4] = torch.tensor([1.0625, 1.1875, 1.875, -1.875])
    values[0, 32:36] = values[0, :4] * 2
    expected = torch.zeros_like(values)
    expected[0, :4] = torch.tensor([1.0, 1.25, 1.75, -1.75])
    expected[0, 32:36] = expected[0, :4] * 2
    torch.testing.assert_close(
        mxfp4_reference.round_mxfp8(torch, values), expected, rtol=0, atol=0
    )


def test_mxfp4_reference_depends_on_expert_and_routing(mxfp4_reference):
    torch = pytest.importorskip("torch")
    torch.manual_seed(37)
    x = torch.randn(2, 64).to(torch.bfloat16)
    ids = torch.tensor([[0], [1]])
    routing = torch.ones(2, 1)
    result = mxfp4_reference.reference_moe(torch, x, ids, routing, 64, 64)
    doubled = mxfp4_reference.reference_moe(torch, x, ids, routing * 2, 64, 64)
    swapped = mxfp4_reference.reference_moe(torch, x, ids.flip(0), routing, 64, 64)
    assert torch.isfinite(result).all() and result.norm() > 0
    torch.testing.assert_close(doubled, result * 2)
    assert not torch.allclose(result, swapped)

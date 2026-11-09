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
        (["--num-experts", "7"], "divisible by four"),
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

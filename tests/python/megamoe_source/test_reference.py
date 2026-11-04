"""Exercise the CPU BF16 oracle independently of the device kernel."""

import math

import pytest
import torch

from tests.python.megamoe_source.reference import reference_bf16


def _case():
    # Four experts on two ranks; only the first hidden dimension is active.
    xs = [torch.zeros(1, 16, dtype=torch.bfloat16) for _ in range(2)]
    for x in xs:
        x[0, 0] = 1
    first = [torch.zeros(2, 2, 16, dtype=torch.bfloat16) for _ in range(2)]
    second = [torch.zeros(2, 16, 1, dtype=torch.bfloat16) for _ in range(2)]
    for owner in range(2):
        for expert in range(2):
            first[owner][expert, :, 0] = 1
            second[owner][expert, 0, 0] = owner * 2 + expert + 1
    ids = [
        torch.tensor([[3, 0]], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.int32),
    ]
    probabilities = [
        torch.tensor([[0.5, 1]], dtype=torch.bfloat16),
        torch.tensor([[1, 0.5]], dtype=torch.bfloat16),
    ]
    return xs, ids, probabilities, first, second


def _contribution(multiplier, probability):
    # Closed-form FFN: silu(1) * 1, followed by the selected multiplier.
    hidden = torch.tensor(1.0 / (1.0 + math.exp(-1.0)), dtype=torch.bfloat16)
    value = (hidden.float() * multiplier).to(torch.bfloat16)
    return (value * probability).to(torch.bfloat16)


def test_two_rank_routing_and_counts():
    outputs, counts = reference_bf16(*_case())
    expected0 = (_contribution(1, 1) + _contribution(4, 0.5)).to(torch.bfloat16)
    expected1 = (_contribution(2, 1) + _contribution(3, 0.5)).to(torch.bfloat16)
    assert outputs[0][0, 0] == expected0
    assert outputs[1][0, 0] == expected1
    assert torch.count_nonzero(outputs[0][0, 1:]) == 0
    assert torch.count_nonzero(outputs[1][0, 1:]) == 0
    assert torch.equal(counts[0], torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(counts[1], torch.tensor([1, 1], dtype=torch.int32))


def test_shared_expert_is_local_and_not_route_weighted():
    xs, ids, probabilities, first, second = _case()
    shared_first = [first[rank][:1].clone() for rank in range(2)]
    shared_second = [second[rank][:1].clone() for rank in range(2)]
    routed, counts = reference_bf16(xs, ids, probabilities, first, second)
    outputs, shared_counts = reference_bf16(
        xs, ids, probabilities, first, second, shared_first, shared_second
    )
    for rank in range(2):
        expected = (routed[rank][0, 0] + _contribution(rank * 2 + 1, 1)).to(
            torch.bfloat16
        )
        assert outputs[rank][0, 0] == expected
        assert torch.equal(counts[rank], shared_counts[rank])


@pytest.mark.parametrize("invalid", [[0, 0], [-1, 0], [0, 4]])
def test_invalid_routes_fail_before_computation(invalid):
    xs, ids, probabilities, first, second = _case()
    ids[0] = torch.tensor([invalid], dtype=torch.int32)
    with pytest.raises(ValueError):
        reference_bf16(xs, ids, probabilities, first, second)


def test_shared_expert_weights_are_paired():
    xs, ids, probabilities, first, second = _case()
    with pytest.raises(ValueError, match="supplied together"):
        reference_bf16(xs, ids, probabilities, first, second, first)

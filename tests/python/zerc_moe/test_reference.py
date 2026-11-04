"""Exercise the CPU BF16 oracle independently of the device kernel."""

import math

import pytest
import torch

from tests.python.zerc_moe.reference import reference_bf16


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


@pytest.mark.parametrize(
    "invalid", ["empty_tokens", "empty_topk", "empty_intermediate"]
)
def test_empty_dimensions_are_rejected(invalid):
    xs, ids, probabilities, first, second = _case()
    if invalid == "empty_tokens":
        xs[0], ids[0], probabilities[0] = xs[0][:0], ids[0][:0], probabilities[0][:0]
    elif invalid == "empty_topk":
        ids[0], probabilities[0] = ids[0][:, :0], probabilities[0][:, :0]
    else:
        first[0], second[0] = first[0][:, :0, :], second[0][:, :, :0]
    with pytest.raises(ValueError):
        reference_bf16(xs, ids, probabilities, first, second)


def _clamp_case(gate, up):
    xs = [torch.zeros((1, 16), dtype=torch.bfloat16) for _ in range(2)]
    for x in xs:
        x[0, 0] = 1
    ids = [torch.tensor([[0]], dtype=torch.int32) for _ in range(2)]
    probabilities = [torch.ones((1, 1), dtype=torch.bfloat16) for _ in range(2)]
    first = [torch.zeros((1, 2, 16), dtype=torch.bfloat16) for _ in range(2)]
    second = [torch.zeros((1, 16, 1), dtype=torch.bfloat16) for _ in range(2)]
    for rank in range(2):
        first[rank][0, 0, 0] = gate
        first[rank][0, 1, 0] = up
        second[rank][0, 0, 0] = 1
    return xs, ids, probabilities, first, second


def _expected_clamped_scalar(gate, up, limit):
    # Independent scalar calculation; inputs in these tests are exactly BF16.
    if limit:
        gate = min(gate, limit)
        up = min(limit, max(-limit, up))
    return torch.tensor(gate / (1 + math.exp(-gate)) * up, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "gate,up,limit",
    [
        (12, 16, 10),
        (12, -16, 10),
        (-12, -16, 10),
        (10, 10, 10),
        (9, -10, 10),
        (12, 16, 0),
        (12, 16, None),
    ],
)
def test_clamped_swiglu_matches_independent_scalar(gate, up, limit):
    outputs, counts = reference_bf16(*_clamp_case(gate, up), activation_clamp=limit)
    expected = _expected_clamped_scalar(gate, up, limit)
    for out in outputs:
        assert out[0, 0] == expected
        assert torch.count_nonzero(out[0, 1:]) == 0
    assert torch.equal(counts[0], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(counts[1], torch.tensor([0], dtype=torch.int32))


def test_clamp_applies_to_routed_and_shared_experts_before_combining():
    xs, ids, probabilities, first, second = _clamp_case(20, 20)
    probabilities[0][0, 0] = 0.5
    shared_first = [w.clone() for w in first]
    for weight in shared_first:
        weight[0, 1, 0] = -20
    shared_second = [w.clone() for w in second]
    output, counts = reference_bf16(
        xs,
        ids,
        probabilities,
        first,
        second,
        shared_first,
        shared_second,
        activation_clamp=10,
    )
    base = _expected_clamped_scalar(20, 20, 10)
    shared = _expected_clamped_scalar(20, -20, 10)
    expected0 = ((base * 0.5).to(torch.bfloat16) + shared).to(torch.bfloat16)
    expected1 = (base + shared).to(torch.bfloat16)
    assert output[0][0, 0] == expected0
    assert output[1][0, 0] == expected1
    assert torch.equal(counts[0], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(counts[1], torch.tensor([0], dtype=torch.int32))


def test_clamp_default_preserves_previous_reference_behavior():
    inputs = _case()
    baseline, counts = reference_bf16(*inputs)
    for disabled in (None, 0):
        actual, actual_counts = reference_bf16(*inputs, activation_clamp=disabled)
        assert all(torch.equal(x, y) for x, y in zip(baseline, actual))
        assert all(torch.equal(x, y) for x, y in zip(counts, actual_counts))

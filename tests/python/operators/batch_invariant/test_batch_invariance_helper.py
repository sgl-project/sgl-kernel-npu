"""Sensitivity checks for the batch-invariance comparison helpers.

These cases prove the comparison fails when it should, so an operator case that
passes cannot be passing because the harness is blind to small differences.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.operators.batch_invariant.batch_invariance import (
    assert_batch_invariant,
    assert_unchanged,
    snapshot,
)


def test_identical_outputs_pass():
    """Equal tensors and equal nested structures are accepted."""
    tensor = torch.linspace(-1, 1, 12, dtype=torch.float32).reshape(3, 4)
    assert_batch_invariant(tensor, tensor.clone())
    assert_batch_invariant((tensor, None, 3), (tensor.clone(), None, 3))


def test_single_ulp_difference_fails():
    """One representable step is enough to fail; no tolerance is applied."""
    tensor = torch.ones(4, dtype=torch.float32)
    other = tensor.clone()
    other[0] += torch.finfo(torch.float32).eps
    with pytest.raises(AssertionError):
        assert_batch_invariant(tensor, other)


def test_batch_dependent_output_fails():
    """A result that depends on the batch size must be rejected."""
    torch.npu.manual_seed(0)
    inputs = torch.randn(8, 4, dtype=torch.float32)
    weight = torch.randn(4, 4, dtype=torch.float32)

    def batch_dependent(rows):
        # Scale by the batch size so the same row differs across batches.
        return (rows @ weight) / rows.shape[0]

    full = batch_dependent(inputs)
    single = batch_dependent(inputs[:1])
    assert not torch.equal(full[:1], single)
    with pytest.raises(AssertionError):
        assert_batch_invariant(full[:1], single)


def test_structural_mismatch_fails():
    """Different output shapes or container types are rejected."""
    tensor = torch.ones(2, dtype=torch.float32)
    with pytest.raises(AssertionError):
        assert_batch_invariant(tensor, tensor.reshape(1, 2))
    with pytest.raises(AssertionError):
        assert_batch_invariant((tensor,), (tensor, tensor))
    with pytest.raises(AssertionError):
        assert_batch_invariant([tensor], (tensor,))


def test_snapshot_detects_mutation():
    """Input mutation is caught against the snapshot taken before the call."""
    tensor = torch.ones(2, dtype=torch.float32)
    before = snapshot((tensor,))
    assert_unchanged((tensor,), before)
    tensor.add_(1)
    with pytest.raises(AssertionError):
        assert_unchanged((tensor,), before)

"""Bitwise comparison helpers for the batch-invariant operator tests."""

import torch


def assert_batch_invariant(result, result_ref, path="output"):
    """Require bitwise-identical results from different batch sizes.

    No tolerance is applied, and nested tuples/lists are compared element-wise
    so multi-output operators keep their structure and order.
    """
    if isinstance(result, torch.Tensor) or isinstance(result_ref, torch.Tensor):
        assert isinstance(result, torch.Tensor) and isinstance(
            result_ref, torch.Tensor
        ), f"{path}: Tensor/non-Tensor mismatch"
        assert result.shape == result_ref.shape, (
            f"{path}: shape mismatch ({tuple(result.shape)} vs "
            f"{tuple(result_ref.shape)})"
        )
        assert result.dtype == result_ref.dtype, f"{path}: dtype mismatch"
        assert result.device == result_ref.device, f"{path}: device mismatch"
        torch.testing.assert_close(
            result.detach(),
            result_ref.detach(),
            atol=0,
            rtol=0,
            equal_nan=False,
            msg=lambda message: f"{path}: {message}",
        )
        return

    assert type(result) is type(result_ref), (
        f"{path}: type mismatch "
        f"({type(result).__name__} vs {type(result_ref).__name__})"
    )
    if isinstance(result_ref, (tuple, list)):
        assert len(result) == len(result_ref), f"{path}: length mismatch"
        for index, (value, other) in enumerate(zip(result, result_ref)):
            assert_batch_invariant(value, other, path=f"{path}[{index}]")
        return
    assert (
        result == result_ref
    ), f"{path}: value mismatch ({result!r} vs {result_ref!r})"


def snapshot(tensors):
    """Clone inputs before a call that must not modify them."""
    return tuple(tensor.clone() for tensor in tensors)


def assert_unchanged(inputs, snapshots, path="input"):
    """Require an operator to leave every input tensor bitwise unchanged."""
    assert len(inputs) == len(
        snapshots
    ), f"{path}: snapshot count mismatch ({len(inputs)} vs {len(snapshots)})"
    for index, (tensor, before) in enumerate(zip(inputs, snapshots)):
        assert_batch_invariant(tensor, before, path=f"{path}[{index}]")


def require_batch_invariant_ops():
    """Skip the calling test when the external library is not installed."""
    import pytest

    pytest.importorskip(
        "batch_invariant_ops",
        reason="the external batch_invariant_ops extension is not installed",
    )

"""Metadata checks for attention-residual scoring and mixing."""

import math

import torch


def validate_mix_inputs(
    prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
):
    """Validate the common contract without inspecting device tensor values."""
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    for name, tensor, ndim in (
        ("prefix_sum", prefix_sum, 2),
        ("bank", bank, 3),
        ("combined_weight", combined_weight, 1),
    ):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor"
        assert tensor.layout == torch.strided, f"{name} must have strided layout"
        assert (
            tensor.ndim == ndim
        ), f"{name} must have {ndim} dimensions, got shape {tuple(tensor.shape)}"
        assert (
            tensor.dtype in supported_dtypes
        ), f"{name} must have float16, bfloat16 or float32 dtype, got {tensor.dtype}"

    num_tokens, hidden_size = prefix_sum.shape
    assert hidden_size > 0, f"hidden size must be positive, got {hidden_size}"
    assert bank.shape[0] == num_tokens and bank.shape[2] == hidden_size, (
        "bank must match prefix_sum's token count and hidden size, "
        f"got prefix_sum={tuple(prefix_sum.shape)}, bank={tuple(bank.shape)}"
    )
    assert combined_weight.shape[0] == hidden_size, (
        "combined_weight must have one entry per hidden feature, "
        f"got shape {tuple(combined_weight.shape)}, hidden size={hidden_size}"
    )
    assert prefix_sum.device == bank.device == combined_weight.device, (
        "prefix_sum, bank and combined_weight must share a device, "
        f"got {prefix_sum.device}, {bank.device} and {combined_weight.device}"
    )
    assert isinstance(num_valid_blocks, int) and not isinstance(
        num_valid_blocks, bool
    ), f"num_valid_blocks must be a Python integer, got {num_valid_blocks!r}"
    if not 0 <= num_valid_blocks <= bank.shape[1]:
        raise ValueError("num_valid_blocks must fit within the residual bank")

    assert isinstance(variance_epsilon, (int, float)) and not isinstance(
        variance_epsilon, bool
    ), "variance_epsilon must be a Python int or float"
    assert (
        math.isfinite(variance_epsilon) and variance_epsilon > 0
    ), f"variance_epsilon must be finite and positive, got {variance_epsilon!r}"

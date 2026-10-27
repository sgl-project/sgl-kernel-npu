"""Metadata validation shared by rotary embedding implementations."""

import torch


def validate_rope_inputs(query, key, cos_sin, rotary_dim, is_neox_style):
    """Check the common contract without reading or synchronizing device data."""
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    for name, tensor, ndim in (
        ("query", query, 3),
        ("key", key, 3),
        ("cos_sin", cos_sin, 2),
    ):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor"
        assert tensor.layout == torch.strided, f"{name} must have strided layout"
        assert (
            tensor.ndim == ndim
        ), f"{name} must have {ndim} dimensions, got shape {tuple(tensor.shape)}"
        assert (
            tensor.dtype in supported_dtypes
        ), f"{name} must have float16, bfloat16 or float32 dtype, got {tensor.dtype}"

    assert all(
        size > 0 for size in query.shape
    ), f"query dimensions must be positive, got {tuple(query.shape)}"
    assert all(
        size > 0 for size in key.shape
    ), f"key dimensions must be positive, got {tuple(key.shape)}"
    assert query.shape[0] == key.shape[0] and query.shape[2] == key.shape[2], (
        "query and key must share token count and head dimension, "
        f"got {tuple(query.shape)} and {tuple(key.shape)}"
    )
    assert (
        query.dtype == key.dtype
    ), f"query and key dtype must match, got {query.dtype} and {key.dtype}"
    assert query.device == key.device == cos_sin.device, (
        "query, key and cos_sin must share a device, "
        f"got {query.device}, {key.device} and {cos_sin.device}"
    )
    assert isinstance(rotary_dim, int) and not isinstance(
        rotary_dim, bool
    ), f"rotary_dim must be an integer, got {rotary_dim!r}"
    assert 0 < rotary_dim <= query.shape[2] and rotary_dim % 2 == 0, (
        "rotary_dim must be positive, even and no greater than the head dimension, "
        f"got rotary_dim={rotary_dim}, head dimension={query.shape[2]}"
    )
    assert isinstance(
        is_neox_style, bool
    ), f"is_neox_style must be bool, got {is_neox_style!r}"
    assert cos_sin.shape[0] >= query.shape[0] and cos_sin.shape[1] == rotary_dim, (
        "cos_sin must have shape [at least T, rotary_dim] with preselected rows, "
        f"got {tuple(cos_sin.shape)}, T={query.shape[0]}, rotary_dim={rotary_dim}"
    )

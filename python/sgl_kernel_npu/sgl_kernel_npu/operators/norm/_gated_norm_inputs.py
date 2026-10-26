"""Metadata checks and output/statistics views for grouped gated normalization."""

import torch


def validate_inputs(x, weight, bias, z, out, group_size):
    if x.ndim != 2 or x.shape[1] <= 0:
        raise ValueError("x must have shape [rows, positive hidden]")
    rows, hidden = x.shape
    size = hidden if group_size is None else group_size
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or hidden % size
    ):
        raise ValueError("group_size must be a positive divisor of the hidden width")
    if x.stride(-1) != 1:
        raise ValueError("x must have unit last-dimension stride")
    for name, value in (("weight", weight), ("bias", bias)):
        if value is None and name == "bias":
            continue
        if (
            value is None
            or value.shape != (hidden,)
            or value.stride(-1) != 1
            or value.device != x.device
        ):
            raise ValueError(f"{name} must be a contiguous hidden vector on x's device")
    for name, value in (("z", z), ("out", out)):
        if value is not None and (
            value.shape != (rows, hidden)
            or value.stride(-1) != 1
            or value.device != x.device
        ):
            raise ValueError(
                f"{name} must match x's shape/device and have unit last-dimension stride"
            )
    maximum = 65536 // x.element_size()
    if size > min(maximum, 1 << (size - 1).bit_length()):
        raise RuntimeError("Feature dim too large.")
    return size


def prepare_reference(x, weight, bias, z, out, group_size, is_rms_norm):
    size = validate_inputs(x, weight, bias, z, out, group_size)
    rows, hidden = x.shape
    groups = hidden // size
    output = torch.empty_like(x) if out is None else out
    mean = (
        None
        if is_rms_norm
        else torch.empty((groups * rows,), device=x.device, dtype=torch.float32)
    )
    rstd = torch.empty((groups * rows,), device=x.device, dtype=torch.float32)
    return (
        x.reshape(rows, groups, size),
        weight.reshape(1, groups, size),
        None if bias is None else bias.reshape(1, groups, size),
        None if z is None else z.reshape(rows, groups, size),
        output,
        output.view(rows, groups, size),
        mean,
        None if mean is None else mean.view(groups, rows).transpose(0, 1),
        rstd,
        rstd.view(groups, rows).transpose(0, 1),
    )

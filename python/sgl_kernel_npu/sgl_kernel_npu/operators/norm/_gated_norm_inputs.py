"""Original grouped-norm constraints and views for the PyTorch reference."""

import torch


def prepare_reference(x, weight, bias, z, out, group_size, is_rms_norm):
    rows, hidden = x.shape
    size = hidden if group_size is None else group_size
    assert hidden % size == 0
    groups = hidden // size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (rows, hidden)
    assert weight.shape == (hidden,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (hidden,)
    if out is not None:
        assert out.shape == x.shape
    output = torch.empty_like(x) if out is None else out
    assert output.stride(-1) == 1
    mean = (
        None
        if is_rms_norm
        else torch.empty((groups * rows,), device=x.device, dtype=torch.float32)
    )
    rstd = torch.empty((groups * rows,), device=x.device, dtype=torch.float32)
    maximum = 65536 // x.element_size()
    block = min(maximum, 1 << (size - 1).bit_length())
    if size > block:
        raise RuntimeError("Feature dim too large.")
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

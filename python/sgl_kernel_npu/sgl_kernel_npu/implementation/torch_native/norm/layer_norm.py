"""Pure PyTorch reference for the CANN LayerNorm operator."""

import math

import torch


@torch.no_grad()
def layer_norm(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
    impl_mode=0,
):
    """Normalize the trailing axes of ``input`` and return (out, mean, rstd).

    The reference works in FP32, matching the sglang native path, then casts the
    result back to the input dtype. ``impl_mode`` only selects a CANN computation
    mode, so it does not change the reference result. ``mean``/``rstd`` use the
    legacy allocation shape: the non-normalized axes followed by one for each
    normalized axis, or ``[0]`` when there are no rows.
    """
    shape = tuple(int(size) for size in normalized_shape)
    values = input.float()
    weight_fp32 = None if weight is None else weight.float()
    bias_fp32 = None if bias is None else bias.float()
    out, mean, rstd = torch.native_layer_norm(
        values, shape, weight_fp32, bias_fp32, eps
    )

    stat_shape = _stat_shape(input.shape, input.dim() - len(shape), len(shape))
    return (
        out.to(input.dtype),
        mean.reshape(stat_shape).to(input.dtype),
        rstd.reshape(stat_shape).to(input.dtype),
    )


def _stat_shape(input_shape, begin_axis, norm_ndim):
    """Return the legacy mean/rstd shape for the given normalized axes."""
    rows = math.prod(input_shape[:begin_axis])
    if rows <= 0:
        return (rows,)
    return tuple(input_shape[:begin_axis]) + (1,) * norm_ndim

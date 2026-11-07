"""Independent PyTorch grouped Gemma RMSNorm; no custom native operator calls."""

import torch


def grouped_gemma_rmsnorm(x, weight, group_size, eps=1e-6):
    """RMS-normalize ``group_size``-long groups and scale by ``1 + weight``.

    The weight offset of group ``g`` is ``(g * group_size) % weight.numel()``,
    which cycles with a period of ``weight.numel() // group_size`` groups. The
    public operator guarantees the group count is a whole number of periods, so
    reshaping to ``[-1, slices, group_size]`` places every group against its own
    slice and no gather is needed.
    """
    width = x.shape[-1] if group_size is None else group_size
    slices = weight.numel() // width
    groups = x.reshape(-1, slices, width).float()
    inverse = torch.rsqrt(groups.square().sum(-1, keepdim=True) / width + eps)
    scale = 1.0 + weight.float().reshape(slices, width)
    return ((groups * inverse) * scale).to(x.dtype).reshape(x.shape)

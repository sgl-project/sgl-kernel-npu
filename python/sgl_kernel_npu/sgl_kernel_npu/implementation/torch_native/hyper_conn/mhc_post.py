"""Independent post mapping in FP32 with one final output cast."""

import torch


def mhc_post(x, h_res, h_out, h_post):
    return (
        torch.bmm(h_res, x.float()) + h_post.unsqueeze(-1) * h_out.float().unsqueeze(1)
    ).to(x.dtype)

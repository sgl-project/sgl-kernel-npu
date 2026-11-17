"""FP32 PyTorch implementation of hyper-connection post mixing."""

import torch


def hc_post(x, residual, post, comb):
    """Match the selected recipes binding's input contract and output dtype."""
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert x.ndim in (2, 3) and all(size > 0 for size in x.shape)
    assert x.shape[-1] in (4096, 7168)
    leading = x.shape[:-1]
    assert residual.shape == (*leading, 4, x.shape[-1])
    assert post.shape == (*leading, 4)
    assert comb.shape == (*leading, 4, 4)
    assert x.dtype in dtypes and residual.dtype == x.dtype
    assert post.dtype in dtypes and comb.dtype == post.dtype
    assert all(t.device == x.device for t in (residual, post, comb))
    assert all(t.is_contiguous() for t in (x, residual, post, comb))
    mixed = (comb.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
    return (post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)

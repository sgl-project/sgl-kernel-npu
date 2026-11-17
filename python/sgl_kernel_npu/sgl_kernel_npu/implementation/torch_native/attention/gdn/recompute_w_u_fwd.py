"""Independent PyTorch evaluation of the GDN WY representation."""

import torch


def _validate_inputs(k, v, beta, g_cumsum, A, cu_seqlens):
    assert k.ndim == v.ndim == A.ndim == 4, "k, v and A must be 4-D"
    B, T, Hg, K = k.shape
    assert all(n > 0 for n in k.shape + v.shape + A.shape), "Inputs must be nonempty"
    H = v.shape[2]
    assert v.shape[:2] == (B, T), "k and v must share batch and sequence dimensions"
    assert H % Hg == 0, "Value heads must be a multiple of key heads"
    assert A.shape[:3] == (B, T, H), "A must have shape [B, T, H, C]"
    assert beta.shape == g_cumsum.shape == (B, T, H), "Gates must have shape [B, T, H]"
    tensors = (k, v, beta, g_cumsum, A)
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert all(t.dtype in dtypes for t in tensors), "Inputs must be FP16, BF16 or FP32"
    assert all(t.device == k.device for t in tensors), "Inputs must share a device"
    if cu_seqlens is not None:
        assert B == 1, "Packed input requires B=1"
        assert (
            cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2
        ), "cu_seqlens must be 1-D with at least two entries"
        assert cu_seqlens.dtype in (
            torch.int32,
            torch.int64,
        ), "cu_seqlens must be int32 or int64"


def recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None):
    _validate_inputs(k, v, beta, g_cumsum, A, cu_seqlens)
    B, T, Hg, K = k.shape
    H, V, C = v.shape[2], v.shape[3], A.shape[-1]
    boundaries = [0, T] if cu_seqlens is None else cu_seqlens.tolist()
    if cu_seqlens is not None:
        assert (
            boundaries[0] == 0 and boundaries[-1] == T
        ), "Boundaries must start at 0 and end at T"
        assert all(
            bos < eos for bos, eos in zip(boundaries[:-1], boundaries[1:])
        ), "Boundaries must strictly increase"
    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)
    for bos, eos in zip(boundaries[:-1], boundaries[1:]):
        for start in range(bos, eos, C):
            end = min(start + C, eos)
            size = end - start
            matrix = A[:, start:end, :, :size].float().permute(0, 2, 1, 3)
            keys = k[:, start:end].float().repeat_interleave(H // Hg, dim=2)
            scale = beta[:, start:end].float()
            decay = g_cumsum[:, start:end].float().exp()
            values = v[:, start:end].float() * scale[..., None]
            keys = keys * scale[..., None] * decay[..., None]
            u[:, start:end] = (
                (matrix @ values.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).to(v.dtype)
            )
            w[:, start:end] = (
                (matrix @ keys.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).to(k.dtype)
            )
    return w, u

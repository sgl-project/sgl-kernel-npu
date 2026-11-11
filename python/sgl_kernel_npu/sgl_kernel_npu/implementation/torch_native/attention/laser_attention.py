"""Pure PyTorch reference for the LaserAttention operator."""

import torch


def laser_attention(
    query,
    key,
    value,
    *,
    atten_mask=None,
    alibi_mask=None,
    drop_mask=None,
    scale_value=1.0,
    head_num=None,
    input_layout="BNSD",
    keep_prob=1.0,
    pre_tokens=2147483647,
    next_tokens=1,
    is_high_precision=True,
):
    """Compute full attention for BNSD inputs and return (log_max_sum, out).

    The device Kernel always runs the dense (non-causal) path with dropout
    disabled and ignores atten_mask/alibi_mask, so this reference mirrors that
    contract. The first output is never written by the Kernel; a zero tensor of
    the documented shape is returned to keep the signature stable.
    """
    if input_layout != "BNSD":
        raise ValueError("laser_attention torch_native supports BNSD only")
    if query.dim() != 4:
        raise ValueError("query must be 4D [B, N, S, D]")
    batch, heads, seq_len, head_dim = query.shape
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("key/value must be 4D [B, N, S, D]")
    if key.shape[0] != batch or value.shape[0] != batch:
        raise ValueError("query/key/value must share the batch dimension")
    if key.shape[2] != seq_len or key.shape[3] != head_dim:
        raise ValueError("key must be [B, Nkv, S, D]")
    if value.shape[2] != seq_len or value.shape[3] != head_dim:
        raise ValueError("value must be [B, Nkv, S, D]")
    if heads % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of key heads")
    if head_num is not None and head_num != heads:
        raise ValueError("head_num must match query.size(1)")
    if atten_mask is not None or alibi_mask is not None:
        raise NotImplementedError("torch_native does not apply atten_mask/alibi_mask")
    if drop_mask is not None or keep_prob != 1.0:
        raise NotImplementedError("torch_native does not apply dropout")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query/key/value must be on the same device")

    group = heads // key.shape[1]
    q = query.float()
    k = key.float().repeat_interleave(group, dim=1)
    v = value.float().repeat_interleave(group, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale_value
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)

    softmax_log_max_sum = torch.zeros(
        (batch, heads, seq_len), dtype=torch.float32, device=query.device
    )
    return softmax_log_max_sum, out

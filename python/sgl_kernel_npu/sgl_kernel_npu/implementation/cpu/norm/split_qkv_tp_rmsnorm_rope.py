"""CPU reference for split QKV, tensor-parallel RMSNorm and NeoX RoPE."""

# Keep FP32 normalization and intermediate input-dtype rounding; use PyTorch
# for dtype-general RoPE and preserve the suffix when rotary_dim < head_dim.
import torch
import torch.distributed as dist


def split_qkv_tp_rmsnorm_rope(
    input,
    cos,
    sin,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps,
    q_weight,
    k_weight,
    rotary_dim,
    tp_world,
    tp_group,
):
    assert head_dim == 1 << (head_dim - 1).bit_length()
    assert q_hidden_size % kv_hidden_size == 0
    device = input.device
    q, k, v = input.cpu().split((q_hidden_size, kv_hidden_size, kv_hidden_size), dim=1)
    if input.shape[0] == 0:
        return tuple(part.clone().to(device) for part in (q, k, v))

    # Reduce just the two local statistics on the process group's device.
    means = torch.stack([part.float().square().mean(1) for part in (q, k)], dim=1)
    if tp_world > 1:
        means = means.to(device)
        dist.all_reduce(means, group=tp_group)
        means = means.cpu() / tp_world

    half = rotary_dim // 2
    cosine = cos.cpu().reshape(-1, rotary_dim)[:, None, :half].float()
    sine = sin.cpu().reshape(-1, rotary_dim)[:, None, :half].float()

    def normalize_rotate(part, weight, mean):
        weighted = (
            part.float()
            * (1.0 / torch.sqrt(mean[:, None] + eps))
            * weight.cpu().flatten()[: part.shape[1]].float()
        )
        heads = weighted.to(input.dtype).reshape(part.shape[0], -1, head_dim).float()
        left, right = heads[..., :half], heads[..., half : 2 * half]
        rotated = torch.cat(
            (
                left * cosine - right * sine,
                right * cosine + left * sine,
                heads[..., 2 * half :],
            ),
            dim=-1,
        )
        return rotated.reshape_as(part).to(device=device, dtype=input.dtype)

    return (
        normalize_rotate(q, q_weight, means[:, 0]),
        normalize_rotate(k, k_weight, means[:, 1]),
        v.contiguous().clone().to(device),
    )

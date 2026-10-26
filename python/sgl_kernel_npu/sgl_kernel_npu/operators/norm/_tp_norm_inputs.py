"""TP QKV metadata validation; no host reads of input tensor values."""

import torch.distributed as dist

from ._qkv_inputs import _validate_layout


def prepare_inputs(
    input,
    cos,
    sin,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    q_weight,
    k_weight,
    rotary_dim,
    tp_world,
    tp_group,
):
    _validate_layout(input, q_hidden_size, kv_hidden_size, head_dim)
    if head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("rotary_dim must be positive, even and within head_dim")
    for name, value in (("cos", cos), ("sin", sin)):
        if (
            value.ndim < 1
            or value.shape[-1] != rotary_dim
            or value.numel() != input.shape[0] * rotary_dim
            or value.device != input.device
        ):
            raise ValueError(
                f"{name} must contain one rotary_dim row per token on the input device"
            )
    for name, value, width in (
        ("q_weight", q_weight, q_hidden_size),
        ("k_weight", k_weight, kv_hidden_size),
    ):
        if value is None or value.numel() < width or value.device != input.device:
            raise ValueError(
                f"{name} must cover the local hidden width on the input device"
            )
    if not isinstance(tp_world, int) or isinstance(tp_world, bool) or tp_world < 1:
        raise ValueError("tp_world must be a positive integer")
    # Preserve the original empty-batch early return without a collective.
    if input.shape[0] and tp_world > 1:
        if not dist.is_initialized():
            raise RuntimeError("TP world > 1 requires an initialized process group")
        if dist.get_world_size(group=tp_group) != tp_world:
            raise ValueError("tp_world must match the supplied process group's size")
    return tuple(value.contiguous() for value in (input, cos, sin, q_weight, k_weight))

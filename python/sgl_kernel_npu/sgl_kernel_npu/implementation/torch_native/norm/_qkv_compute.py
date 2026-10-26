"""Shared numerical operations on prepared Q/K heads and rotary tensors."""

import torch


def normalize(values, eps, weight, bias=None, *, gemma=False):
    values = values.float()
    if eps is not None:
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
        values = values * (weight.float() + 1.0 if gemma else weight.float())
        if bias is not None:
            values = values + bias.float()
    return values


def rotate(values, sin, cos, *, neox=True):
    rope_dim = sin.shape[-1]
    rot, tail = values[..., :rope_dim], values[..., rope_dim:]
    half = rope_dim // 2
    if neox:
        left, right = rot[..., :half], rot[..., half:]
        rotated = torch.cat((-right, left), dim=-1)
        output = rot * cos.float() + rotated * sin.float()
    else:
        even, odd = rot[..., 0::2], rot[..., 1::2]
        sin, cos = sin[..., :half].float(), cos[..., :half].float()
        output = torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)
    return torch.cat((output, tail), dim=-1)


def flatten_heads(values, dtype):
    return (
        values.reshape(values.shape[0], values.shape[1] * values.shape[2])
        .to(dtype)
        .contiguous()
    )

# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Copied from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/utils.py
# -*- coding: utf-8 -*-

import contextlib
import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

is_gather_supported = hasattr(triton.language, "gather")
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None

else:
    gather = tl.gather


def custom_device_ctx(index: int):
    return torch.npu.device(index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    cu_seqlens_i64 = cu_seqlens.to(torch.int64)
    return cu_seqlens_i64[1:] - cu_seqlens_i64[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in prepare_lens(cu_seqlens).unbind()
        ]
    )


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


def fused_qkvzba_split_reshape_cat_torch(
    mixed_qkvz: torch.Tensor,  # [B, 3072]
    mixed_ba: torch.Tensor,  # [B, 16]
    num_heads_qk: int = 4,
    num_heads_v: int = 8,
    head_qk: int = 128,
    head_v: int = 128,
):
    B = mixed_qkvz.shape[0]
    v_group_size = num_heads_v // num_heads_qk  # = 2

    # Step 1: Reshape to [B, num_heads_qk, per_head_dim]
    per_head_dim = 2 * head_qk + 2 * v_group_size * head_v  # 768
    x = mixed_qkvz.view(B, num_heads_qk, per_head_dim)

    # Extract components per head
    q = x[:, :, :head_qk]  # [B, 4, 128]
    k = x[:, :, head_qk : 2 * head_qk]  # [B, 4, 128]
    v_groups = x[:, :, 2 * head_qk : 2 * head_qk + v_group_size * head_v]  # [B, 4, 256]
    z_groups = x[:, :, 2 * head_qk + v_group_size * head_v :]  # [B, 4, 256]

    # Reshape V and Z to [B, num_heads_v, head_v]
    v = v_groups.reshape(B, num_heads_v, head_v)  # [B, 8, 128]
    z = z_groups.reshape(B, num_heads_v, head_v)  # [B, 8, 128]

    # Build mixed_qkv = [Q_flat, K_flat, V_flat]
    # Q_flat: concatenate all q heads → [B, 4*128]
    q_flat = q.reshape(B, -1)
    k_flat = k.reshape(B, -1)
    v_flat = v.reshape(B, -1)
    mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=1)  # [B, 2048]

    # Process mixed_ba: [B, 16] → view as [B, 4, 4] → split b/a
    ba = mixed_ba.view(B, num_heads_qk, 2 * v_group_size)  # [B, 4, 4]
    b = ba[:, :, :v_group_size].reshape(B, num_heads_v)  # [B, 8]
    a = ba[:, :, v_group_size:].reshape(B, num_heads_v)  # [B, 8]

    return mixed_qkv, z, b, a

"""Top-k/top-p renormalization using fused filtering and RN restoration.

The only implementation is the packed-key Triton path for NPU FP32
probabilities. There is no environment-controlled dispatch or scatter fallback.
"""

from typing import Union

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


def _prepare_probs(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")
    if probs.dtype != torch.float32:
        raise ValueError("probs must use torch.float32")
    if probs.device.type != "npu":
        raise ValueError("probs must be on an NPU")
    if not 1 <= probs.shape[1] <= 2**23:
        raise ValueError("probs must have vocab_size in [1, 2**23] for exact FP32 keys")
    return probs.contiguous()


def _as_batch_threshold(
    value: Union[torch.Tensor, int, float],
    probs: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.full((probs.shape[0],), value, device=probs.device, dtype=dtype)
    if value.ndim == 0:
        return value.to(device=probs.device, dtype=dtype).expand(probs.shape[0])
    if value.ndim != 1 or value.shape[0] != probs.shape[0]:
        raise ValueError(f"{name} must be a scalar or have shape [batch]")
    return value.to(device=probs.device, dtype=dtype)


@triton.jit
def _filter_encode_kernel(
    values,
    indices,
    cumulative,
    thresholds,
    keys,
    R: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    THRESHOLD_STRIDE: tl.constexpr,
    TOP_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < V
        address = row * V + offsets
        value = tl.load(values + address, valid, other=0.0)
        token = tl.load(indices + address, valid, other=0).to(tl.float32)
        if TOP_P:
            threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
            prefix = tl.load(cumulative + address, valid, other=0.0)
            # Preserve inclusive cumsum minus current value, including its
            # rounding. Do not substitute a different exclusive scan.
            removed = prefix - value > threshold
            value = tl.where(removed, 0.0, value)
            tl.store(values + address, value, valid)
        else:
            threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
            value = tl.where(offsets >= threshold, 0.0, value)
            tl.store(values + address, value, valid)
        # V <= 2**23 makes every encoded integer exactly representable in FP32.
        key = token * 2.0 + (value != 0.0).to(tl.float32)
        tl.store(keys + address, key, valid)


@triton.jit
def _restore_kernel(
    probs,
    ordered_keys,
    denominator,
    output,
    R: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < V
        address = row * V + offsets
        value = tl.load(probs + address, valid, other=0.0)
        key = tl.load(ordered_keys + address, valid, other=0.0)
        removed = key == offsets.to(tl.float32) * 2.0
        value = tl.where(removed, 0.0, value)
        scale = tl.load(denominator + row)
        # Fixed round-to-nearest division; never replace with rcp * x.
        value = tl.div_rn(value, scale)
        tl.store(output + address, value, valid)


def filter_and_renorm(
    probs, sorted_values, sorted_indices, thresholds, cumulative=None
):
    """Filter native sorted values, restore vocabulary order and divide with RN.

    The caller supplies contiguous NPU FP32 [rows, vocab] tensors, a nonempty
    vocabulary of at most 2**23 entries, and converted/clamped row thresholds.
    cumulative selects top-p; its absence selects top-k. Only sorted_values is
    modified. Keep the native sorted-value reduction order and denominator floor.
    """
    rows, vocab = probs.shape
    if rows == 0:
        return torch.empty_like(probs)
    blocks = triton.cdiv(vocab, 2048)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    keys = torch.empty_like(sorted_values)
    _filter_encode_kernel[grid](
        sorted_values,
        sorted_indices,
        cumulative if cumulative is not None else sorted_values,
        thresholds,
        keys,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        cumulative is not None,
        2048,
    )
    denominator = sorted_values.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    # Sort exact FP32 token keys; discard sort indices. This avoids the slow
    # full-vocabulary scatter while retaining the native probability-sort ties.
    ordered_keys = keys.sort(dim=-1).values
    output = torch.empty_like(probs)
    _restore_kernel[grid](
        probs,
        ordered_keys,
        denominator,
        output,
        rows,
        vocab,
        blocks,
        2048,
    )
    return output


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_ks: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Keep top-k tokens and renormalize NPU FP32 probabilities with RN division."""
    probs = _prepare_probs(probs)
    top_ks = _as_batch_threshold(top_ks, probs, torch.long, "top_ks").clamp(
        min=1, max=probs.shape[1]
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    return filter_and_renorm(probs, sorted_probs, sorted_indices, top_ks)


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_ps: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep the native sorted nucleus and renormalize NPU FP32 probabilities.

    Selection uses inclusive cumsum minus the current value, preserving the
    original rounding and tie behavior. Both endpoints of top_p are supported.
    """
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(
        min=0.0, max=1.0
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    return filter_and_renorm(
        probs, sorted_probs, sorted_indices, top_ps, cumulative_probs
    )

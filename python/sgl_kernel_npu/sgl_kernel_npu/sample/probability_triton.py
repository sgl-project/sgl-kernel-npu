"""Fused filtering and packed-key restoration for FP32 NPU probabilities.

Keep the native probability sort, cumsum, masked sorted-value sum and FP32 key
sort. No scatter, approximate nucleus selection or alternative tie rule is used.
"""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


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
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < V
        address = row * V + offsets
        value = tl.load(values + address, valid, other=0.0)
        token = tl.load(indices + address, valid, other=0).to(tl.float32)
        if MODE == 1:
            threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
            prefix = tl.load(cumulative + address, valid, other=0.0)
            # Preserve inclusive cumsum minus current value, including its
            # rounding. Do not substitute a different exclusive scan.
            removed = prefix - value > threshold
            value = tl.where(removed, 0.0, value)
            tl.store(values + address, value, valid)
        elif MODE == 2:
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
    FUSED_DIV: tl.constexpr,
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
        if FUSED_DIV:
            scale = tl.load(denominator + row)
            # Explicit round-to-nearest division; never replace with rcp * x.
            value = tl.div_rn(value, scale)
        tl.store(output + address, value, valid)


def filter_encode(sorted_values, sorted_indices, thresholds=None, cumulative=None):
    """Mask sorted_values in place and encode keys; inputs use native sort order.

    thresholds=None encodes already masked values (useful for isolated tests).
    Otherwise cumulative selects top-p; its absence selects top-k. Thresholds
    must already be converted and clamped by probability.py.
    """
    rows, vocab = sorted_values.shape
    keys = torch.empty_like(sorted_values)
    if rows == 0:
        return keys
    blocks = triton.cdiv(vocab, 2048)
    mode = 0 if thresholds is None else (1 if cumulative is not None else 2)
    stride = 0 if thresholds is None else thresholds.stride(0)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    _filter_encode_kernel[grid](
        sorted_values,
        sorted_indices,
        cumulative if cumulative is not None else sorted_values,
        thresholds if thresholds is not None else sorted_values,
        keys,
        rows,
        vocab,
        blocks,
        stride,
        mode,
        2048,
    )
    return keys


def restore_from_ordered_keys(probs, ordered_keys, denominator, division="native"):
    """Restore the original vocabulary order; native division is the default."""
    if division not in ("native", "rn"):
        raise ValueError("division must be 'native' or 'rn'")
    rows, vocab = probs.shape
    output = torch.empty_like(probs)
    if rows:
        blocks = triton.cdiv(vocab, 2048)
        _restore_kernel[(min(rows * blocks, get_device_properties()[1]),)](
            probs,
            ordered_keys,
            denominator,
            output,
            rows,
            vocab,
            blocks,
            division == "rn",
            2048,
        )
    if division == "native":
        output.div_(denominator)
    return output


def filter_and_renorm(
    probs,
    sorted_values,
    sorted_indices,
    thresholds,
    cumulative=None,
    division="native",
):
    keys = filter_encode(sorted_values, sorted_indices, thresholds, cumulative)
    # Keep shape, layout, reduction order and clamp identical to b47970c.
    denominator = sorted_values.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    ordered_keys = keys.sort(dim=-1).values
    return restore_from_ordered_keys(probs, ordered_keys, denominator, division)

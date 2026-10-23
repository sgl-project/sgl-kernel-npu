"""Torch-native Markov projection with fused, dtype-correct add and argmax."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

_BLOCK_V = 4096
_IDX_SENTINEL = tl.constexpr(2147483647)
_STEP_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}
_markov_launchers = {}


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _load_step_logits(
    base_ptr,
    bias_ptr,
    row,
    v,
    V: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    STEP_DTYPE: tl.constexpr,
):
    # Explicit FP32 arithmetic followed by a cast is necessary on Triton-Ascend
    # 3.2: an implicit BF16 add can reach the reduction without BF16 rounding.
    base = tl.load(base_ptr + row * BASE_STRIDE + v, v < V, 0).to(tl.float32)
    bias = tl.load(bias_ptr + row * BIAS_STRIDE + v, v < V, 0).to(tl.float32)
    return (base + bias).to(STEP_DTYPE).to(tl.float32)


@triton.jit
def _markov_greedy_row_kernel(
    base_ptr,
    bias_ptr,
    out_ptr,
    V: tl.constexpr,
    ROWS: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
    INDEX_TILE: tl.constexpr,
    STEP_DTYPE: tl.constexpr,
):
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        best = float("-inf")
        best_start = 0
        for part in range(tl.cdiv(V, TILE)):
            v = part * TILE + tl.arange(0, TILE)
            score = _load_step_logits(
                base_ptr, bias_ptr, row64, v, V, BASE_STRIDE, BIAS_STRIDE, STEP_DTYPE
            )
            cur = tl.reduce(tl.where(v < V, score, float("-inf")), 0, _maximum_with_nan)
            take = (cur > best) | ((cur != cur) & (best == best))
            best_start = tl.where(take, part * TILE, best_start)
            best = tl.where(take, cur, best)
        # Resolve indices only in the winning tile. Smaller index blocks keep
        # the compare/select/reduce working set within A3's UB capacity.
        result = _IDX_SENTINEL
        for sub in range(tl.cdiv(TILE, INDEX_TILE)):
            v = best_start + sub * INDEX_TILE + tl.arange(0, INDEX_TILE)
            score = _load_step_logits(
                base_ptr, bias_ptr, row64, v, V, BASE_STRIDE, BIAS_STRIDE, STEP_DTYPE
            )
            wins = tl.where(best != best, score != score, score == best)
            idx = tl.min(tl.where((v < V) & wins, v, _IDX_SENTINEL), 0)
            result = tl.minimum(result, idx)
        tl.store(out_ptr + row, result.to(tl.int64))


@triton.jit
def _markov_greedy_add_argmax_kernel(
    base_ptr,
    bias_ptr,
    vals_ptr,
    idxs_ptr,
    V: tl.constexpr,
    ROWS: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    STEP_DTYPE: tl.constexpr,
):
    for task in range(tl.program_id(0), ROWS * PARTS, tl.num_programs(0)):
        row = (task // PARTS).to(tl.int64)
        part = task % PARTS
        v = part * BLOCK + tl.arange(0, BLOCK)
        valid = v < V
        # Match torch.add's promoted output dtype before selecting a token.
        rounded = _load_step_logits(
            base_ptr, bias_ptr, row, v, V, BASE_STRIDE, BIAS_STRIDE, STEP_DTYPE
        )
        score = tl.where(valid, rounded, float("-inf"))
        best = tl.reduce(score, 0, _maximum_with_nan)
        wins = tl.where(best != best, score != score, score == best)
        idx = tl.min(tl.where(valid & wins, v, _IDX_SENTINEL), 0)
        tl.store(vals_ptr + task, best)
        tl.store(idxs_ptr + task, idx)


@triton.jit
def _markov_greedy_combine_kernel(
    vals_ptr,
    idxs_ptr,
    out_ptr,
    PARTS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        part = tl.arange(0, BLOCK)
        valid = part < PARTS
        vals = tl.load(vals_ptr + row * PARTS + part, valid, float("-inf"))
        idxs = tl.load(idxs_ptr + row * PARTS + part, valid, _IDX_SENTINEL)
        best = tl.reduce(vals, 0, _maximum_with_nan)
        wins = tl.where(best != best, vals != vals, vals == best)
        idx = tl.min(tl.where(valid & wins, idxs, _IDX_SENTINEL), 0)
        tl.store(out_ptr + row, idx.to(tl.int64))


def markov_greedy_step_triton(
    *,
    base_logits: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    """Match ``(base_logits + F.linear(prev_embeds, w2_weight)).argmax(-1)``.

    Projection and addition both retain Torch native's rounding behavior.
    NaNs win over finite values; ties select the first vocabulary index.
    """
    assert base_logits.ndim == prev_embeds.ndim == w2_weight.ndim == 2
    assert base_logits.device == prev_embeds.device == w2_weight.device
    bs, vocab = base_logits.shape
    assert vocab > 0 and w2_weight.shape[0] == vocab
    assert prev_embeds.shape == (bs, w2_weight.shape[1])
    assert base_logits.stride(1) == 1, "base_logits rows must be contiguous"
    assert w2_weight.stride(1) == 1, "markov_w2 weight rows must be contiguous"
    if bs == 0:
        return torch.empty((bs,), dtype=torch.int64, device=base_logits.device)
    # Let CANN select the matrix-multiply implementation for the actual SoC.
    bias = F.linear(prev_embeds, w2_weight)
    # Native reduction is faster than the fused row scan in this batch range
    # on A3; do not force a Triton path when it loses to the reference.
    if 16 <= bs < 32:
        return (base_logits + bias).argmax(-1)
    out = torch.empty((bs,), dtype=torch.int64, device=base_logits.device)
    step_dtype = _STEP_DTYPES[torch.promote_types(base_logits.dtype, bias.dtype)]
    parts = triton.cdiv(vocab, _BLOCK_V)
    inputs = (base_logits, bias)
    key = (
        base_logits.device,
        bs,
        vocab,
        tuple((x.dtype, x.stride(0), x.data_ptr() % 16) for x in inputs),
    )
    launchers = _markov_launchers.get(key)
    stream = triton.runtime.driver.active.get_current_stream(base_logits.device.index)
    if bs >= 32:
        if launchers is None:
            cores = get_device_properties()[1]
            tile = 4096 if step_dtype == tl.float32 else 8192
            tile = min(tile, triton.next_power_of_2(vocab))
            grid = (min(cores, bs),)
            kernel = _markov_greedy_row_kernel.warmup(
                *inputs,
                out,
                vocab,
                bs,
                base_logits.stride(0),
                bias.stride(0),
                tile,
                min(tile, 2048),
                step_dtype,
                grid=grid,
                multibuffer=False,
            )
            launchers = (kernel[(*grid, 1, 1)],)
            _markov_launchers[key] = launchers
        launchers[0](*inputs, out, stream=stream)
        return out
    vals = torch.empty((bs, parts), dtype=torch.float32, device=base_logits.device)
    idxs = torch.empty((bs, parts), dtype=torch.int32, device=base_logits.device)
    if launchers is None:
        cores = get_device_properties()[1]
        grid = (min(cores, bs * parts),)
        partial = _markov_greedy_add_argmax_kernel.warmup(
            *inputs,
            vals,
            idxs,
            vocab,
            bs,
            base_logits.stride(0),
            bias.stride(0),
            parts,
            _BLOCK_V,
            step_dtype,
            grid=grid,
        )
        combine_grid = (min(cores, bs),)
        combine = _markov_greedy_combine_kernel.warmup(
            vals,
            idxs,
            out,
            parts,
            bs,
            triton.next_power_of_2(parts),
            grid=combine_grid,
        )
        launchers = (partial[(*grid, 1, 1)], combine[(*combine_grid, 1, 1)])
        _markov_launchers[key] = launchers
    partial_launch, combine_launch = launchers
    partial_launch(*inputs, vals, idxs, stream=stream)
    combine_launch(vals, idxs, out, stream=stream)
    return out

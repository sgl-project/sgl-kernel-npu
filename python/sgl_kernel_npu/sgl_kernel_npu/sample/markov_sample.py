"""One Markov draft step: add the step bias and pick one token per row, the
argmax at temperature 0 and a Gumbel-max sample otherwise.

Ported from SGLang's DSpark MarkovSampleStep for Triton-Ascend.
"""

import functools
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

# A 1024-word tile keeps the live Philox and reduction vectors within UB.
_BLOCK = 1024
# Without Philox temporaries, larger tiles reduce the number of partials on A3.
_PRECOMPUTED_BLOCK = 4096
_NO_INDEX = tl.constexpr(2147483647)
# 2**-32: maps a 32-bit Philox word to a uniform in (0, 1].
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_PHILOX_ROUNDS = 10
_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


@triton.jit
def _round_to_step_dtype(value, ROUND: tl.constexpr):
    if ROUND == 1:
        value = value.to(tl.bfloat16).to(tl.float32)
    if ROUND == 2:
        value = value.to(tl.float16).to(tl.float32)
    return value


@triton.jit
def _negative_log1p_negative(uniform):
    # Compensate the rounding of 1-u: the CANN libdevice log1p loses small u.
    complement = 1.0 - uniform
    correction = (1.0 - complement) - uniform
    noise = -tl.log(complement) - correction / complement
    noise = tl.where(complement == 0.0, float("inf"), noise)
    return tl.where(uniform == 0.0, 0.0, noise)


@triton.jit
def _markov_sample_partial_kernel(
    B,
    X,
    T,
    OFFSET,
    L,
    P,
    E,
    BS,
    XS,
    LS,
    ES,
    WIDTH,
    PADDED,
    VOCAB_START,
    PARTS,
    ROWS,
    KEY,
    STEP,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE: tl.constexpr,
    SELECT: tl.constexpr,
    SAMPLE: tl.constexpr,
    USE_EXP_NOISE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    for task in range(tl.program_id(0), ROWS * PARTS, tl.num_programs(0)):
        row, part = task // PARTS, task % PARTS
        i = part * BLOCK + tl.arange(0, BLOCK)
        valid = i < WIDTH
        in_shard = i < PADDED
        value = tl.load(X + row * XS + i, in_shard, 0).to(tl.float32)
        if HAS_BIAS:
            value += tl.load(B + row * BS + i, valid, 0).to(tl.float32)
        value = _round_to_step_dtype(value, ROUND)
        if STORE:
            tl.store(L + row * LS + i, value.to(L.dtype.element_ty), in_shard)
        if SELECT:
            _select_block(
                value,
                T,
                OFFSET,
                P,
                E,
                ES,
                row,
                part,
                i,
                valid,
                PARTS,
                KEY,
                STEP,
                VOCAB_START,
                BLOCK,
                SAMPLE,
                USE_EXP_NOISE,
                PHILOX_ROUNDS,
            )


@triton.jit
def _select_block(
    value,
    T,
    OFFSET,
    P,
    E,
    ES,
    row,
    part,
    i,
    valid,
    PARTS,
    KEY,
    STEP,
    VOCAB_START,
    BLOCK: tl.constexpr,
    SAMPLE: tl.constexpr,
    USE_EXP_NOISE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    key = value
    if SAMPLE:
        temperature = tl.load(T + row)
    else:
        temperature = 0.0
    if temperature != 0.0:
        if USE_EXP_NOISE:
            exp_noise = tl.load(E + row * ES + i, valid, 1.0)
        else:
            # One Philox call yields four words; word k goes to token 4 * q + k.
            quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
            lane = tl.zeros([BLOCK // 4], tl.int64)
            w0, w1, w2, w3 = tl.philox(
                KEY,
                quad.to(tl.uint32),
                (lane + row).to(tl.uint32),
                (lane + STEP).to(tl.uint32),
                ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
                PHILOX_ROUNDS,
            )
            bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
            # Triton-Ascend can sign-extend Philox uint32 words when widening.
            bits = bits.to(tl.int64) & 0xFFFFFFFF
            uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
            exp_noise = _negative_log1p_negative(uniform)
        key = value / temperature - tl.log(exp_noise)
    nan = valid & (key != key)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, key, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (key == maximum))
    idx = tl.min(tl.where(wins, i + VOCAB_START, _NO_INDEX), 0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(P + (row * PARTS + part) * 2, maximum)
    tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


@triton.jit
def _markov_sample_finish_kernel(
    P,
    OUT,
    BS,
    PARTS: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), BS, tl.num_programs(0)):
        i = tl.arange(0, BLOCK)
        valid = i < PARTS * WORLD
        rank, part = i // PARTS, i % PARTS
        offset = ((rank * BS + row) * PARTS + part) * 2
        value = tl.load(P + offset, valid, -float("inf"))
        idx = tl.load(P + offset + 1, valid, 0).to(tl.int32, bitcast=True)
        # Candidate indices are nonnegative int32 bit patterns.
        idx &= _NO_INDEX
        valid &= idx != _NO_INDEX
        nan = valid & (value != value)
        has_nan = tl.sum(nan.to(tl.int32), 0) > 0
        maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
        wins = tl.where(has_nan, nan, valid & (value == maximum))
        result = tl.min(tl.where(wins, idx, _NO_INDEX), 0)
        tl.store(OUT + row, result.to(tl.int64))


def _launch_partial(
    base_logits,
    bias,
    *,
    width,
    store,
    temperatures,
    key,
    offset,
    step,
    step_dtype,
    has_sampled_rows,
    exp_noise=None,
    vocab_start=0,
    select=True,
):
    # Returns the [rows, parts * 2] (value, index bits) candidates of each block.
    rows, padded = base_logits.shape
    # The padded width fixes the transport shape on every rank; WIDTH masks padding.
    block = (
        _BLOCK
        if exp_noise is None
        else min(_PRECOMPUTED_BLOCK, triton.next_power_of_2(padded))
    )
    parts = triton.cdiv(padded, block)
    assert parts > 0
    partial = torch.empty(
        (rows, parts, 2), device=base_logits.device, dtype=torch.float32
    )
    _markov_sample_partial_kernel[(min(rows * parts, get_device_properties()[1]),)](
        base_logits if bias is None else bias,
        base_logits,
        temperatures,
        offset,
        base_logits if store is None else store,
        partial,
        base_logits if exp_noise is None else exp_noise,
        0 if bias is None else bias.stride(0),
        base_logits.stride(0),
        0 if store is None else store.stride(0),
        0 if exp_noise is None else exp_noise.stride(0),
        width,
        padded,
        vocab_start,
        parts,
        rows,
        key,
        step,
        block,
        _ROUND_CODE[step_dtype],
        bias is not None,
        store is not None,
        select,
        has_sampled_rows is not False,
        exp_noise is not None,
        _PHILOX_ROUNDS,
        num_warps=1,
    )
    return partial.view(rows, parts * 2)


def _check(
    base_logits,
    bias,
    temperatures,
    offset,
    step_dtype,
    vocab_start,
    gather,
    gather_logits,
    logits_out,
):
    assert base_logits.ndim == bias.ndim == 2
    assert bias.shape[0] == base_logits.shape[0]
    assert bias.shape[1] <= base_logits.shape[1]
    assert bias.stride(1) == base_logits.stride(1) == 1
    assert temperatures.shape == (base_logits.shape[0],)
    assert temperatures.dtype == torch.float32 and temperatures.stride(0) == 1
    assert offset.dtype == torch.int64 and offset.numel() == 1
    assert step_dtype in _ROUND_CODE, step_dtype
    # Philox words go to tokens in quads of global ids.
    assert vocab_start % 4 == 0, vocab_start
    if logits_out is not None:
        assert logits_out.dtype == step_dtype and logits_out.stride(1) == 1
        if gather is None:
            assert logits_out.shape == base_logits.shape
        else:
            assert gather_logits is not None
            assert logits_out.shape[0] == base_logits.shape[0]


def markov_sample_step(
    *,
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    temperatures: torch.Tensor,
    key: int,
    offset: torch.Tensor,
    step: int,
    step_dtype: torch.dtype,
    vocab_start: int = 0,
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    gather_logits: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    logits_out: Optional[torch.Tensor] = None,
    has_sampled_rows: Optional[bool] = None,
    exp_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add bias, round to step_dtype and select global token ids on NPU.

    Zero temperatures use argmax; other rows use Philox-keyed Gumbel sampling.
    bias covers valid columns of each padded base_logits shard. gather exchanges
    per-block candidates; when logits_out is kept, gather_logits exchanges the
    step logits instead. has_sampled_rows=False compiles a greedy-only kernel.

    exp_noise optionally supplies precomputed FP32 Exp(1) samples. Generate them
    before this call, for example with ``torch.empty(...).exponential_()`` on NPU,
    to bypass Triton Philox. They must be positive, with unit column stride and
    the same shape as base_logits (or logits_out for gathered kept logits).
    The caller owns their RNG sequence; key/offset/step apply only without them.
    Capturing exponential_() requires support from the installed torch-npu build.
    Otherwise, refresh exp_noise outside the graph before each replay.
    """
    _check(
        base_logits,
        bias,
        temperatures,
        offset,
        step_dtype,
        vocab_start,
        gather,
        gather_logits,
        logits_out,
    )
    if exp_noise is not None:
        noise_shape = (
            logits_out.shape
            if gather is not None and logits_out is not None
            else base_logits.shape
        )
        assert exp_noise.shape == noise_shape
        assert exp_noise.device == base_logits.device
        assert exp_noise.dtype == torch.float32 and exp_noise.stride(1) == 1
    rows, width = bias.shape
    launch = functools.partial(
        _launch_partial,
        temperatures=temperatures,
        key=key,
        offset=offset,
        step=step,
        step_dtype=step_dtype,
        has_sampled_rows=has_sampled_rows,
        exp_noise=exp_noise,
    )
    if gather is not None and logits_out is not None:
        local = torch.empty(
            base_logits.shape, device=base_logits.device, dtype=step_dtype
        )
        launch(base_logits, bias, width=width, store=local, select=False)
        full = gather_logits(local)[:, : logits_out.shape[1]]
        gathered = launch(full, None, width=full.shape[1], store=logits_out)
    else:
        partial = launch(
            base_logits,
            bias,
            width=width,
            store=logits_out,
            vocab_start=vocab_start,
        )
        gathered = partial if gather is None else gather(partial)
    parts = gathered.shape[1] // 2
    world = gathered.shape[0] // rows
    result = torch.empty(rows, device=base_logits.device, dtype=torch.int64)
    _markov_sample_finish_kernel[(min(rows, get_device_properties()[1]),)](
        gathered,
        result,
        rows,
        parts,
        world,
        triton.next_power_of_2(parts * world),
        num_warps=1,
    )
    return result

"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

_IDX_SENTINEL = tl.constexpr(2147483647)
_MARKOV_BLOCK_V = 256
_MARKOV_BLOCK_R = 32


class MarkovGreedyStep:
    """One greedy markov draft step, fused.

    Computes ``argmax_v(base_logits[:, v] + dot(w2_weight[v, :], prev_embeds))``
    in a single pass over the (vocab x rank) weight: no full-vocab bias or
    step-logits materialization, no separate GEMV / add / two-pass argmax
    launches. Numerics: the dot and the add accumulate in fp32, while
    MarkovSampleStep rounds the bias and the sum to the step dtype; near-tie
    winners can differ, so a one-row block can draft differently from the same
    row in a larger batch. Drafts are proposals only (target verify guards the
    output), so the impact is bounded to accept-rate noise on near ties.
    """

    @classmethod
    def triton(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        return markov_greedy_step_triton(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )

    execute = triton


@triton.jit
def _markov_greedy_partial_kernel(
    base_ptr,
    embed_ptr,
    w2_ptr,
    tile_val_ptr,
    tile_idx_ptr,
    V,
    R,
    stride_base_row,
    stride_embed_row,
    stride_w2_v,
    n_tiles,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    _grid_0: tl.constexpr,
    _grid_1: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0 * _grid_1, tl.num_programs(0)):
        row = _job // _grid_1
        tile = _job % _grid_1
        offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        for r0 in range(0, R, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            mask_r = offs_r < R
            embed = tl.load(
                embed_ptr + row * stride_embed_row + offs_r, mask=mask_r, other=0.0
            ).to(tl.float32)
            w2 = tl.load(
                w2_ptr + offs_v[:, None] * stride_w2_v + offs_r[None, :],
                mask=mask_v[:, None] & mask_r[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w2 * embed[None, :], axis=1)
        base = tl.load(
            base_ptr + row * stride_base_row + offs_v, mask=mask_v, other=float("-inf")
        ).to(tl.float32)
        score = tl.where(mask_v, base + acc, float("-inf"))
        tile_best = tl.max(score, axis=0)
        idx = tl.where(score == tile_best, offs_v, _IDX_SENTINEL)
        tl.store(tile_val_ptr + row * n_tiles + tile, tile_best)
        tl.store(tile_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _markov_greedy_combine_kernel(
    tile_val_ptr,
    tile_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        row = _job
        offs = tl.arange(0, BLOCK_TILES)
        mask = offs < n_tiles
        vals = tl.load(
            tile_val_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
        )
        idxs = tl.load(
            tile_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
        )
        best = tl.max(vals, axis=0)
        cand = tl.where(vals == best, idxs, _IDX_SENTINEL)
        tl.store(next_tokens_ptr + row, tl.min(cand, axis=0).to(tl.int64))


def markov_greedy_step_triton(
    *, base_logits: torch.Tensor, prev_embeds: torch.Tensor, w2_weight: torch.Tensor
) -> torch.Tensor:
    bs, vocab = base_logits.shape
    rank = w2_weight.shape[1]
    device = base_logits.device
    assert base_logits.stride(1) == 1, "base_logits rows must be contiguous"
    assert w2_weight.stride(1) == 1, "markov_w2 weight rows must be contiguous"
    prev_embeds = prev_embeds.contiguous()
    n_tiles = triton.cdiv(vocab, _MARKOV_BLOCK_V)
    tile_vals = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    tile_idxs = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)
    _markov_greedy_partial_kernel[
        max(1, min(get_device_properties()[1], bs * n_tiles)),
    ](
        base_logits,
        prev_embeds,
        w2_weight,
        tile_vals,
        tile_idxs,
        vocab,
        rank,
        base_logits.stride(0),
        prev_embeds.stride(0),
        w2_weight.stride(0),
        n_tiles,
        BLOCK_V=_MARKOV_BLOCK_V,
        BLOCK_R=_MARKOV_BLOCK_R,
        _grid_0=bs,
        _grid_1=n_tiles,
        num_warps=1,
    )
    _markov_greedy_combine_kernel[max(1, min(get_device_properties()[1], bs)),](
        tile_vals,
        tile_idxs,
        next_tokens,
        n_tiles,
        BLOCK_TILES=triton.next_power_of_2(n_tiles),
        _grid_0=bs,
        num_warps=1,
    )
    return next_tokens


from sgl_kernel_npu.sample.dspark_kv_proj import CommitKvProj as CommitKvProj
from sgl_kernel_npu.sample.dspark_kv_proj import (
    _dequant_linear_weight as _dequant_linear_weight,
)

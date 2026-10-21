"""Torch oracles copied from SGLang 7036f4b0ad for migrated CUDA cases.

Reference code is test-only and never imports or calls NPU implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ceil_align(n, a):
    return (n + a - 1) // a * a


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
    def torch(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        return markov_greedy_step(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )


def markov_greedy_step(
    *, base_logits: torch.Tensor, prev_embeds: torch.Tensor, w2_weight: torch.Tensor
) -> torch.Tensor:
    step_logits = base_logits + F.linear(prev_embeds, w2_weight)
    return torch.argmax(step_logits, dim=-1)


def _dequant_linear_weight(linear: torch.nn.Module) -> torch.Tensor:
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(torch.bfloat16)
    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"unsupported wkv weight dtype {weight.dtype} for the fused commit kv proj; execute() should have routed this to the torch path (_fused_commit_kv_proj_supported)"
    block = 128
    scale = linear.weight_scale_inv
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    assert (
        tuple(scale.shape) == expected_scale_shape
    ), f"wkv weight_scale_inv shape {tuple(scale.shape)} does not match the 128x128 block grid {expected_scale_shape} for weight {tuple(weight.shape)}; execute() should have routed this to the torch path (_fused_commit_kv_proj_supported)"
    scale_full = scale.repeat_interleave(block, dim=0)[:out_dim]
    scale_full = scale_full.repeat_interleave(block, dim=1)[:, :in_dim]
    return (weight.to(torch.float32) * scale_full.to(torch.float32)).to(torch.bfloat16)


class CommitKvProj:
    @staticmethod
    def torch(*, main_x, wkv_linears, allow_strided_output=False):
        return [linear(main_x)[0] for linear in wkv_linears]

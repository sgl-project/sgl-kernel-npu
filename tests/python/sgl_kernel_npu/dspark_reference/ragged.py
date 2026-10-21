"""Torch oracles copied from SGLang 7036f4b0ad for migrated CUDA cases.

Reference code is test-only and never imports or calls NPU implementations.
"""

from __future__ import annotations

import msgspec
import torch


def ceil_align(n, a):
    return (n + a - 1) // a * a


class PaddedToBucket:

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )


def pad_verify_lens_to_bucket(
    *, verify_lens: torch.Tensor, graph_num_tokens: int, bs: int, padded_bs: int
) -> torch.Tensor:
    assert (
        padded_bs >= bs
    ), f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this batch's requests"
    device = verify_lens.device
    num_pad_reqs = padded_bs - bs
    padded = verify_lens.to(torch.int32)
    leftover = graph_num_tokens - padded.to(torch.int64).sum()
    if num_pad_reqs > 0:
        base = leftover // num_pad_reqs
        rem = leftover - base * num_pad_reqs
        pad_block = base + (
            torch.arange(num_pad_reqs, device=device, dtype=torch.int64) < rem
        )
        padded = torch.cat([padded, pad_block.to(torch.int32)])
    else:
        padded = padded.clone()
        padded[-1] = (padded[-1].to(torch.int64) + leftover).to(torch.int32)
    return padded


class QoIndptrResult(msgspec.Struct):
    qo_indptr: torch.Tensor
    extend_start_loc: torch.Tensor


class BuildQoIndptr:

    @classmethod
    def torch(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        return build_qo_indptr(verify_lens=verify_lens)


def build_qo_indptr(*, verify_lens: torch.Tensor) -> QoIndptrResult:
    verify_lens = verify_lens.to(torch.int32)
    cumsum = torch.cumsum(verify_lens, dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=verify_lens.device)
    qo_indptr = torch.cat([zero, cumsum])
    extend_start_loc = qo_indptr[:-1].clone()
    return QoIndptrResult(qo_indptr=qo_indptr, extend_start_loc=extend_start_loc)

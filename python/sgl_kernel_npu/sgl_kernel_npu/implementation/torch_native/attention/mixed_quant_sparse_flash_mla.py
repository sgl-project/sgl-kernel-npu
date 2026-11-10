# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl under implementation/cannbot-dsl for the full text of the License.

"""Independent mixed FP8/FP4 sparse MLA reference using CPU Torch arithmetic."""

import torch


def _decode_rows(rows, fp4):
    rows = rows.contiguous()
    encoded = rows[..., : 256 if fp4 else 512]
    scales = rows[..., 256 if fp4 else 512 :].contiguous().view(torch.bfloat16).float()
    if fp4:
        codes = torch.stack((encoded & 15, encoded >> 4), dim=-1).flatten(-2).long()
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        values = levels[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    else:
        values = encoded.contiguous().view(torch.float8_e4m3fn).float()
    return (
        (values * scales.repeat_interleave(16 if fp4 else 32, dim=-1))
        .bfloat16()
        .float()
    )


def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse
):
    """Write caller-owned out/lse; ignore backend-specific scheduling metadata.

    TND queries, one shared KV head and PA_BBND packed pages match the DSL path.
    CPU execution is a precision reference, not a graph-capturable fast path.
    """
    if quant_mode != 1 or layout_q != "TND" or layout_kv != "PA_BBND":
        raise NotImplementedError("Reference supports quant_mode=1, TND/PA_BBND")
    if sinks is None:
        raise ValueError("sinks is required")
    if q.shape[1:] != (64, 512):
        raise ValueError("Expected q[T,64,512]")

    def cpu(x):
        return None if x is None else x.detach().cpu()

    with torch.device("cpu"):
        query = cpu(q).float()
        sink = cpu(sinks).float()
        bounds = [0, q.shape[0]] if cu_seqlens_q is None else cpu(cu_seqlens_q).tolist()
        used = None if seqused_q is None else cpu(seqused_q).tolist()
        sides = []
        for kv, table, indices, lengths, fp4 in (
            (ori_kv, ori_block_table, ori_sparse_indices, ori_topk_length, False),
            (cmp_kv, cmp_block_table, cmp_sparse_indices, cmp_topk_length, True),
        ):
            if kv is not None:
                if table is None or indices is None:
                    raise ValueError(
                        "Each KV side requires its block table and indices"
                    )
                sides.append((cpu(kv), cpu(table), cpu(indices), cpu(lengths), fp4))
        result = torch.zeros(query.shape, dtype=q.dtype)
        logsum = torch.empty((1, q.shape[0], q.shape[1]), dtype=torch.float32)
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        for batch, (start, end) in enumerate(zip(bounds, bounds[1:])):
            for row in range(start, end):
                selected_sides = []
                if used is None or row - start < used[batch]:
                    for kv, table, indices, lengths, fp4 in sides:
                        n = (
                            indices.shape[-1]
                            if lengths is None
                            else int(lengths.reshape(-1)[row])
                        )
                        ids = indices[row].reshape(-1)[:n].long()
                        ids = ids[ids >= 0]
                        page_size = kv.shape[1]
                        pages = table[batch, ids // page_size].long()
                        selected_sides.append(
                            _decode_rows(kv[pages, ids % page_size, 0], fp4)
                        )
                # BF16 PV probabilities and independent dense FP64 LSE reference.
                row_max, row_sum = sink.clone(), torch.ones_like(sink)
                acc = torch.zeros((q.shape[1], q.shape[2]))
                dense_logits = [sink.double()[:, None]]
                for values in selected_sides:
                    dense_logits.append(query[row].double() @ values.double().T * scale)
                    for offset in range(0, len(values), 128):
                        tile = values[offset : offset + 128]
                        scores = query[row] @ tile.T * scale
                        new_max = torch.maximum(row_max, scores.amax(-1))
                        correction = torch.exp(row_max - new_max)
                        weights = torch.exp(scores - new_max[:, None])
                        row_sum = row_sum * correction + weights.sum(-1)
                        acc = (
                            acc * correction[:, None]
                            + weights.bfloat16().float() @ tile
                        )
                        row_max = new_max
                result[row] = (acc / row_sum[:, None]).to(q.dtype)
                logsum[0, row] = torch.logsumexp(
                    torch.cat(dense_logits, dim=1), dim=1
                ).float()
    out.copy_(result.to(out.device))
    if return_softmax_lse:
        lse.copy_(logsum.to(lse.device))

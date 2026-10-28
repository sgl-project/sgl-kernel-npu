"""Independent CPU reference for the BF16 BSND paged SWA path.

This deliberately decodes the packed KV bytes and resolves the page table
with ordinary PyTorch operations. It is a correctness oracle, not a runtime
fallback. Compressed KV, sparse slot indices, TND and graph capture require
the existing dedicated native tests.
"""

import torch


def _unpack_kv(packed):
    rows = packed.detach().cpu().contiguous().view(torch.uint8)
    if rows.shape[-1] != 640:
        raise ValueError("packed KV rows must contain 640 bytes")
    rope = rows[..., :128].contiguous().view(torch.bfloat16).reshape(-1, 64)
    nope = rows[..., 128:576].contiguous().view(torch.float8_e4m3fn)
    nope = nope.float().reshape(-1, 7, 64)
    # E8M0 scale byte is an exponent with bias 127.
    exponent = rows[..., 576:583].reshape(-1, 7).to(torch.int32) - 127
    scale = torch.pow(2.0, exponent.float()).unsqueeze(-1)
    dequant_nope = (nope * scale).to(torch.bfloat16).reshape(-1, 448)
    return torch.cat((dequant_nope, rope), dim=-1)


def kv_quant_sparse_attn_sharedkv(
    q,
    *,
    ori_kv,
    ori_block_table,
    seqused_kv,
    sinks,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    layout_q="BSND",
    layout_kv="PA_ND",
    ori_win_left=127,
    ori_win_right=0,
    softmax_scale=1.0,
    **kwargs,
):
    """Return attention and empty LSE, matching the native operator's outputs.

    Only the uncompressed BSND paged sliding-window path is covered. Optional
    native-only metadata and quantization attributes are accepted in kwargs.
    """
    if layout_q != "BSND" or layout_kv != "PA_ND":
        raise NotImplementedError("reference supports BSND/PA_ND only")
    if (
        cmp_kv is not None
        or ori_sparse_indices is not None
        or cmp_sparse_indices is not None
    ):
        raise NotImplementedError("reference supports uncompressed paged SWA only")
    if q.dtype != torch.bfloat16 or q.ndim != 4 or q.shape[-1] != 512:
        raise ValueError("q must be BF16 [batch, query, heads, 512]")
    if ori_kv.dtype != torch.float8_e4m3fn:
        raise ValueError("ori_kv must contain packed FP8 KV")

    kv = _unpack_kv(ori_kv)
    table = ori_block_table.detach().cpu()
    lengths = seqused_kv.detach().cpu().tolist()
    query = q.detach().cpu()
    sink_logits = sinks.detach().cpu().float()
    block_size = ori_kv.shape[1]
    result = torch.zeros_like(query)
    for batch in range(query.shape[0]):
        kv_length = int(lengths[batch])
        for token in range(query.shape[1]):
            aligned = kv_length - query.shape[1] + token
            left = max(0, aligned - ori_win_left)
            right = min(kv_length - 1, aligned + ori_win_right)
            if right < left:
                continue
            logical = torch.arange(left, right + 1, device="cpu")
            physical = (
                table[batch, logical // block_size].long() * block_size
                + logical % block_size
            )
            selected = kv[physical].float()
            scores = query[batch, token].float() @ selected.T
            scores *= softmax_scale
            # The sink contributes probability mass with a zero value vector.
            maximum = torch.maximum(scores.max(dim=-1).values, sink_logits)
            weights = torch.exp(scores - maximum[:, None])
            denominator = weights.sum(dim=-1) + torch.exp(sink_logits - maximum)
            result[batch, token] = ((weights @ selected) / denominator[:, None]).to(
                torch.bfloat16
            )
    return result.to(q.device), torch.empty(0, dtype=torch.float32, device=q.device)

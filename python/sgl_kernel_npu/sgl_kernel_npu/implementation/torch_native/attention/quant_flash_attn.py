"""Independent eager MXFP8 attention reference, without a CANN dependency.

The numerical conventions follow the MXFP8 golden in ops-transformer v9.2.0:
attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test/common/
quant_flash_attn_golden.py. In particular, P is rounded to E4M3 before PV while
the softmax denominator uses unrounded probabilities. This implementation
unpacks the public paged layouts and evaluates one request at a time on CPU;
it is an accuracy reference, not a production fallback or graph implementation.
"""

import math

import torch


def _decode_scale(scale):
    encoded = scale.detach().view(torch.uint8).cpu()
    value = torch.exp2(encoded.float() - 127)
    return value.masked_fill(encoded == 255, float("nan"))


def _attention(q, k, v, softmax_scale, p_scale, causal):
    """FP32 online softmax with FP8 P; tensors have head/sequence/channel axes."""
    heads, q_len, _ = q.shape
    kv_len = k.shape[1]
    output = q.new_zeros((heads, q_len, v.shape[-1]))
    lse = q.new_full((heads, q_len), float("inf"))
    if kv_len == 0:
        return output, lse

    # MXFP8 golden tiles scores at 256 keys (128 for D256). Round the running
    # maximum to a power-of-two exponent before applying the probability scale.
    kv_tile = 128 if q.shape[-1] == 256 else 256
    ln2 = math.log(2)
    log_p = math.log(p_scale)
    for q_start in range(0, q_len, 128):
        query = q[:, q_start : q_start + 128]
        row_shape = (*query.shape[:-1], 1)
        maximum = q.new_full(row_shape, torch.finfo(torch.float32).min)
        denominator = q.new_zeros(row_shape)
        numerator = q.new_zeros((*query.shape[:-1], v.shape[-1]))
        for kv_start in range(0, kv_len, kv_tile):
            key = k[:, kv_start : kv_start + kv_tile]
            value = v[:, kv_start : kv_start + kv_tile]
            scores = (query @ key.transpose(-1, -2)) * softmax_scale
            if causal:
                query_position = torch.arange(
                    q_start, q_start + query.shape[1], device=q.device
                )
                key_position = torch.arange(
                    kv_start, kv_start + key.shape[1], device=q.device
                )
                mask = key_position[None, :] > (
                    query_position[:, None] + kv_len - q_len
                )
                scores.masked_fill_(mask, float("-inf"))
            tile_maximum = torch.ceil(scores.amax(-1, keepdim=True) / ln2) * ln2
            next_maximum = torch.maximum(maximum, tile_maximum - log_p)
            probabilities = torch.exp(scores - next_maximum)
            correction = torch.exp(maximum - next_maximum)
            denominator = denominator * correction + probabilities.sum(-1, keepdim=True)
            rounded_p = probabilities.to(torch.float8_e4m3fn).float()
            numerator = numerator * correction + rounded_p @ value
            maximum = next_maximum
        nonempty = denominator > 0
        result = numerator / torch.where(nonempty, denominator, 1)
        output[:, q_start : q_start + query.shape[1]] = result
        lse[:, q_start : q_start + query.shape[1]] = torch.where(
            nonempty, maximum + torch.log(denominator), float("inf")
        ).squeeze(-1)
    return output, lse


def quant_flash_attn(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    *,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
):
    """Evaluate mode-1 paged attention on CPU and return outputs on q.device.

    Supports TND Q/output, PA_BBND or PA_BNBD KV, TND/N2TGD Q scales and mask modes 0/3.
    Mode 3 expects the canonical compressed causal mask; its pattern is derived
    from effective Q/KV lengths. metadata and maximum-length hints are scheduling
    information and do not affect the reference. No inputs are modified.
    """
    if (
        quant_mode != 1
        or layout_q != "TND"
        or layout_out != "TND"
        or layout_kv not in ("PA_BBND", "PA_BNBD")
        or layout_q_descale not in ("TND", "N2TGD")
        or mask_mode not in (0, 3)
        or sinks is not None
        or win_left != -1
        or win_right != -1
    ):
        raise NotImplementedError(
            "The reference covers MXFP8 TND attention with PA_BBND or PA_BNBD KV, "
            "TND/N2TGD Q scales, mask 0/3 and no sinks or sliding window."
        )
    if any(tensor.dtype != torch.float8_e4m3fn for tensor in (q, k, v)):
        raise NotImplementedError("The MXFP8 reference requires E4M3 Q/K/V.")
    if block_table is None or cu_seqlens_q is None:
        raise ValueError("The paged TND reference needs block_table and cu_seqlens_q.")
    if seqused_kv is None and cu_seqlens_kv is None:
        raise ValueError("Provide seqused_kv or cu_seqlens_kv for the reference.")
    del attn_mask, metadata, max_seqlen_q, max_seqlen_kv

    query = q.detach().cpu().float()
    key = k.detach().cpu().float()
    value = v.detach().cpu().float()
    q_scale = _decode_scale(q_descale)
    if layout_q_descale == "N2TGD":
        q_scale = q_scale.permute(1, 0, 2, 3, 4)
    q_scale = q_scale.reshape(*query.shape[:-1], -1).repeat_interleave(32, dim=-1)
    query = query * q_scale[..., : query.shape[-1]]
    k_scale = _decode_scale(k_descale)
    v_scale = _decode_scale(v_descale)
    if layout_kv == "PA_BBND":
        # Unpack pages in [page, head, token, ...] order below.
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        k_scale = k_scale.transpose(1, 2)
        v_scale = v_scale.transpose(1, 2)
    k_scale = k_scale.flatten(-2).repeat_interleave(32, dim=-1)
    key = key * k_scale[..., : key.shape[-1]]
    # Each V scale pair covers consecutive groups of 32 tokens, not channels.
    v_scale = v_scale.permute(0, 1, 2, 4, 3)
    v_scale = v_scale.reshape(value.shape[0], value.shape[1], -1, value.shape[-1])
    value = value * v_scale.repeat_interleave(32, dim=2)[:, :, : value.shape[2]]

    q_boundaries = cu_seqlens_q.detach().cpu().tolist()
    q_lengths = (
        seqused_q.detach().cpu().tolist()
        if seqused_q is not None
        else [end - start for start, end in zip(q_boundaries, q_boundaries[1:])]
    )
    kv_lengths = (
        seqused_kv.detach().cpu().tolist()
        if seqused_kv is not None
        else cu_seqlens_kv.detach().cpu().diff().tolist()
    )
    pages = block_table.detach().cpu().long()
    probability_scale = 1.0 if p_scale is None else p_scale.detach().cpu().item()
    page_size = key.shape[2]
    groups = query.shape[1] // key.shape[1]
    output = query.new_zeros((*query.shape[:-1], value.shape[-1]), dtype=torch.bfloat16)
    lse = query.new_full((query.shape[1], query.shape[0]), float("inf"))
    for batch, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        if q_len == 0 or kv_len == 0:
            continue
        page_count = (kv_len + page_size - 1) // page_size
        page_ids = pages[batch, :page_count]
        request_key = key[page_ids].permute(1, 0, 2, 3).flatten(1, 2)[:, :kv_len]
        request_value = value[page_ids].permute(1, 0, 2, 3).flatten(1, 2)[:, :kv_len]
        request_key = request_key.repeat_interleave(groups, dim=0)
        request_value = request_value.repeat_interleave(groups, dim=0)
        start = q_boundaries[batch]
        request_query = query[start : start + q_len].transpose(0, 1)
        result, request_lse = _attention(
            request_query,
            request_key,
            request_value,
            softmax_scale,
            probability_scale,
            mask_mode == 3,
        )
        output[start : start + q_len] = result.transpose(0, 1).to(torch.bfloat16)
        lse[:, start : start + q_len] = request_lse
    return output.to(q.device), (
        lse.to(q.device)
        if return_softmax_lse
        else torch.empty((0,), dtype=torch.float32, device=q.device)
    )

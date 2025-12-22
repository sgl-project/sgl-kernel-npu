import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def attention_sinks_kernel(
    query,
    k_cache,
    v_cache,
    sinks,
    attn_out,
    block_tables,
    kv_seq_lens,
    scale,
    sliding_window_size,
    q_head_num: tl.constexpr,
    k_head_num: tl.constexpr,
    block_group_size: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    i_s, i_gh = tl.program_id(0), tl.program_id(1)
    i_kvh = i_gh * block_group_size // (q_head_num // k_head_num)

    kv_seq_len = tl.load(kv_seq_lens + i_s)
    page_num = tl.cdiv(kv_seq_len, PAGE_SIZE)
    page_num = min(page_num, MAX_BLOCKS)

    start_page_num = 0
    start_kv_len = 0
    if sliding_window_size != -1 and kv_seq_len > sliding_window_size:
        start_kv_len = (kv_seq_len - sliding_window_size).to(tl.int32)
        start_page_num = start_kv_len // PAGE_SIZE

    cur_page_start = i_s * MAX_BLOCKS
    offset_page = tl.arange(0, PAGE_SIZE)
    offset_d = tl.arange(0, D)
    Br: tl.constexpr = block_group_size

    sink = tl.load(sinks + i_gh * block_group_size + tl.arange(0, Br))
    history_max = tl.zeros([Br], dtype=tl.float32) + sink
    l = tl.zeros([Br], dtype=tl.float32)
    acc = tl.zeros([Br, D], dtype=tl.float32)

    offset_seq = (i_s * q_head_num + i_gh * block_group_size + tl.arange(0, Br)) * D
    q = tl.load(query + offset_seq[:, None] + offset_d[None, :]).to(tl.float32)

    for page_idx in range(start_page_num, page_num):
        block_idx = tl.load(block_tables + cur_page_start + page_idx)
        mask_page = ((page_idx * PAGE_SIZE + offset_page) < kv_seq_len) & (
            (page_idx * PAGE_SIZE + offset_page) >= start_kv_len
        )

        offset_k = (
            block_idx * PAGE_SIZE * k_head_num * D
            + offset_page[:, None] * k_head_num * D
            + i_kvh * D
            + offset_d[None, :]
        )
        k = tl.load(k_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)
        v = tl.load(v_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)

        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)
        qk = qk * scale
        qk = tl.where(mask_page[None, :], qk, float("-inf"))

        new_e_max = tl.maximum(tl.max(qk, 1), history_max)
        re_scale = tl.exp(history_max - new_e_max)
        p_exp = tl.exp(qk - new_e_max[:, None])

        # Online softmax update
        l = l * re_scale + tl.sum(p_exp, 1)
        acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)

        history_max = new_e_max

    sink = tl.math.exp(sink - history_max)
    l = l + sink
    acc = acc / l[:, None]
    tl.store(
        attn_out + offset_seq[:, None] + offset_d[None, :],
        acc.to(attn_out.type.element_ty),
    )


def attention_sinks_triton(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    S = query.shape[0]
    D = query.shape[-1] // q_head_num
    PAGE_SIZE = k_cache.shape[1]
    v_head_dim = v_cache.shape[-1]

    group_block_size = min(q_head_num // k_head_num, 16)
    group_block_num = q_head_num // group_block_size

    attn_output = torch.zeros(
        (S, q_head_num, v_head_dim),
        dtype=query.dtype,
        device=query.device,
    )

    grid = [S, group_block_num]
    attention_sinks_kernel[grid](
        query,
        k_cache,
        v_cache,
        sinks,
        attn_output,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
        group_block_size,
        D,
        PAGE_SIZE,
        block_tables.stride(0),
    )

    return attn_output.reshape(-1, q_head_num * v_head_dim)


@triton.jit
def attention_sinks_prefill_kernel(
    query,
    k_cache,
    v_cache,
    sinks,
    attn_out,
    cum_seq_lens,
    block_tables,
    kv_seq_lens,
    scale,
    sliding_window_size,
    q_head_num: tl.constexpr,
    k_head_num: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    i_b, i_qh = tl.program_id(0), tl.program_id(1)
    i_kvh = i_qh // (q_head_num // k_head_num)

    q_end_offset = tl.load(cum_seq_lens + i_b)
    q_start_offset = 0
    q_start_offset = q_start_offset.to(q_end_offset.dtype)
    if i_b > 0:
        q_start_offset = tl.load(cum_seq_lens + i_b - 1)

    Br: tl.constexpr = 16

    for i_s in range(q_start_offset, q_end_offset, Br):
        kv_seq_len = tl.load(kv_seq_lens + i_b) + i_s - q_end_offset + 1

        page_num = tl.cdiv(kv_seq_len + Br, PAGE_SIZE)
        page_num = min(page_num, MAX_BLOCKS)

        kv_seq_len_block = kv_seq_len + tl.arange(0, Br)
        start_kv_len_block = tl.zeros([Br], dtype=tl.int32)

        start_page_num = 0
        if sliding_window_size != -1:
            start_kv_len = max((kv_seq_len - sliding_window_size).to(tl.int32), 0)
            start_page_num = start_kv_len // PAGE_SIZE
            start_kv_len_block = max(
                (kv_seq_len_block - sliding_window_size).to(tl.int32), 0
            )

        cur_page_start = i_b * MAX_BLOCKS
        offset_page = tl.arange(0, PAGE_SIZE)
        offset_d = tl.arange(0, D)

        sink = tl.load(sinks + i_qh)
        history_max = tl.zeros([Br], dtype=tl.float32) + sink
        l = tl.zeros([Br], dtype=tl.float32)
        acc = tl.zeros([Br, D], dtype=tl.float32)

        offset_q = i_qh * D + offset_d
        offset_seq = (tl.arange(0, Br) + i_s) * D * q_head_num
        mask_seq = (tl.arange(0, Br) + i_s) < q_end_offset
        q = tl.load(
            query + offset_seq[:, None] + offset_q[None, :], mask=mask_seq[:, None]
        ).to(tl.float32)

        for page_idx in range(start_page_num, page_num):
            block_idx = tl.load(block_tables + cur_page_start + page_idx)
            cur_offset_page = page_idx * PAGE_SIZE + offset_page
            mask_page = (cur_offset_page[None, :] < kv_seq_len_block[:, None]) & (
                cur_offset_page[None, :] >= start_kv_len_block[:, None]
            )

            offset_k = (
                block_idx * PAGE_SIZE * k_head_num * D
                + offset_page[:, None] * k_head_num * D
                + i_kvh * D
                + offset_d[None, :]
            )
            k = tl.load(k_cache + offset_k).to(tl.float32)
            v = tl.load(v_cache + offset_k).to(tl.float32)

            k = tl.trans(k, (1, 0))
            qk = tl.dot(q, k)
            qk = qk * scale
            qk = tl.where(mask_page, qk, float("-inf"))

            new_e_max = tl.maximum(tl.max(qk, 1), history_max)
            re_scale = tl.exp(history_max - new_e_max)
            p_exp = tl.exp(qk - new_e_max[:, None])

            # Online softmax update
            l = l * re_scale + tl.sum(p_exp, 1)
            acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)

            history_max = new_e_max

        sink = tl.math.exp(sink - history_max)
        l = l + sink
        acc = acc / l[:, None]
        tl.store(
            attn_out + offset_seq[:, None] + offset_q[None, :],
            acc.to(attn_out.type.element_ty),
            mask=mask_seq[:, None],
        )


def attention_sinks_prefill_triton(
    query,
    k_cache,
    v_cache,
    sinks,
    seq_lens,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    S = query.shape[0]
    D = query.shape[-1] // q_head_num
    PAGE_SIZE = k_cache.shape[1]
    v_head_dim = v_cache.shape[-1]
    attn_output = torch.zeros(
        (S, q_head_num, v_head_dim),
        dtype=query.dtype,
        device=query.device,
    )

    cum_seq_lens = torch.cumsum(seq_lens, dim=0)
    B = seq_lens.shape[0]

    grid = [B, q_head_num]
    attention_sinks_prefill_kernel[grid](
        query,
        k_cache,
        v_cache,
        sinks,
        attn_output,
        cum_seq_lens,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
        D,
        PAGE_SIZE,
        block_tables.stride(0),
    )

    return attn_output.reshape(-1, q_head_num * v_head_dim)

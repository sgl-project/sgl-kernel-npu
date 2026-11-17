"""Sparse GQA decode Triton main kernel."""

import triton
import triton.language as tl


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit
def _gqa_share_sparse_decode_bnsd_kernel(
    q_ptr,  # [B, QH, D]
    sink_ptr,  # optional [QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    v_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks] or typed direct-map placeholder
    req_to_token_ptr,  # [num_requests, max_context] in direct-map mode
    req_pool_indices_ptr,  # [B] in direct-map mode
    idx_ptr,  # [KVH, B, max_topk]
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_rtt_r,
    stride_rtt_t,
    max_req_to_token_cols,
    num_pages,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    CHUNK_SIZE_T: tl.constexpr,
    # Selected (topk) blocks gathered into one K/V tile + one dot per loop step.
    # 1 == per-block path (bit-identical); only the online-softmax grouping changes.
    BLOCKS_PER_STEP: tl.constexpr,
    # Hoist the chunk's topk-idx load + page-id gather into a vectorized prologue,
    # breaking the per-step idx->page->K load dependency chain.
    PREFETCH_IDX: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_DIRECT_PAGE_LOOKUP: tl.constexpr,
    SANITIZE_PAGE_IDS: tl.constexpr,
):
    """Triton split-K sparse decode kernel with GQA sharing and online softmax."""
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.minimum(tl.load(seq_lens + pid_b).to(tl.int32), max_kv_len)

    # TopK list base for this KV head and request.
    # Iterate the fixed topk range and mask out -1 entries (tl.sum >= 0 is buggy on Ascend).
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b

    chunk_start_topk = pid_c * CHUNK_SIZE_T

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    dim_mask = off_d < head_dim

    # Q: [H, D]
    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    # Sink belongs only to chunk 0 so it is counted once across split-topk chunks.
    if HAS_SINK and pid_c == 0:
        sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
            None, :
        ] * stride_sink_d
        sink = tl.load(
            sink_ptr + sink_offsets,
            mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale
        m_i = qsink
        lse_i = qsink
    else:
        _neg = -1.0e30
        m_i = tl.full((BLOCK_SIZE_H,), _neg, dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), _neg, dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    # Iterate over the fixed topk slice assigned to this chunk. The actual valid
    # length is encoded by -1 sentinels in topk_idx.
    if BLOCKS_PER_STEP == 1:
        off_t_pf = tl.arange(0, BLOCK_SIZE_T)
        if PREFETCH_IDX:
            # Vectorized prologue: one gather for all chunk's selected blocks and
            # their page ids, so K/V loads in the loop depend only on registers.
            chunk_end_topk_pf = tl.minimum(chunk_start_topk + CHUNK_SIZE_T, max_topk)
            topk_pos_all = chunk_start_topk + off_t_pf
            logical_all = tl.load(
                idx_base + topk_pos_all * stride_ti_t,
                mask=topk_pos_all < chunk_end_topk_pf,
                other=-1,
            ).to(tl.int32)
            valid_pf = logical_all >= 0
            safe_pf = tl.maximum(logical_all, 0)
            if USE_DIRECT_PAGE_LOOKUP:
                req_idx_pf = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
                token_cols_pf = tl.minimum(
                    safe_pf * block_size, max_req_to_token_cols - 1
                )
                token_slots_pf = tl.load(
                    req_to_token_ptr
                    + req_idx_pf * stride_rtt_r
                    + token_cols_pf * stride_rtt_t,
                    mask=valid_pf,
                    other=0,
                ).to(tl.int64)
                phys_pf = token_slots_pf // block_size
                if SANITIZE_PAGE_IDS:
                    phys_pf = tl.minimum(tl.maximum(phys_pf, 0), num_pages - 1)
            else:
                phys_pf = tl.load(
                    block_table_ptr + pid_b * stride_bt_b + safe_pf * stride_bt_n,
                    mask=valid_pf,
                    other=0,
                ).to(tl.int64)
            phys_pf32 = phys_pf.to(tl.int32)
        for step in tl.range(CHUNK_SIZE_T):
            if PREFETCH_IDX:
                # Dynamic register-vector index via where+sum ([T] int ops).
                logical_block = tl.sum(
                    tl.where(off_t_pf == step, logical_all, 0), axis=0
                )
                physical_block = tl.sum(
                    tl.where(off_t_pf == step, phys_pf32, 0), axis=0
                ).to(tl.int64)
                valid_block = logical_block >= 0
            else:
                topk_pos = chunk_start_topk + step
                in_topk_range = topk_pos < max_topk

                logical_block = tl.load(
                    idx_base + topk_pos * stride_ti_t,
                    mask=in_topk_range,
                    other=-1,
                ).to(tl.int32)
                valid_block = logical_block >= 0

                if USE_DIRECT_PAGE_LOOKUP:
                    req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
                    safe_logical_block = tl.maximum(logical_block, 0)
                    token_col = tl.minimum(
                        safe_logical_block * block_size, max_req_to_token_cols - 1
                    )
                    token_slot = tl.load(
                        req_to_token_ptr
                        + req_idx * stride_rtt_r
                        + token_col * stride_rtt_t,
                        mask=valid_block,
                        other=0,
                    ).to(tl.int64)
                    physical_block = token_slot // block_size
                    if SANITIZE_PAGE_IDS:
                        physical_block = tl.minimum(
                            tl.maximum(physical_block, 0), num_pages - 1
                        )
                else:
                    physical_block = tl.load(
                        block_table_ptr
                        + pid_b * stride_bt_b
                        + logical_block * stride_bt_n,
                        mask=valid_block,
                        other=0,
                    ).to(tl.int64)

            pos = logical_block * block_size + off_n
            pos_mask = valid_block & (pos < seq_len)

            # K: [D, N]
            k_offsets = (
                physical_block * stride_k_block
                + off_n[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=dim_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )

            # V: [N, D]
            v_offsets = (
                physical_block * stride_v_block
                + off_n[:, None] * stride_v_offset
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d
            )
            v = tl.load(
                v_cache_ptr + v_offsets,
                mask=pos_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            # [H, D] @ [D, N] -> [H, N]
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(pos_mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            # Direct path: m_ij is finite (finite-floor init or qsink), so sentinel
            # slots yield p=0 and acc_o no-op naturally.
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp(m_i - m_ij)[:, None] + tl.dot(p.to(v.dtype), v)
            lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
            m_i = m_ij
    else:
        # Multi-block path: gather BLOCKS_PER_STEP blocks into one K/V tile per
        # step. Idx/page gathers are per-column (redundant x block_size but L2 hits).
        sub_id = off_n // block_size  # [N] which selected block in this step
        inn = off_n % block_size  # [N] token offset within that block
        chunk_end_topk = tl.minimum(chunk_start_topk + CHUNK_SIZE_T, max_topk)
        num_steps = tl.cdiv(chunk_end_topk - chunk_start_topk, BLOCKS_PER_STEP)
        for step in tl.range(num_steps, num_stages=1, disallow_acc_multi_buffer=True):
            topk_pos_col = chunk_start_topk + step * BLOCKS_PER_STEP + sub_id
            logical_block_col = tl.load(
                idx_base + topk_pos_col * stride_ti_t,
                mask=topk_pos_col < chunk_end_topk,
                other=-1,
            ).to(tl.int32)
            valid_col = logical_block_col >= 0
            safe_logical_col = tl.maximum(logical_block_col, 0)

            if USE_DIRECT_PAGE_LOOKUP:
                req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
                token_col = tl.minimum(
                    safe_logical_col * block_size, max_req_to_token_cols - 1
                )
                token_slot = tl.load(
                    req_to_token_ptr
                    + req_idx * stride_rtt_r
                    + token_col * stride_rtt_t,
                    mask=valid_col,
                    other=0,
                ).to(tl.int64)
                physical_block_col = token_slot // block_size
                if SANITIZE_PAGE_IDS:
                    physical_block_col = tl.minimum(
                        tl.maximum(physical_block_col, 0), num_pages - 1
                    )
            else:
                physical_block_col = tl.load(
                    block_table_ptr
                    + pid_b * stride_bt_b
                    + safe_logical_col * stride_bt_n,
                    mask=valid_col,
                    other=0,
                ).to(tl.int64)

            pos = logical_block_col * block_size + inn
            pos_mask = valid_col & (pos < seq_len)

            # K: [D, BPS*block_size]
            k_offsets = (
                physical_block_col[None, :] * stride_k_block
                + inn[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=dim_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )

            # [H, D] @ [D, BPS*block_size] -> [H, BPS*block_size]
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(pos_mask[None, :], qk, float("-inf"))

            # All-invalid step must not touch the accumulator (-inf - -inf = nan).
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            # Direct path: m_ij finite (finite-floor init) -> all-invalid step is
            # a natural no-op (p=0, acc_o unchanged); no has_valid guard.
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            # V load sequenced after qk/p so its UB live range starts as K's ends
            # (K+V tiles overflow the 192KB UB at BLOCK_SIZE_N >= 256 otherwise).
            v_offsets = (
                physical_block_col[:, None] * stride_v_block
                + inn[:, None] * stride_v_offset
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d
            )
            v = tl.load(
                v_cache_ptr + v_offsets,
                mask=pos_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            acc_o = acc_o * tl.exp(m_i - m_ij)[:, None] + tl.dot(p.to(v.dtype), v)
            lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
            m_i = m_ij

    # Final scale.
    # Empty chunks keep lse_i=-inf and should output clean zeros.
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]

    # Store partial output: [C, B, QH, D]
    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )

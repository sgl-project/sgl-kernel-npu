# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

PAD_SLOT_ID = -1


@triton.jit(
    do_not_specialize=[
        "batch",
        "state_len",
        "num_cache_lines",
        "stride_x_seq",
        "stride_x_token",
        "stride_conv_state_seq",
        "stride_state_indices",
        "stride_o_seq",
        "stride_o_token",
    ]
)
def _causal_conv1d_update_kernel_npu_tiled(
    # Pointers
    x_ptr,  # (batch, seqlen, dim) OR (num_tokens, dim) for varlen
    w_ptr,  # (width, dim)
    bias_ptr,
    conv_state_ptr,  # (num_cache_lines, dim, state_len)
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # same shape as x_ptr
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,  # max seqlen for varlen, or exact seqlen
    state_len,  # effective state_len computed in wrapper
    num_cache_lines,
    # Strides
    stride_x_seq,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices,
    stride_o_seq,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    # others
    pad_slot_id: tl.constexpr,
    # Meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # <= 6
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # tiling
    BLOCK_N: tl.constexpr,  # channel tile (C_TILE)
    B_TILE: tl.constexpr,  # batch tile
    T_CHUNK: tl.constexpr,  # token chunk for state update
):
    # program ids
    pid_b = tl.program_id(0)  # batch-tile id
    pid_c = tl.program_id(1)  # channel-tile id

    # channel indices for this program
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_w = idx_feats < dim

    # preload weights once per program (shared by B_TILE sequences)
    w_base = w_ptr + idx_feats * stride_w_dim
    # define to avoid "undefined" in branches
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )

    # bias vector once per program
    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # token index vector for chunked copy
    tok_vec = tl.arange(0, T_CHUNK)  # [T_CHUNK]

    # process B_TILE sequences inside the same program instance
    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi  # scalar tl.int32
        lane_active = b < batch  # scalar predicate

        # -------------------------
        # APC mapping (optional)
        # -------------------------
        if IS_APC_ENABLED:
            conv_state_init = tl.load(
                initial_state_idx + b, mask=lane_active, other=0
            ).to(tl.int32)
            current_last_index = tl.load(
                block_idx_last_scheduled_token + b, mask=lane_active, other=0
            ).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        # input cache line
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        # -------------------------
        # varlen (optional): revise seqlen_run and state_len_run like original kernel does
        # -------------------------
        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(
                tl.int64
            )
            qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(
                tl.int64
            )
            seqlen_run = (qe - qs).to(tl.int32)
            # revise effective state_len for shorter sequences (same formula as original)
            state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
            x_offset = (qs * stride_x_token).to(tl.int64)
            o_offset = (qs * stride_o_token).to(tl.int64)
        else:
            seqlen_run = tl.full((), seqlen, tl.int32)
            state_len_run = tl.full((), state_len, tl.int32)
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)

        # empty sequence -> skip (avoid early return because other lanes in tile)
        lane_active = lane_active & (seqlen_run > 0)

        # -------------------------
        # spec decoding offset (optional)
        # -------------------------
        if IS_SPEC_DECODING:
            conv_state_token_offset = (
                tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(
                    tl.int64
                )
                - 1
            )
            shift = tl.full((), 1, tl.int32)  # sliding by 1 in spec mode
        else:
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = seqlen_run  # normal mode shift by seqlen

        # -------------------------
        # STEP 1: read initial history cols BEFORE state update (out==x safe)
        # -------------------------
        conv_states_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )
        prior_tokens = (
            conv_states_base + conv_state_token_offset * stride_conv_state_tok
        )

        # define history vectors as zeros then load conditionally
        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        if KERNEL_WIDTH >= 2:
            col0 = tl.load(
                prior_tokens + 0 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(
                prior_tokens + 1 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(
                prior_tokens + 2 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 5:
            col3 = tl.load(
                prior_tokens + 3 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 6:
            col4 = tl.load(
                prior_tokens + 4 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)

        # -------------------------
        # STEP 2: chunked state update (replaces original NP2_STATELEN x BLOCK_N big block)
        # Semantics: conv_state <- concat(old_state, x)[-state_len_run:].
        # - If seqlen_run >= state_len_run: dst[:] = x[seqlen_run - state_len_run : seqlen_run]
        # - Else: keep = state_len_run - seqlen_run,
        #         dst[0:keep] = src[shift : shift+keep], dst[keep:keep+seqlen_run] = x[0:seqlen_run]
        # -------------------------
        # output cache line
        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run

        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), zero_i32).to(
            tl.int32
        )
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), zero_i32).to(
            tl.int32
        )

        # base pointers
        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = (
            conv_state_ptr
            + conv_states_offset * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )

        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        # A) shift old state into dst[0:keep_shift)  (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            src_tok = (dst_tok + shift).to(tl.int32)  # [T_CHUNK]
            m_tok = (
                use_shift
                & (dst_tok < keep_shift)
                & (src_tok < state_len_run)
                & (dst_tok < state_len_run)
            )
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )

            src_ptrs = (
                state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            )
            dst_ptrs = (
                state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            )
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        # B) append x into dst[keep_shift : keep_shift+seqlen_run) (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, seqlen, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            dst_tok = (keep_shift + x_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_offset < num_cache_lines)
            )

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = (
                state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            )
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # C) if seqlen_run >= state_len_run, overwrite dst with the tail of x
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            x_tok = (tail_start + dst_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_offset < num_cache_lines)
            )

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = (
                state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            )
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # -------------------------
        # STEP 3/4/5: causal conv1d (+ optional SiLU) and store output
        # This is original STEP3~5, but per-lane and without debug_barrier.
        # -------------------------
        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        # compute each token; keep tl.range so varlen can use seqlen_run as runtime trip count (like original)
        for idx_token in tl.range(seqlen_run):
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            # same selection logic as original (unrolled by KERNEL_WIDTH)
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 1:
                    # only x[t] * w0
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(
                        x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                    ).to(tl.float16)
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)

                acc += matrix_x.to(tl.float32) * matrix_w  # [BLOCK_N]

            if HAS_BIAS:
                acc += acc_bias

            acc = acc.to(tl.float16, fp_downcast_rounding="rtne")

            # roll history window
            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc.to(tl.float32)
                acc = acc / (1.0 + tl.exp(-acc))

            # store output
            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


def causal_conv1d_update_v2(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, seqlen, dim]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., state_len, dim), where state_len >= width - 1
    weight: (width, dim)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
            if conv_state_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, seqlen, dim) or (num_tokens, dim), same shape as `x`
    """
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(1)

    if query_start_loc is None:
        batch, seqlen, dim = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    width, _ = weight.shape
    num_cache_lines, state_len_total, _ = conv_state.size()

    # overwrite-on-x strategy same as original
    out = x

    stride_w_width, stride_w_dim = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_token, stride_x_dim = x.stride()
        stride_o_seq, stride_o_token, stride_o_dim = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_token, stride_istate_dim = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )

    # effective state_len exactly as original
    if num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    # -------- tiling heuristic--------
    # keep program count around ~[80..160]
    # vector core 40
    # TODO: use driver to get the vector core num
    CORE_HINT = 40
    # channel tile: 512 when dim large (reduce tasks), else 256
    block_n = 512 if dim >= 512 else 256
    g = triton.cdiv(dim, block_n)
    target = 2 * CORE_HINT  # ~80
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    # clamp to small set
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8

    # token chunk based on block_n (32KB UB idea); conservative
    t_chunk = 1 if block_n == 512 else 48

    def grid(META):
        return (
            triton.cdiv(batch, META["B_TILE"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel_npu_tiled[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )

    if unsqueeze:
        out = out.squeeze(1)
    return out.to(original_x_dtype)


def causal_conv1d_fn_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seqlens: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seqlens: (batch,)
    has_initial_state: (batch,)
    initial_states: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)

    out = out[..., :seqlen]
    if return_final_states:
        if initial_states is None:
            # x is unextended: sequences without an initial state start
            # (width - 1) tokens later, so shift their gather window back.
            base = seqlens - (width - 1) * (~has_initial_state)
        else:
            # x was extended with (width - 1) initial-state tokens per row;
            # the last (width - 1) inputs of every sequence sit at
            # [seqlens, seqlens + width - 2) regardless of has_initial_state.
            base = seqlens
        positions = base.unsqueeze(1) + torch.arange(
            width - 1, device=seqlens.device
        ).unsqueeze(0)
        indices = positions.unsqueeze(1).expand(-1, x.shape[-2], -1)
        final_states = (
            x.gather(2, indices.clamp(min=0)).masked_fill(indices < 0, 0).to(dtype_in)
        )  # (batch, dim, width - 1)
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states)


def prepare_data(
    x: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
):
    # Clamp cache_indices so pad slots (pad_slot_id=-1) do not wrap around via
    # negative indexing; pad rows are zeroed by has_initial_state and never
    # contribute to any output.
    initial_states = (
        torch.index_select(conv_states, 0, cache_indices.clamp(min=0))
        * has_initial_state[:, None, None]
        if has_initial_state is not None and has_initial_state.any()
        else None
    )

    seqlens = query_start_loc[1:] - query_start_loc[:-1]

    if x.ndim == 3:
        return x, initial_states, seqlens, None

    dtype, device = weight.dtype, weight.device
    batch_size = seqlens.size(0)
    max_T = seqlens.max()
    dim, cu_seq_len = x.size(0), query_start_loc[-1]

    x_flat = torch.zeros(size=(dim, batch_size * max_T), dtype=dtype, device=device)

    base_idx = torch.arange(batch_size, device=device, dtype=torch.int32) * max_T
    base_t = torch.arange(max_T, device=device, dtype=torch.int32).unsqueeze(0)
    mask = base_t < seqlens.unsqueeze(1)
    indices = (base_idx.unsqueeze(1) + base_t)[mask]

    x_flat.index_copy_(1, indices, x[..., :cu_seq_len])
    x_pad = x_flat.view(dim, batch_size, max_T).transpose(0, 1).contiguous()

    return x_pad, initial_states, seqlens, indices


def causal_conv1d_fn_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if has_initial_state is None:
        # Final-state gather in causal_conv1d_fn_native requires the mask even
        # when no sequence carries an initial state.
        has_initial_state = torch.zeros(
            cache_indices.shape[0], dtype=torch.bool, device=x.device
        )
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    assert query_start_loc[-1] <= x.shape[-1], f"{query_start_loc=}, {x.shape=}"

    x_pad, initial_state_pad, seqlens, indices = prepare_data(
        x, weight, query_start_loc, cache_indices, has_initial_state, conv_states
    )

    out, final_states_out = causal_conv1d_fn_native(
        x_pad,
        weight,
        bias,
        seqlens=seqlens,
        has_initial_state=has_initial_state,
        initial_states=initial_state_pad,
        activation=activation,
        return_final_states=True,
    )
    # Skip pad slots: pad_slot_id=-1 would wrap around via negative indexing
    # in index_copy_ and silently corrupt the last pool slot.
    valid = cache_indices != pad_slot_id
    if valid.all():
        conv_states.index_copy_(0, cache_indices, final_states_out)
    else:
        conv_states.index_copy_(0, cache_indices[valid], final_states_out[valid])

    if x.ndim == 3:
        return out  # [batch_size, dim, seq_len]

    out = out.transpose(1, 2).contiguous().view(out.size(0) * out.size(2), out.size(1))
    out_final = torch.index_select(out, 0, indices).transpose(0, 1).contiguous()

    pad_seq_len = x.size(-1) - out_final.size(-1)
    if pad_seq_len > 0:
        out_final = F.pad(out_final, (0, pad_seq_len))

    return out_final  # [dim, cu_seq_len]


@triton.jit()
def _causal_conv1d_update_kernel_no_cache_len_no_mtp(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    out_ptr,
    pad_slot_id,
    batch: tl.constexpr,
    dim: tl.constexpr,
    align_val: tl.constexpr,
    state_len: tl.constexpr,
    seq_len: tl.constexpr,
    width: tl.constexpr,
    out_len: tl.constexpr,
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
):
    pid = tl.program_id(0)
    cat_len: tl.constexpr = state_len + seq_len
    sub_state_len: tl.constexpr = state_len - seq_len
    sub_align_dim: tl.constexpr = DIM_BLOCK // align_val

    conv_begin: tl.constexpr = (cat_len - width + 1) - seq_len

    if IS_CONTINUOUS_BATCHING:
        conv_batch_offs = tl.load(conv_state_indices_ptr + pid)
    else:
        conv_batch_offs = pid

    if USE_PAD_SLOT:
        if conv_batch_offs == pad_slot_id:
            # skip padding
            return

    for doffs in range(0, dim, DIM_BLOCK):

        conv_state = tl.load(
            conv_state_ptr
            + conv_batch_offs * conv_batch_stride
            + doffs * state_len
            + tl.arange(0, DIM_BLOCK * state_len)
        )
        conv_state_T = (
            conv_state.reshape(sub_align_dim, align_val * state_len)
            .trans()
            .reshape(align_val, state_len * sub_align_dim)
            .trans()
            .reshape(
                state_len * DIM_BLOCK,
            )
        )

        x = tl.load(
            x_ptr
            + pid * x_batch_stride
            + doffs * seq_len
            + tl.arange(0, DIM_BLOCK * seq_len)
        )
        x_T = (
            x.reshape(sub_align_dim, align_val * seq_len)
            .trans()
            .reshape(align_val, seq_len * sub_align_dim)
            .trans()
            .reshape(
                seq_len * DIM_BLOCK,
            )
        )

        x_new_T = tl.full([cat_len * DIM_BLOCK], 0, x_ptr.dtype.element_ty)
        x_new_T = al.insert_slice(
            x_new_T,
            conv_state_T,
            offsets=(0,),
            sizes=(state_len * DIM_BLOCK,),
            strides=(1,),
        )  # [cat_len , DIM_BLOCK].view(-1)
        x_new_T = al.insert_slice(
            x_new_T,
            x_T,
            offsets=(state_len * DIM_BLOCK,),
            sizes=(seq_len * DIM_BLOCK,),
            strides=(1,),
        )

        new_conv_state_T = al.extract_slice(
            x_new_T, (seq_len * DIM_BLOCK,), (state_len * DIM_BLOCK,), (1,)
        )  # [state_len, DIM_BLOCK].view(-1)
        new_conv_state = (
            new_conv_state_T.reshape(state_len * align_val, sub_align_dim)
            .trans()
            .reshape(sub_align_dim * state_len, align_val)
            .trans()
            .reshape(
                DIM_BLOCK * state_len,
            )
        )  # [DIM_BLOCK, state_len].view(-1)
        tl.store(
            conv_state_ptr
            + conv_batch_offs * conv_batch_stride
            + doffs * state_len
            + tl.arange(0, DIM_BLOCK * state_len),
            new_conv_state,
        )

        weight = tl.load(weight_ptr + doffs * width + tl.arange(0, DIM_BLOCK * width))
        weight_T = (
            weight.reshape(sub_align_dim, align_val * width)
            .trans()
            .reshape(align_val, width * sub_align_dim)
            .trans()
            .reshape(
                width * DIM_BLOCK,
            )
        )  # [width, DIM_BLOCK].view(-1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + doffs + tl.arange(0, DIM_BLOCK))
        else:
            bias = 0

        if width == cat_len:
            result = (
                tl.sum((x_new_T.to(tl.float32) * weight_T).reshape(width, DIM_BLOCK), 0)
                + bias
            )
            if SILU_ACTIVATION:
                result = result / (1 + tl.exp(-result))
            tl.store(
                out_ptr
                + pid * out_batch_stride
                + (doffs + tl.arange(0, DIM_BLOCK)) * out_len,
                result,
            )
        else:
            for i in range(seq_len):
                x_conv_part = al.extract_slice(
                    x_new_T, ((conv_begin + i) * DIM_BLOCK), (width * DIM_BLOCK), (1,)
                ).to(tl.float32)
                result = (
                    tl.sum((x_conv_part * weight_T).reshape(width, DIM_BLOCK), 0) + bias
                )
                if SILU_ACTIVATION:
                    result = result / (1 + tl.exp(-result))
                tl.store(
                    out_ptr
                    + pid * out_batch_stride
                    + (doffs + tl.arange(0, DIM_BLOCK)) * out_len,
                    result,
                )


@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        # mask_1d = (idx_token < seqlen) & (
        #     idx_feats < dim
        # )  # token-index  # feature-index
        maskL = idx_feats < dim
        maskR = tl.full(maskL.shape, False, tl.int1)
        mask_1d = tl.where(idx_token < seqlen, maskL, maskR)

        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [seq(cache line), step, dim, win(K-1)]
            base_ptr = (
                intermediate_conv_window_ptr
                + conv_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            if KERNEL_WIDTH >= 2:
                tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
            if KERNEL_WIDTH >= 3:
                tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
            if KERNEL_WIDTH >= 4:
                tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)


def torch_causal_conv1d_update_npu(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_update: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
):
    bsz, hidden_size, seq_len = hidden_state.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_state], dim=-1).to(weight.dtype)
    if conv_state_update is not None:
        for i in range(seq_len):
            end = i - seq_len + 1
            start = end - state_len
            slice_range = slice(start, end if end != 0 else None)
            conv_state_update[:, i] = hidden_states_new[:, :, slice_range]
    else:
        conv_state_update = hidden_states_new[:, :, -state_len:]

    out = torch.sum(
        hidden_states_new.to(torch.float32) * weight.to(torch.float32),
        dim=-1,
        keepdim=True,
    )
    if bias is not None:
        out = out + bias.to(torch.float32)[None, :, None]
    out = out.to(hidden_state.dtype)
    if activation in ["silu", "swish"]:
        out = F.silu(out)
    out = out.to(hidden_state.dtype)
    return out, conv_state_update


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    if cache_seqlens is None and num_accepted_tokens is None:
        conv_state_update = conv_state[conv_state_indices]
        out, conv_state[conv_state_indices] = torch_causal_conv1d_update_npu(
            x,
            conv_state_update,
            weight,
            bias=bias,
            activation=activation,
        )
    else:
        _causal_conv1d_update_kernel[grid](
            # Pointers to matrices
            x,
            weight,
            bias,
            conv_state,
            cache_seqlens,
            conv_state_indices,
            num_accepted_tokens,
            intermediate_conv_window if intermediate_conv_window is not None else x,
            out,
            # Matrix dimensions
            batch,
            dim,
            seqlen,
            state_len,
            num_cache_lines,
            # stride
            stride_x_seq,
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_state_indices,
            stride_inter_seq,
            stride_inter_step,
            stride_inter_dim,
            stride_inter_win,
            stride_o_seq,
            stride_o_dim,
            stride_o_token,
            # others
            pad_slot_id,
            # META
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
            IS_SPEC_DECODING=num_accepted_tokens is not None,
            NP2_STATELEN=np2_statelen,
            USE_PAD_SLOT=pad_slot_id is not None,
            BLOCK_N=128,
            SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        )
    if unsqueeze:
        out = out.squeeze(-1)
    return out


# ---------------------------------------------------------------------------
# Window-major pool ops for short-conv hybrid models (LFM2 / LFM2-MoE).
#
# The NPU MambaPool allocates conv states as [layers, pool, window, channels]
# (sglang's _init_npu_conv_state), so the ops below take a per-layer pool view
# of (slots, state_len, dim) and normalize channel-major views transparently.
# Prefill is plain CANN torch ops (the older causal_conv1d_fn_npu carries a
# seqlens.max() sync point and -1-wrapping pad writes through index_copy_);
# decode dispatches to the causal_conv1d_update_v2 Triton kernel, which skips
# pad slots internally and is therefore CUDA-graph-capture safe.
# ---------------------------------------------------------------------------


def _conv_state_pool_view(conv_states: torch.Tensor, dim: int, state_len: int):
    """Normalize a conv-state pool view to (n, dim, state_len).

    The NPU MambaPool allocates conv states as [slots, window, channels]
    (_init_npu_conv_state), while the computation below follows the CUDA-side
    convention [slots, channels, window]. Returns the normalized view plus
    whether the caller must transpose writes back.
    """
    if conv_states.shape[-1] == state_len and conv_states.shape[-2] == dim:
        return conv_states, False
    assert conv_states.shape[-2] == state_len and conv_states.shape[-1] == dim, (
        f"conv-state pool {tuple(conv_states.shape)} matches neither "
        f"(slots, {dim}, {state_len}) nor (slots, {state_len}, {dim})"
    )
    return conv_states.transpose(1, 2), True


def causal_conv1d_fn_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    activation: Optional[str],
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Varlen prefill (packed layout) causal conv1d on NPU.

    x: (dim, cu_seq); conv_states: per-layer pool view
    (n_slots, dim, state_len) or (n_slots, state_len, dim), updated in-place;
    returns (dim, cu_seq).
    """
    assert x.dim() == 2, (
        "NPU causal_conv1d prefill requires the packed varlen "
        "(dim, cu_seq) layout with query_start_loc"
    )
    dim, _ = x.shape
    width = weight.shape[1]
    pool = conv_states
    conv_states, transposed = _conv_state_pool_view(conv_states, dim, width - 1)
    state_len = conv_states.shape[-1]
    dev = x.device
    seq_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
    batch = seq_lens.numel()
    max_t = int(seq_lens.max().item())
    # Slot 0 is reserved as the dummy write target for padded tokens (see
    # sglang's MambaSlotAllocator.clear), so remapping pads there can never
    # clobber a real request's state.
    idx = cache_indices.to(torch.int64)
    idx = torch.where(idx == pad_slot_id, torch.zeros_like(idx), idx)

    # (batch, dim, state_len) initial states; zero for requests that start
    # a fresh sequence (no cached prefix).
    init = x.new_zeros(batch, dim, state_len)
    if has_initial_state is not None and bool(has_initial_state.any()):
        init = conv_states.index_select(0, idx).to(x.dtype)
        init = init * has_initial_state[:, None, None].to(x.dtype)

    # Scatter the packed varlen input into a (batch, dim, max_t) buffer.
    tok = torch.arange(max_t, device=dev)
    valid = tok.unsqueeze(0) < seq_lens.unsqueeze(1)
    flat_pos = (
        torch.arange(batch, device=dev, dtype=torch.int64).unsqueeze(1) * max_t
        + tok
    )[valid]
    x_pad = x.new_zeros(dim, batch * max_t)
    x_pad.index_copy_(1, flat_pos, x)
    x_cat = torch.cat(
        [init, x_pad.view(dim, batch, max_t).transpose(0, 1)], dim=-1
    )  # (batch, dim, state_len + max_t)

    out = F.conv1d(x_cat, weight.unsqueeze(1), bias, groups=dim)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    out = (
        out.transpose(1, 2)
        .reshape(batch * max_t, dim)
        .index_select(0, flat_pos)
        .transpose(0, 1)
    )

    # In-place state update: a rolling window over [init, tokens]. Column c
    # holds token start_b + (c - (state_len - L_b)) when that index is valid
    # (the sequence is long enough), otherwise the matching init column --
    # the same left-truncation the CUDA kernel applies to short sequences
    # (zero init for fresh requests). Right-padded windows would instead
    # write [token, 0, 0] for short seqs, so gather the tail directly.
    starts = query_start_loc[:-1].to(torch.int64)
    col = torch.arange(state_len, device=dev).unsqueeze(0) - (
        state_len - seq_lens
    ).unsqueeze(1)  # (batch, state_len); < 0 -> take from init
    from_x = col >= 0
    # Clamp the gather range: a trailing zero-length row would index one
    # past the end (its columns are zeroed by from_x below anyway).
    gather_idx = (starts.unsqueeze(1) + col.clamp(min=0)).clamp(max=x.size(1) - 1)
    tok = x.index_select(1, gather_idx.reshape(-1))
    tok = tok.view(dim, batch, state_len).transpose(0, 1)
    tok = tok * from_x[:, None, :].to(x.dtype)
    # Columns taken from init must be the OLDEST ones that survive the
    # roll: init[c + L], not init[c] (e.g. L=1 turns [s0,s1,s2] into
    # [s1,s2,token]).
    base_idx = (
        torch.arange(state_len, device=dev).unsqueeze(0) + seq_lens.unsqueeze(1)
    ).clamp(max=state_len - 1)
    base = init.gather(2, base_idx.unsqueeze(1).expand(batch, dim, state_len))
    base = base * (~from_x)[:, None, :].to(x.dtype)
    new_states = base + tok
    if transposed:
        # Write through the original (state_len, dim) pool allocation:
        # aclnnInplaceIndexCopy rejects both a non-contiguous source and a
        # non-contiguous self (the normalized view is a transpose).
        new_states = new_states.transpose(1, 2).contiguous()
        pool.index_copy_(0, idx, new_states)
    else:
        conv_states.index_copy_(0, idx, new_states)
    return out


def causal_conv1d_update_npu_v2(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Decode causal conv1d state update on NPU (window-major pool).

    x: (batch, dim) or (batch, dim, seqlen); conv_state: per-layer pool view
    (n_slots, state_len, dim) -- the NPU MambaPool's unified
    [layers, pool, window, channels]. Dispatches to the
    causal_conv1d_update_v2 Triton kernel, which requires the window-major
    shape (a strided transpose view overflows the NPU unified buffer at
    kernel-compile time), so channel-major shaped pools
    (n_slots, dim, state_len) and pool-less callers
    (conv_state_indices is None, plain per-batch states) fall back to plain
    torch ops. Padded rows (index == pad_slot_id) are skipped inside the
    kernel, so no index remapping is needed and the call is graph-capture
    safe.
    """
    if conv_state_indices is None:
        # The Triton kernel always dereferences the slot-index tensor.
        return _causal_conv1d_update_torch(
            x, conv_state, weight, bias, activation, None, pad_slot_id
        )
    if conv_state.shape[-1] != x.shape[1]:
        # Channel-major shaped pool: torch fallback (it normalizes the
        # layout itself and remaps pads to the reserved slot 0).
        return _causal_conv1d_update_torch(
            x, conv_state, weight, bias, activation, conv_state_indices, pad_slot_id
        )
    squeeze = x.dim() == 2
    if squeeze:
        # (batch, dim): the v2 wrapper unsqueezes to (batch, 1, dim)
        # itself, so no copy is needed on the common decode step.
        x_in = x
    else:
        # Kernel consumes (batch, seqlen, dim); callers hold (b, dim, s).
        x_in = x.transpose(1, 2).contiguous()
    w = weight.t().contiguous()  # (dim, width) -> (width, dim)
    return _update_via_v2_kernel(
        x_in, conv_state, w, bias, activation, conv_state_indices, pad_slot_id, squeeze
    )


def _update_via_v2_kernel(
    x_in: torch.Tensor,
    conv_state: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    squeeze: bool,
) -> torch.Tensor:
    out = causal_conv1d_update_v2(
        x_in,
        conv_state,
        weight_t,
        bias,
        "silu" if activation in ("silu", "swish") else None,
        conv_state_indices=(
            conv_state_indices
            if conv_state_indices.dtype == torch.int32
            else conv_state_indices.to(torch.int32)
        )
        if conv_state_indices is not None
        else None,
        pad_slot_id=pad_slot_id,
    )
    if squeeze:
        return out
    return out.transpose(1, 2)


def _causal_conv1d_update_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
) -> torch.Tensor:
    """Pure-torch fallback for causal_conv1d_update_npu_v2.

    Handles both pool layouts; pads are remapped to slot 0 (reserved as the
    dummy write target for pads, see sglang's MambaSlotAllocator.clear) so
    the scatter stays graph-capture safe.
    """
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(-1)
    seqlen = x.shape[-1]
    dim = x.shape[1]
    pool = conv_state
    conv_state, transposed = _conv_state_pool_view(conv_state, dim, weight.shape[1] - 1)
    state_len = conv_state.shape[-1]

    idx = None
    valid = None
    if conv_state_indices is not None:
        idx = conv_state_indices.to(torch.int64)
        valid = idx != pad_slot_id
        # Gather pad rows from slot 0 (their outputs are unused) so the
        # PAD_SLOT_ID (-1) index never reaches the NPU gather op.
        state = conv_state.index_select(
            0, torch.where(valid, idx, torch.zeros_like(idx))
        )
    else:
        state = conv_state
    hidden = torch.cat([state, x], dim=-1)  # (n, dim, state_len + seqlen)

    out = F.conv1d(hidden, weight.unsqueeze(1), bias, groups=hidden.shape[1])
    out = out[..., :seqlen]
    if activation in ("silu", "swish"):
        out = F.silu(out)

    new_state = hidden[:, :, -state_len:]
    if transposed:
        # Write through the original (state_len, dim) pool allocation:
        # aclnnInplaceIndexCopy / copy_ reject non-contiguous self/source.
        new_state = new_state.transpose(1, 2).contiguous()
        target = pool
    else:
        target = conv_state
    if idx is None:
        target.copy_(new_state)
    else:
        safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
        target.index_copy_(0, safe_idx, new_state)

    if squeeze:
        out = out.squeeze(-1)
    return out

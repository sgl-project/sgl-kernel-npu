# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def _is_arch35():
    """A5 (Ascend950) uses the request-bank cycle layout; A3 uses the explicit
    per-token table. The two cache_mode=2 ABIs differ (see sglang's
    ascend_dsv4_backend)."""
    name = torch_npu.npu.get_device_name(0).lower()
    return "950" in name


def _softmax_columns(z):
    z_max = np.max(z, axis=0, keepdims=True)
    z_stable = z - z_max
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def _rms_norm(x, weight, eps):
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    x = x * np.reciprocal(np.sqrt(var + eps))
    return weight * x


def _rotary_emb(x, rope_sin, rope_cos, rotary_mode):
    sc = x.shape[0]
    rope_head_dim = x.shape[-1]
    rope_sin = rope_sin.reshape(sc, rope_head_dim)
    rope_cos = rope_cos.reshape(sc, rope_head_dim)
    y = np.zeros(shape=x.shape, dtype=x.dtype)
    group = rope_head_dim // 2
    for s in range(sc):
        for i in range(group):
            if rotary_mode == 1:
                a = x[s][i]
                b = x[s][i + group]
                y[s][i] = a * rope_cos[s][i] - b * rope_sin[s][i]
                y[s][i + group] = (
                    a * rope_sin[s][i + group] + b * rope_cos[s][i + group]
                )
            if rotary_mode == 2:
                idx = 2 * i
                a = x[s][idx]
                b = x[s][idx + 1]
                y[s][idx] = a * rope_cos[s][idx] - b * rope_sin[s][idx]
                y[s][idx + 1] = a * rope_sin[s][idx + 1] + b * rope_cos[s][idx + 1]
    return y


def _build_explicit_state_loc_table(
    start_pos, capacities, block_size, coff, cmp_ratio, banks_per_batch=2
):
    history_size = coff * cmp_ratio
    max_capacity = max(max(capacities, default=0), 1)
    bank_ids = torch.arange(len(start_pos) * banks_per_batch, dtype=torch.int32).view(
        len(start_pos), banks_per_batch
    )
    dummy_bank = int(bank_ids.max().item()) + 1 if bank_ids.numel() else 0
    dummy_loc = dummy_bank * block_size
    table = torch.full(
        (len(start_pos), history_size + max_capacity), dummy_loc, dtype=torch.int32
    )
    for batch_idx, (batch_start, capacity) in enumerate(zip(start_pos, capacities)):
        for column in range(history_size + capacity):
            position = batch_start - history_size + column
            if position < 0:
                continue
            bank = int(bank_ids[batch_idx, (position // block_size) % banks_per_batch])
            table[batch_idx, column] = bank * block_size + position % block_size
    return table, dummy_bank + 1, dummy_loc


def _explicit_state_loc(block_table, b_idx, seq_idx, batch_start_pos, history_size):
    table_column = history_size + seq_idx - batch_start_pos
    return int(block_table[b_idx, table_column])


def _read_state_page_cache(
    state,
    b_idx,
    start_seq_idx,
    end_seq_idx,
    block_table,
    d_start,
    d_end,
    cache_mode=1,
    batch_start_pos=0,
    history_size=0,
):
    result = np.zeros(
        shape=(end_seq_idx - start_seq_idx, d_end - d_start), dtype=np.float32
    )
    block_size = state.shape[1]
    seq_cnt = end_seq_idx - start_seq_idx
    if cache_mode == 2:
        state_flat = state.reshape(-1, state.shape[-1])
        for offset in range(seq_cnt):
            if _is_arch35():
                # A5 request-bank: one bank id per request, in-bank ring offset
                # derived from the seq position (matches the arch35 kernel).
                state_loc = int(block_table[b_idx]) * state.shape[1] + (
                    (start_seq_idx + offset) % state.shape[1]
                )
            else:
                state_loc = _explicit_state_loc(
                    block_table,
                    b_idx,
                    start_seq_idx + offset,
                    batch_start_pos,
                    history_size,
                )
            result[offset] = state_flat[state_loc, d_start:d_end]
        return result
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        block_id = block_table[b_idx][cur_seq_id // block_size]
        block_start_seq_id = cur_seq_id % block_size
        can_read_seq_cnt = block_size - block_start_seq_id
        if can_read_seq_cnt > seq_cnt - finish_cnt:
            can_read_seq_cnt = seq_cnt - finish_cnt
        result[finish_cnt : (finish_cnt + can_read_seq_cnt), :] = state[
            block_id : (block_id + 1),
            block_start_seq_id : (block_start_seq_id + can_read_seq_cnt),
            d_start:d_end,
        ]
        finish_cnt = finish_cnt + can_read_seq_cnt
    return result


def _write_state_page_cache(
    state,
    update_position,
    sc_new_state,
    b_idx,
    start_seq_idx,
    end_seq_idx,
    block_table,
    cache_mode=1,
    batch_start_pos=0,
    history_size=0,
):
    block_size = state.shape[1]
    seq_cnt = end_seq_idx - start_seq_idx
    if cache_mode == 2:
        state_flat = state.reshape(-1, state.shape[-1])
        update_flat = update_position.reshape(-1, update_position.shape[-1])
        for offset in range(seq_cnt):
            if _is_arch35():
                state_loc = int(block_table[b_idx]) * state.shape[1] + (
                    (start_seq_idx + offset) % state.shape[1]
                )
            else:
                state_loc = _explicit_state_loc(
                    block_table,
                    b_idx,
                    start_seq_idx + offset,
                    batch_start_pos,
                    history_size,
                )
            state_flat[state_loc] = sc_new_state[offset]
            update_flat[state_loc] = True
        return
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        block_id = block_table[b_idx][cur_seq_id // block_size]
        block_start_seq_id = cur_seq_id % block_size
        can_write_seq_cnt = block_size - block_start_seq_id
        if can_write_seq_cnt > seq_cnt - finish_cnt:
            can_write_seq_cnt = seq_cnt - finish_cnt
        if block_id != 0:
            state[
                block_id : (block_id + 1),
                block_start_seq_id : (block_start_seq_id + can_write_seq_cnt),
                :,
            ] = sc_new_state[finish_cnt : (finish_cnt + can_write_seq_cnt), :]
            update_position[
                block_id : (block_id + 1),
                block_start_seq_id : (block_start_seq_id + can_write_seq_cnt),
                :,
            ] = True
        finish_cnt = finish_cnt + can_write_seq_cnt


def _reference_compressor(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    update_kv,
    update_score,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    block_table,
    cu_seqlens,
    seqused,
    start_pos,
    rope_head_dim,
    cmp_ratio,
    coff,
    norm_eps,
    rotary_mode,
    cache_mode,
):
    x_dtype = x.dtype
    x = x.to(torch.float32).numpy()
    wkv = wkv.to(torch.float32).numpy()
    wgate = wgate.to(torch.float32).numpy()
    kv_state_torch = kv_state
    score_state_torch = score_state
    kv_state = kv_state.numpy()
    score_state = score_state.numpy()
    ape = ape.numpy()
    norm_weight = norm_weight.to(torch.float32).numpy()
    rope_sin = rope_sin.to(torch.float32).numpy()
    rope_cos = rope_cos.to(torch.float32).numpy()
    matmul_dtype = np.float32

    new_kv_state = np.matmul(x, wkv.T, dtype=matmul_dtype)
    new_score_state = np.matmul(x, wgate.T, dtype=matmul_dtype)

    B = len(start_pos)
    head_dim = wkv.shape[0] // coff
    bs_combine_flag = cu_seqlens is not None

    if not bs_combine_flag:
        S = x.shape[1]
        new_kv_state = new_kv_state.reshape(B * S, new_kv_state.shape[-1])
        new_score_state = new_score_state.reshape(B * S, new_score_state.shape[-1])
        cmp_kv = np.zeros(
            shape=(B, (S + cmp_ratio - 1) // cmp_ratio, head_dim), dtype=matmul_dtype
        )
    else:
        cmp_kv = np.zeros(
            shape=(min(x.shape[0], x.shape[0] // cmp_ratio + B), head_dim),
            dtype=matmul_dtype,
        )

    cmp_kv_mask = np.zeros_like(cmp_kv, dtype=bool)

    out_sum_sc_cnt = 0
    for b_idx in range(B):
        batch_out_sc_id = 0
        batch_start_pos = start_pos[b_idx]
        if seqused is not None:
            batch_seq_used = seqused[b_idx]
        else:
            batch_seq_used = (
                cu_seqlens[b_idx + 1] - cu_seqlens[b_idx]
                if bs_combine_flag
                else x.shape[1]
            )
        compress_seq_id = (batch_start_pos + batch_seq_used) // cmp_ratio * cmp_ratio

        batch_seq_idx = 0
        while batch_seq_idx < batch_seq_used:
            start_seq_idx = batch_start_pos + batch_seq_idx
            end_seq_idx = start_seq_idx // cmp_ratio * cmp_ratio + cmp_ratio
            if end_seq_idx > batch_start_pos + batch_seq_used:
                end_seq_idx = batch_start_pos + batch_seq_used

            base_offset = cu_seqlens[b_idx] if bs_combine_flag else b_idx * x.shape[1]
            start_offset = base_offset + (start_seq_idx - batch_start_pos)
            end_offset = base_offset + (end_seq_idx - batch_start_pos)

            start_seq_id_in_sc = start_seq_idx % cmp_ratio
            end_seq_idx_in_sc = start_seq_id_in_sc + (end_seq_idx - start_seq_idx)
            new_score_state[start_offset:end_offset, :] = np.add(
                new_score_state[start_offset:end_offset, :],
                ape[start_seq_id_in_sc:end_seq_idx_in_sc, :],
            )
            save_flag = (
                True
                if cache_mode == 1
                else (start_seq_idx >= (compress_seq_id - (coff - 1) * cmp_ratio))
            )
            compress_flag = start_seq_idx < compress_seq_id

            if save_flag:
                tmp_kv = new_kv_state[start_offset:end_offset, :]
                tmp_sc = new_score_state[start_offset:end_offset, :]
                _write_state_page_cache(
                    kv_state,
                    update_kv,
                    tmp_kv,
                    b_idx,
                    start_seq_idx,
                    end_seq_idx,
                    block_table,
                    cache_mode=cache_mode,
                    batch_start_pos=batch_start_pos,
                    history_size=coff * cmp_ratio,
                )
                _write_state_page_cache(
                    score_state,
                    update_score,
                    tmp_sc,
                    b_idx,
                    start_seq_idx,
                    end_seq_idx,
                    block_table,
                    cache_mode=cache_mode,
                    batch_start_pos=batch_start_pos,
                    history_size=coff * cmp_ratio,
                )

            if compress_flag:
                sc_kv_state = np.zeros(
                    shape=(coff, cmp_ratio, head_dim), dtype=matmul_dtype
                )
                sc_score_state = np.full(
                    shape=(coff, cmp_ratio, head_dim),
                    fill_value=-float("inf"),
                    dtype=matmul_dtype,
                )
                coff_id = coff - 1
                d_start = coff_id * head_dim
                d_end = (coff_id + 1) * head_dim
                cnt_from_state = 0
                if batch_start_pos == start_seq_idx:
                    cnt_from_state = batch_start_pos % cmp_ratio
                    if cnt_from_state > 0:
                        copy_start = batch_start_pos - cnt_from_state
                        copy_end = batch_start_pos
                        sc_kv_state[coff_id, 0:cnt_from_state, :] = (
                            _read_state_page_cache(
                                kv_state,
                                b_idx,
                                copy_start,
                                copy_end,
                                block_table,
                                d_start,
                                d_end,
                                cache_mode=cache_mode,
                                batch_start_pos=batch_start_pos,
                                history_size=coff * cmp_ratio,
                            )
                        )
                        sc_score_state[coff_id, 0:cnt_from_state, :] = (
                            _read_state_page_cache(
                                score_state,
                                b_idx,
                                copy_start,
                                copy_end,
                                block_table,
                                d_start,
                                d_end,
                                cache_mode=cache_mode,
                                batch_start_pos=batch_start_pos,
                                history_size=coff * cmp_ratio,
                            )
                        )
                sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = new_kv_state[
                    start_offset:end_offset, d_start:d_end
                ]
                sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = new_score_state[
                    start_offset:end_offset, d_start:d_end
                ]

                if coff == 2:
                    coff_id = 0
                    d_start = coff_id * head_dim
                    d_end = (coff_id + 1) * head_dim
                    cnt_from_state = 0
                    if batch_start_pos == start_seq_idx:
                        cnt_from_state = cmp_ratio
                        if batch_start_pos >= cmp_ratio:
                            copy_start = (
                                batch_start_pos
                                - batch_start_pos % cmp_ratio
                                - cmp_ratio
                            )
                            copy_end = copy_start + cnt_from_state
                            sc_kv_state[coff_id, 0:cnt_from_state, :] = (
                                _read_state_page_cache(
                                    kv_state,
                                    b_idx,
                                    copy_start,
                                    copy_end,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                    batch_start_pos=batch_start_pos,
                                    history_size=coff * cmp_ratio,
                                )
                            )
                            sc_score_state[coff_id, 0:cnt_from_state, :] = (
                                _read_state_page_cache(
                                    score_state,
                                    b_idx,
                                    copy_start,
                                    copy_end,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                    batch_start_pos=batch_start_pos,
                                    history_size=coff * cmp_ratio,
                                )
                            )
                    elif start_seq_idx - cmp_ratio < batch_start_pos:
                        cnt_from_state = batch_start_pos % cmp_ratio
                        if cnt_from_state > 0:
                            copy_start = batch_start_pos - batch_start_pos % cmp_ratio
                            copy_end = batch_start_pos
                            sc_kv_state[coff_id, 0:cnt_from_state, :] = (
                                _read_state_page_cache(
                                    kv_state,
                                    b_idx,
                                    copy_start,
                                    copy_end,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                    batch_start_pos=batch_start_pos,
                                    history_size=coff * cmp_ratio,
                                )
                            )
                            sc_score_state[coff_id, 0:cnt_from_state, :] = (
                                _read_state_page_cache(
                                    score_state,
                                    b_idx,
                                    copy_start,
                                    copy_end,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                    batch_start_pos=batch_start_pos,
                                    history_size=coff * cmp_ratio,
                                )
                            )
                    if cnt_from_state < cmp_ratio:
                        pre_start = start_offset - (cmp_ratio - cnt_from_state)
                        pre_end = start_offset
                        sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = (
                            new_kv_state[pre_start:pre_end, d_start:d_end]
                        )
                        sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = (
                            new_score_state[pre_start:pre_end, d_start:d_end]
                        )

                sc_kv_state = sc_kv_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = sc_score_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = _softmax_columns(sc_score_state)
                sc_data = sc_kv_state * sc_score_state
                sc_cmp_kv = np.sum(sc_data, axis=0, keepdims=True)
                sc_cmp_kv = _rms_norm(sc_cmp_kv, norm_weight, norm_eps)
                sc_cmp_kv[:, -rope_head_dim:] = _rotary_emb(
                    sc_cmp_kv[:, -rope_head_dim:],
                    rope_sin[out_sum_sc_cnt, :],
                    rope_cos[out_sum_sc_cnt, :],
                    rotary_mode,
                )
                if bs_combine_flag:
                    cmp_kv[out_sum_sc_cnt, :] = sc_cmp_kv
                    cmp_kv_mask[out_sum_sc_cnt, :] = 1
                else:
                    cmp_kv[b_idx, batch_out_sc_id, :] = sc_cmp_kv
                    cmp_kv_mask[b_idx, batch_out_sc_id, :] = 1
                batch_out_sc_id += 1
                out_sum_sc_cnt += 1

            batch_seq_idx = end_seq_idx - batch_start_pos

    if isinstance(kv_state_torch, torch.Tensor):
        kv_state_torch.copy_(torch.from_numpy(kv_state))
    if isinstance(score_state_torch, torch.Tensor):
        score_state_torch.copy_(torch.from_numpy(score_state))
    return torch.tensor(cmp_kv).to(x_dtype), cmp_kv_mask


def _make_inputs(
    start_pos,
    seq_len,
    coff,
    cmp_ratio,
    head_dim,
    hidden,
    cache_mode,
    layout="TH",
    dtype=torch.bfloat16,
    batch=1,
    block_size=16,
    seed=20260813,
    ring_size=None,
):
    gen = torch.Generator().manual_seed(seed)
    ww = coff * head_dim
    if layout == "TH":
        x = (torch.randn(batch * seq_len, hidden, generator=gen) * 0.02).to(dtype)
        cu_seqlens = torch.arange(0, batch * seq_len + 1, seq_len, dtype=torch.int32)
    else:
        x = (torch.randn(batch, seq_len, hidden, generator=gen) * 0.02).to(dtype)
        cu_seqlens = None
    wkv = (torch.randn(ww, hidden, generator=gen) * 0.02).to(dtype)
    wgate = (torch.randn(ww, hidden, generator=gen) * 0.02).to(dtype)
    ape = torch.randn(cmp_ratio, ww, generator=gen).float() * 0.01
    norm_weight = torch.randn(head_dim, generator=gen).float() * 0.02 + 1.0
    total_tokens = batch * seq_len
    rope_rows = min(total_tokens, total_tokens // cmp_ratio + batch)
    rope_sin = torch.randn(rope_rows, 64, generator=gen).float() * 0.01
    rope_cos = (
        torch.ones(rope_rows, 64)
        + torch.randn(rope_rows, 64, generator=gen).float() * 0.01
    )

    if cache_mode == 2:
        history = coff * cmp_ratio
        state_block_size = max(block_size, history + seq_len - 1, 1)
        if ring_size is not None:
            # Real sglang A5 ring: state_block_size = ring_size (fixed, small
            # for c4 = 8). Overrides the safe max() formula so wrap-around
            # write/read overlaps can be exercised.
            state_block_size = ring_size
        if _is_arch35():
            # A5 request-bank: one bank id per request; the kernel derives the
            # in-bank ring offset from seq position itself.
            bank_ids = torch.arange(batch, dtype=torch.int32)
            block_table = bank_ids
            block_num = max(int(bank_ids.max().item()) + 1, batch)
        else:
            capacities = [seq_len] * batch
            block_table, block_num, _ = _build_explicit_state_loc_table(
                start_pos, capacities, state_block_size, coff, cmp_ratio, banks_per_batch=1
            )
    else:
        max_block = (max(start_pos) + seq_len + block_size - 1) // block_size
        block_table = torch.zeros(batch, max_block, dtype=torch.int32)
        for i in range(batch):
            block_table[i, 0] = 1
        block_num = batch * max_block
        state_block_size = block_size

    kv_state = (
        torch.randn(block_num, state_block_size, ww, generator=gen).float() * 0.01
    )
    score_state = (
        torch.randn(block_num, state_block_size, ww, generator=gen).float() * 0.01
    )
    state_cache = torch.zeros(block_num, state_block_size, 2 * ww)
    state_cache[..., :ww] = kv_state
    state_cache[..., ww:] = score_state
    seqused = [seq_len] * batch
    return dict(
        x=x,
        wkv=wkv,
        wgate=wgate,
        state_cache=state_cache,
        ape=ape,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        block_table=block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        kv_state=kv_state,
        score_state=score_state,
    )


def _run_case(p, coff, cmp_ratio, head_dim, cache_mode, dtype, rotary_mode=2):
    ww = coff * head_dim
    kv_state = p["kv_state"].clone()
    score_state = p["score_state"].clone()
    update_kv = torch.zeros_like(kv_state, dtype=torch.bool)
    update_score = torch.zeros_like(score_state, dtype=torch.bool)

    # CPU reference
    ref, ref_mask = _reference_compressor(
        p["x"],
        p["wkv"],
        p["wgate"],
        kv_state,
        score_state,
        update_kv,
        update_score,
        p["ape"],
        p["norm_weight"],
        p["rope_sin"],
        p["rope_cos"],
        block_table=p["block_table"],
        cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
        seqused=p["seqused"],
        start_pos=p["start_pos"],
        rope_head_dim=64,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )
    mask_t = torch.from_numpy(np.asarray(ref_mask))

    # NPU
    state_npu = p["state_cache"].clone().npu()
    cu_t = p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None
    npu_out = torch.ops.npu.compressor(
        p["x"].npu(),
        p["wkv"].npu(),
        p["wgate"].npu(),
        state_npu,
        p["ape"].npu(),
        p["norm_weight"].npu(),
        p["rope_sin"].npu(),
        p["rope_cos"].npu(),
        state_block_table=p["block_table"].npu(),
        cu_seqlens=cu_t,
        seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
        start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
        rope_head_dim=64,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
        state_cache_stride_dim0=0,
    )
    torch_npu.npu.synchronize()

    if mask_t.numel() == 0:
        return 0.0
    diff = (npu_out.cpu() - ref).abs()
    sel = diff[mask_t]
    return sel.max().item() if sel.numel() > 0 else 0.0


class TestCompressor(unittest.TestCase):
    def _assert_ok(self, maxdiff, tol=0.05):
        self.assertLess(maxdiff, tol)

    def test_ring_c4li(self):
        p = _make_inputs([13], 9, 2, 4, 128, 1024, 2, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 2, 4, 128, 2, torch.bfloat16))

    def test_ring_c4a(self):
        p = _make_inputs([13], 9, 2, 4, 512, 1024, 2, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 2, 4, 512, 2, torch.bfloat16))

    def test_ring_c128a_sp200(self):
        p = _make_inputs([200], 129, 1, 128, 512, 1024, 2, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 1, 128, 512, 2, torch.bfloat16))

    def test_ring_c128a_sp300(self):
        p = _make_inputs([300], 129, 1, 128, 512, 1024, 2, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 1, 128, 512, 2, torch.bfloat16))

    def test_ring_c128a_sp0(self):
        p = _make_inputs([0], 129, 1, 128, 512, 1024, 2, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 1, 128, 512, 2, torch.bfloat16))

    def test_ring_real_c4_wrap(self):
        # Real sglang A5 c4 ring_size=8. start=8, seq=8 makes write [12,16)%8
        # == read [4,8)%8 = {4,5,6,7}, so the original SaveState-before-ReadState
        # order clobbers the history before it is read.
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        p = _make_inputs([8], 8, 2, 4, 512, 1024, 2, "TH", torch.bfloat16, 1, 16, ring_size=8)
        self._assert_ok(_run_case(p, 2, 4, 512, 2, torch.bfloat16))

    def test_ring_real_c4_cross_slice(self):
        # ring_size=8 with a long seq: many slices wrap around the small ring,
        # so an earlier slice's SaveState can land on a row a later slice still
        # needs as history (cross-slice clobber). Requires the deferred
        # CommitState path to hold.
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        p = _make_inputs([8], 32, 2, 4, 512, 1024, 2, "TH", torch.bfloat16, 1, 16, ring_size=8)
        self._assert_ok(_run_case(p, 2, 4, 512, 2, torch.bfloat16))

    def test_ring_real_c4_longseq(self):
        # Long seq (128) with real ring_size=8: many blocks, exercises the
        # CommitState path under real load (mm1-reservation / cross-db width).
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        p = _make_inputs([8], 128, 2, 4, 512, 1024, 2, "TH", torch.bfloat16, 1, 16, ring_size=8)
        self._assert_ok(_run_case(p, 2, 4, 512, 2, torch.bfloat16))

    def test_ring_real_c4_multi_round(self):
        # Multi-round continuous decode: start positions advance each round,
        # kv/score state accumulates in-place on the CPU reference (torch
        # tensors handed to _reference_compressor, which writes them via
        # _write_state_page_cache) and on the NPU state_cache. Verifies the
        # exact state write-back (not just cmp_kv), mirroring cann-ops
        # RingHarness.run_rounds.
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        batch, capacity, rounds, ring_size = 2, 8, 4, 8
        p0 = _make_inputs([8, 16], capacity, coff, ratio, head_dim, hidden, 2, "TH",
                          torch.bfloat16, batch, 16, ring_size=ring_size)
        kv_state = p0["kv_state"]
        score_state = p0["score_state"]
        state_npu = p0["state_cache"].clone().npu()
        block_table = p0["block_table"].npu()
        wkv_npu = p0["wkv"].npu()
        wgate_npu = p0["wgate"].npu()
        ape_npu = p0["ape"].npu()
        norm_npu = p0["norm_weight"].npu()
        cu_t = (torch.arange(0, batch + 1, dtype=torch.int32) * capacity).npu()
        cu_list = [i * capacity for i in range(batch + 1)]
        starts = [8, 16]
        for r in range(rounds):
            p = _make_inputs(starts, capacity, coff, ratio, head_dim, hidden, 2, "TH",
                             torch.bfloat16, batch, 16, seed=3000 + r, ring_size=ring_size)
            update_kv = torch.zeros_like(kv_state, dtype=torch.bool)
            update_score = torch.zeros_like(score_state, dtype=torch.bool)
            ref, mask = _reference_compressor(
                p["x"], p0["wkv"], p0["wgate"], kv_state, score_state,
                update_kv, update_score, p0["ape"], p0["norm_weight"],
                p["rope_sin"], p["rope_cos"],
                block_table=p0["block_table"], cu_seqlens=cu_list,
                seqused=[capacity, capacity], start_pos=starts,
                rope_head_dim=64, cmp_ratio=ratio, coff=coff,
                norm_eps=1e-6, rotary_mode=2, cache_mode=2)
            out = torch.ops.npu.compressor(
                p["x"].npu(), wkv_npu, wgate_npu, state_npu, ape_npu, norm_npu,
                p["rope_sin"].npu(), p["rope_cos"].npu(),
                state_block_table=block_table, cu_seqlens=cu_t,
                seqused=torch.tensor([capacity, capacity], dtype=torch.int32).npu(),
                start_pos=torch.tensor(starts, dtype=torch.int32).npu(),
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
            torch_npu.npu.synchronize()
            mask_t = torch.from_numpy(np.asarray(mask))
            if mask_t.numel():
                diff = (out.cpu() - ref).abs()[mask_t]
                self.assertLess(diff.max().item(), 0.05, f"round {r}: output")
            expected = torch.cat([kv_state, score_state], dim=-1)
            sdiff = (state_npu.cpu() - expected).abs().max().item()
            self.assertLess(sdiff, 1e-2, f"round {r}: state write-back")
            starts = [s + capacity for s in starts]

    def test_ring_real_c4_mtp_overwrite(self):
        # MTP verify then rejected-suffix overwrite: the follow-up call writes
        # at the same absolute ring positions ((start+valid) wraps the small
        # ring), so rows touched by the verify call are re-written after it.
        # accepted prefix rows must stay consistent (mirrors cann-ops
        # RingHarness.test_bf16_ring_mtp_verify_and_rejected_suffix_overwrite).
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        capacity, ring_size = 8, 8
        p0 = _make_inputs([8], capacity, coff, ratio, head_dim, hidden, 2, "TH",
                          torch.bfloat16, 1, 16, ring_size=ring_size)
        wkv_npu = p0["wkv"].npu()
        wgate_npu = p0["wgate"].npu()
        ape_npu = p0["ape"].npu()
        norm_npu = p0["norm_weight"].npu()
        block_table = p0["block_table"].npu()
        init_kv = p0["kv_state"]
        init_score = p0["score_state"]
        init_state = p0["state_cache"].clone().npu()
        for accepted in (1, 2, 4):
            kv_state = init_kv.clone()
            score_state = init_score.clone()
            state_npu = init_state.clone()
            calls = [(8, 4), (8 + accepted, capacity - accepted)]
            for r, (start, valid) in enumerate(calls):
                p = _make_inputs([start], valid, coff, ratio, head_dim, hidden, 2, "TH",
                                 torch.bfloat16, 1, 16, seed=4000 + accepted * 10 + r,
                                 ring_size=ring_size)
                cu_t = p["cu_seqlens"].npu()
                cu_list = p["cu_seqlens"].tolist()
                update_kv = torch.zeros_like(kv_state, dtype=torch.bool)
                update_score = torch.zeros_like(score_state, dtype=torch.bool)
                ref, mask = _reference_compressor(
                    p["x"], p0["wkv"], p0["wgate"], kv_state, score_state,
                    update_kv, update_score, p0["ape"], p0["norm_weight"],
                    p["rope_sin"], p["rope_cos"],
                    block_table=p0["block_table"], cu_seqlens=cu_list,
                    seqused=[valid], start_pos=[start],
                    rope_head_dim=64, cmp_ratio=ratio, coff=coff,
                    norm_eps=1e-6, rotary_mode=2, cache_mode=2)
                out = torch.ops.npu.compressor(
                    p["x"].npu(), wkv_npu, wgate_npu, state_npu, ape_npu, norm_npu,
                    p["rope_sin"].npu(), p["rope_cos"].npu(),
                    state_block_table=block_table, cu_seqlens=cu_t,
                    seqused=torch.tensor([valid], dtype=torch.int32).npu(),
                    start_pos=torch.tensor([start], dtype=torch.int32).npu(),
                    rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                    rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
                torch_npu.npu.synchronize()
                mask_t = torch.from_numpy(np.asarray(mask))
                if mask_t.numel():
                    diff = (out.cpu() - ref).abs()[mask_t]
                    self.assertLess(diff.max().item(), 0.05, f"mtp a{accepted} r{r}: output")
                expected = torch.cat([kv_state, score_state], dim=-1)
                sdiff = (state_npu.cpu() - expected).abs().max().item()
                self.assertLess(sdiff, 1e-2, f"mtp a{accepted} r{r}: state")

    def test_ring_real_c4_batch256_multi_round(self):
        # Large-batch multi-round: batch=256 independent request-bank rings
        # (A5 arch35 gives one bank per request), rounds=2. Scales the
        # multi-round accumulation across many banks at once; with the flag
        # based handshake this also stressed the dbIdx-reuse race.
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        batch, capacity, rounds, ring_size = 256, 8, 2, 8
        p0 = _make_inputs(list(range(8, 8 + batch * capacity, capacity)), capacity,
                          coff, ratio, head_dim, hidden, 2, "TH", torch.bfloat16,
                          batch, ring_size, ring_size=ring_size)
        kv_state = p0["kv_state"]
        score_state = p0["score_state"]
        state_npu = p0["state_cache"].clone().npu()
        block_table = p0["block_table"].npu()
        wkv_npu = p0["wkv"].npu()
        wgate_npu = p0["wgate"].npu()
        ape_npu = p0["ape"].npu()
        norm_npu = p0["norm_weight"].npu()
        cu_t = (torch.arange(0, batch + 1, dtype=torch.int32) * capacity).npu()
        cu_list = [i * capacity for i in range(batch + 1)]
        starts = list(range(8, 8 + batch * capacity, capacity))
        for r in range(rounds):
            p = _make_inputs(starts, capacity, coff, ratio, head_dim, hidden, 2, "TH",
                             torch.bfloat16, batch, ring_size, seed=5000 + r, ring_size=ring_size)
            update_kv = torch.zeros_like(kv_state, dtype=torch.bool)
            update_score = torch.zeros_like(score_state, dtype=torch.bool)
            ref, mask = _reference_compressor(
                p["x"], p0["wkv"], p0["wgate"], kv_state, score_state,
                update_kv, update_score, p0["ape"], p0["norm_weight"],
                p["rope_sin"], p["rope_cos"],
                block_table=p0["block_table"], cu_seqlens=cu_list,
                seqused=[capacity] * batch, start_pos=starts,
                rope_head_dim=64, cmp_ratio=ratio, coff=coff,
                norm_eps=1e-6, rotary_mode=2, cache_mode=2)
            out = torch.ops.npu.compressor(
                p["x"].npu(), wkv_npu, wgate_npu, state_npu, ape_npu, norm_npu,
                p["rope_sin"].npu(), p["rope_cos"].npu(),
                state_block_table=block_table, cu_seqlens=cu_t,
                seqused=torch.tensor([capacity] * batch, dtype=torch.int32).npu(),
                start_pos=torch.tensor(starts, dtype=torch.int32).npu(),
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
            torch_npu.npu.synchronize()
            mask_t = torch.from_numpy(np.asarray(mask))
            expected = torch.cat([kv_state, score_state], dim=-1)
            sdiff = (state_npu.cpu() - expected).abs()
            sdiff_max = sdiff.max().item()
            if mask_t.numel():
                diff = (out.cpu() - ref).abs()[mask_t]
                dmax = diff.max().item()
                self.assertLess(dmax, 0.05, f"b256 round {r}: output")
            self.assertLess(sdiff_max, 1e-2, f"b256 round {r}: state write-back")
            starts = [s + capacity for s in starts]

    def test_c2_large_batch(self):
        # cache_mode=2 large batch without a ring_size override: on A5 this is
        # the request-bank layout, on A3 the explicit per-token layout. Large
        # loopTimes exercises dbIdx reuse (loopTimes > dbWorkspaceRatio). A5 is
        # documented to reorder same-flagId CrossCoreSetFlag calls; this test
        # also checks whether the same dbIdx-reuse race exists on A3.
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        batch = 256
        p = _make_inputs([8] * batch, 8, coff, ratio, head_dim, hidden, 2, "TH",
                         torch.bfloat16, batch, 16)
        # diagnostic: locate the bad region (output vs state write-back)
        kv = p["kv_state"].clone()
        sc = p["score_state"].clone()
        ref, mask = _reference_compressor(
            p["x"], p["wkv"], p["wgate"], kv, sc,
            torch.zeros_like(kv, dtype=torch.bool),
            torch.zeros_like(sc, dtype=torch.bool),
            p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
            block_table=p["block_table"],
            cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
            seqused=p["seqused"], start_pos=p["start_pos"],
            rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
            rotary_mode=2, cache_mode=2)
        mask_t = torch.from_numpy(np.asarray(mask))
        state_npu = p["state_cache"].clone().npu()
        out = torch.ops.npu.compressor(
            p["x"].npu(), p["wkv"].npu(), p["wgate"].npu(), state_npu,
            p["ape"].npu(), p["norm_weight"].npu(), p["rope_sin"].npu(), p["rope_cos"].npu(),
            state_block_table=p["block_table"].npu(),
            cu_seqlens=p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None,
            seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
            start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
            rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
            rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
        torch_npu.npu.synchronize()
        d = (out.cpu() - ref).abs()[mask_t]
        self._assert_ok(d.max().item() if d.numel() > 0 else 0.0)

    def test_ring_real_c4_batch64(self):
        # batch=64, ring_size=8 (real c4). loopTimes = 64*8/256 = 2, no dbIdx
        # reuse. Distinguishes "db-reuse issue" from "multi-block ring_size=8".
        if not _is_arch35():
            self.skipTest("A5 request-bank ring layout only")
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        batch = 64
        p = _make_inputs(list(range(8, 8 + batch * 8, 8)), 8, coff, ratio, head_dim, hidden, 2, "TH",
                         torch.bfloat16, batch, 8, ring_size=8)
        # diagnostic: check state write-back too (batch64 previously only checked output)
        kv = p["kv_state"].clone()
        sc = p["score_state"].clone()
        ref, mask = _reference_compressor(
            p["x"], p["wkv"], p["wgate"], kv, sc,
            torch.zeros_like(kv, dtype=torch.bool),
            torch.zeros_like(sc, dtype=torch.bool),
            p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
            block_table=p["block_table"],
            cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
            seqused=p["seqused"], start_pos=p["start_pos"],
            rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
            rotary_mode=2, cache_mode=2)
        mask_t = torch.from_numpy(np.asarray(mask))
        state_npu = p["state_cache"].clone().npu()
        out = torch.ops.npu.compressor(
            p["x"].npu(), p["wkv"].npu(), p["wgate"].npu(), state_npu,
            p["ape"].npu(), p["norm_weight"].npu(), p["rope_sin"].npu(), p["rope_cos"].npu(),
            state_block_table=p["block_table"].npu(),
            cu_seqlens=p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None,
            seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
            start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
            rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
            rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
        torch_npu.npu.synchronize()
        d = (out.cpu() - ref).abs()[mask_t]
        self._assert_ok(d.max().item() if d.numel() > 0 else 0.0)

    def test_ratio128_c2_sparse_small(self):
        # sglang DSV4 decode shape: cache_mode=2, coff=1, ratio=128 with only a
        # few rows per request (seqused << 128). Regression for the sglang
        # launch hang (ratio=128 kernel never finishes).
        if not _is_arch35():
            self.skipTest("A5 only")
        coff, ratio, head_dim, hidden = 1, 128, 128, 1024
        batch = 4
        p = _make_inputs(list(range(8, 8 + batch * 8, 8)), 8, coff, ratio, head_dim, hidden, 2, "TH",
                         torch.bfloat16, batch, 8, ring_size=8)
        kv = p["kv_state"].clone()
        sc = p["score_state"].clone()
        ref, mask = _reference_compressor(
            p["x"], p["wkv"], p["wgate"], kv, sc,
            torch.zeros_like(kv, dtype=torch.bool),
            torch.zeros_like(sc, dtype=torch.bool),
            p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
            block_table=p["block_table"],
            cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
            seqused=p["seqused"], start_pos=p["start_pos"],
            rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
            rotary_mode=2, cache_mode=2)
        mask_t = torch.from_numpy(np.asarray(mask))
        state_npu = p["state_cache"].clone().npu()
        out = torch.ops.npu.compressor(
            p["x"].npu(), p["wkv"].npu(), p["wgate"].npu(), state_npu,
            p["ape"].npu(), p["norm_weight"].npu(), p["rope_sin"].npu(), p["rope_cos"].npu(),
            state_block_table=p["block_table"].npu(),
            cu_seqlens=p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None,
            seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
            start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
            rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
            rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
        print("R128C2: launched", flush=True)
        torch_npu.npu.synchronize()
        print("R128C2: kernel returned+synced", flush=True)
        d = (out.cpu() - ref).abs()[mask_t]
        self._assert_ok(d.max().item() if d.numel() > 0 else 0.0)

    def test_fp16_c4a(self):
        p = _make_inputs([0], 8192, 2, 4, 512, 4096, 2, "TH", torch.float16, 1, 128)
        self._assert_ok(_run_case(p, 2, 4, 512, 2, torch.float16))

    def test_fp16_c128a(self):
        p = _make_inputs([0], 8192, 1, 128, 512, 4096, 2, "TH", torch.float16, 1, 128)
        self._assert_ok(_run_case(p, 1, 128, 512, 2, torch.float16))

    def test_ge_helper_tiling_upload(self):
        # Exercise the ge_helper TilingCache::CopyTo_ path (ge_helper.h) used by
        # the compressor host tiling upload, on both A3 and A5: distinct configs
        # each fill a fresh fixed tiling slot (cache miss -> aclrtMemcpyAsync
        # upload on the current stream), and repeated configs hit the cache.
        # The in-graph (capture/replay) path is covered separately by
        # test_copyto_upload_acceptance.
        # 1) distinct configs -> each is a fresh tiling cache miss (CopyTo_)
        miss_cases = [
            ([13], 9, 2, 4, 128, 1024, 2, 2, 4, 128),
            ([13], 9, 2, 4, 512, 1024, 2, 2, 4, 512),
            ([200], 129, 1, 128, 512, 1024, 2, 1, 128, 512),
        ]
        for i, (sp, seq_len, coff, ratio, head, hidden, cmode, c2, r2, h2) in enumerate(
            miss_cases
        ):
            p = _make_inputs(
                sp, seq_len, coff, ratio, head, hidden, cmode, "TH",
                torch.bfloat16, 1, 16, seed=8000 + i,
            )
            self._assert_ok(_run_case(p, c2, r2, h2, cmode, torch.bfloat16))

        # 2) same config repeatedly -> tiling cache hit (no CopyTo_)
        p = _make_inputs(
            [13], 9, 2, 4, 128, 1024, 2, "TH", torch.bfloat16, 1, 16, seed=8000
        )
        for _ in range(3):
            self._assert_ok(_run_case(p, 2, 4, 128, 2, torch.bfloat16))

    def test_copyto_async_leak(self):
        # CopyTo_ must fully upload the tiling before returning. Every other test
        # calls torch_npu.npu.synchronize() after each op, which hides any H2D that
        # CopyTo_ failed to wait on. Here several distinct-config calls (each a fresh
        # tiling cache miss -> CopyTo_) are issued back-to-back with NO interleaved
        # sync, so a leaked async H2D would corrupt the next call's tiling read.
        cases = [
            ([13], 9, 2, 4, 128, 1024, 2, 2, 4, 128),
            ([13], 9, 2, 4, 512, 1024, 2, 2, 4, 512),
            ([200], 129, 1, 128, 512, 1024, 2, 1, 128, 512),
        ]
        pending = []
        for i, (sp, seq_len, coff, ratio, head, hidden, cmode, c2, r2, h2) in enumerate(cases):
            p = _make_inputs(sp, seq_len, coff, ratio, head, hidden, cmode, "TH",
                             torch.bfloat16, 1, 16, seed=8000 + i)
            kv = p["kv_state"].clone()
            sc = p["score_state"].clone()
            ref, mask = _reference_compressor(
                p["x"], p["wkv"], p["wgate"], kv, sc,
                torch.zeros_like(kv, dtype=torch.bool),
                torch.zeros_like(sc, dtype=torch.bool),
                p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
                block_table=p["block_table"],
                cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
                seqused=p["seqused"], start_pos=p["start_pos"],
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=cmode,
            )
            mask_t = torch.from_numpy(np.asarray(mask))
            state_npu = p["state_cache"].clone().npu()
            out = torch.ops.npu.compressor(
                p["x"].npu(), p["wkv"].npu(), p["wgate"].npu(), state_npu,
                p["ape"].npu(), p["norm_weight"].npu(),
                p["rope_sin"].npu(), p["rope_cos"].npu(),
                state_block_table=p["block_table"].npu(),
                cu_seqlens=p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None,
                seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
                start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=cmode, state_cache_stride_dim0=0,
            )
            pending.append((out, ref, mask_t))
        torch_npu.npu.synchronize()
        for out, ref, mask_t in pending:
            if mask_t.numel() == 0:
                continue
            d = (out.cpu() - ref).abs()[mask_t]
            self._assert_ok(d.max().item() if d.numel() > 0 else 0.0)

    def test_copyto_upload_acceptance(self):
        # Replicates the real-inference symptom (acceptance rate collapses to 0 on
        # A3 with the raw default-stream aclrtMemcpy upload): a graph captured once
        # is replayed step-by-step like decode. Each replay reads the SAME
        # fixed-address tiling that CopyTo_ uploaded during warmup; an upload that
        # is not ordered with the kernel stream makes replays diverge from the CPU
        # reference and acceptance dies. Guards the stream-ordered aclrtMemcpyAsync
        # CopyTo_ (and would catch a regression back to the raw default-stream copy).
        p = _make_inputs([200], 129, 1, 128, 512, 1024, 2, "TH", torch.bfloat16, 1, 16, seed=12345)

        x_n = p["x"].clone().npu()
        wkv_n = p["wkv"].clone().npu()
        wgate_n = p["wgate"].clone().npu()
        ape_n = p["ape"].clone().npu()
        norm_n = p["norm_weight"].clone().npu()
        sine_n = p["rope_sin"].clone().npu()
        cose_n = p["rope_cos"].clone().npu()
        tbl_n = p["block_table"].clone().npu()
        cu_n = p["cu_seqlens"].clone().npu()
        used_n = torch.tensor(p["seqused"], dtype=torch.int32).npu()
        start_n = torch.tensor(p["start_pos"], dtype=torch.int32).npu()
        kw = dict(
            rope_head_dim=64,
            cmp_ratio=128,
            coff=1,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=2,
            state_cache_stride_dim0=0,
        )

        def _call(state_n):
            return torch.ops.npu.compressor(
                x_n, wkv_n, wgate_n, state_n, ape_n, norm_n, sine_n, cose_n,
                state_block_table=tbl_n, cu_seqlens=cu_n, seqused=used_n,
                start_pos=start_n, **kw,
            )

        state2 = p["state_cache"].clone().npu()
        _call(state2)  # eager warmup: fills the tiling cache via CopyTo_
        torch_npu.npu.synchronize()
        _call(p["state_cache"].clone().npu())  # warmup (cache hit)
        torch_npu.npu.synchronize()
        state2.copy_(p["state_cache"])
        torch_npu.npu.synchronize()

        g = torch.npu.NPUGraph()
        capture_stream = torch_npu.npu.Stream()
        with torch_npu.npu.graph(g, stream=capture_stream, auto_dispatch_capture=True):
            out_graph = _call(state2)
        torch_npu.npu.synchronize()

        # decode-style replays against fresh inputs; each must match the reference.
        # A tiling upload not ordered with the kernel stream breaks these -> FAIL.
        for step in range(10):
            gen = torch.Generator().manual_seed(2000 + step)
            x_n.copy_((torch.randn(p["x"].shape, generator=gen) * 0.02).to(p["x"].dtype))
            state2.copy_(p["state_cache"])
            torch_npu.npu.synchronize()
            g.replay()
            torch_npu.npu.synchronize()
            ref, mask2 = _reference_compressor(
                x_n.cpu(), p["wkv"], p["wgate"],
                p["kv_state"].clone(), p["score_state"].clone(),
                torch.zeros_like(p["kv_state"], dtype=torch.bool),
                torch.zeros_like(p["score_state"], dtype=torch.bool),
                p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
                block_table=p["block_table"], cu_seqlens=p["cu_seqlens"].tolist(),
                seqused=p["seqused"], start_pos=p["start_pos"],
                rope_head_dim=64, cmp_ratio=128, coff=1, norm_eps=1e-6,
                rotary_mode=2, cache_mode=2,
            )
            mask_t = torch.from_numpy(np.asarray(mask2)).bool()
            if mask_t.numel() == 0:
                continue
            og = out_graph.cpu().float()[mask_t]
            rf = torch.as_tensor(ref).float()[mask_t]
            self.assertFalse(
                torch.isnan(og).any() or torch.isnan(rf).any(),
                f"step {step}: NaN in graph replay (tiling upload broken)",
            )
            self.assertLess(
                (og - rf).abs().max().item(), 0.05,
                f"step {step}: graph replay diverged from reference (tiling upload broken)",
            )

    def test_copyto_prefill_large_batch(self):
        # A3 regression: the raw default-stream aclrtMemcpy broke prefill at large
        # batch (acceptance -> 0). The default-stream H2D races the kernel stream;
        # a heavy prefill kernel (large batch * seq -> large loopTimes) widens the
        # copy window and exposes the race. Guard the stream-ordered
        # aclrtMemcpyAsync CopyTo_: prefill with a large batch must stay correct.
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        cases = [
            ([8] * 64, 16, 64),    # tokenSize 1024, loopTimes 4
            ([8] * 128, 16, 128),  # tokenSize 2048, loopTimes 8
            ([8] * 256, 8, 256),   # tokenSize 2048, loopTimes 8
        ]
        for start_pos, seq_len, batch in cases:
            p = _make_inputs(start_pos, seq_len, coff, ratio, head_dim, hidden, 2,
                             "TH", torch.bfloat16, batch, 16, seed=41000)
            self._assert_ok(_run_case(p, coff, ratio, head_dim, 2, torch.bfloat16))

    def test_copyto_nosync_prefill_miss(self):
        # The test that DISTINGUISHES the stream-ordered aclrtMemcpyAsync CopyTo_
        # from the raw default-stream aclrtMemcpy that broke A3 prefill (acceptance
        # -> 0). Every other test calls torch_npu.npu.synchronize() after each op
        # (_run_case does), which flushes ALL streams and hides the cross-stream
        # ordering bug (default-stream upload vs kernel stream). Here several
        # large-batch prefill calls, each a fresh tiling-cache miss -> CopyTo_, are
        # issued back-to-back with NO interleaved sync, so each kernel reads the
        # tiling its CopyTo_ just uploaded on the current stream. A raw
        # default-stream upload is not ordered with the kernel stream and must
        # produce wrong output; aclrtMemcpyAsync on the current stream is ordered.
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        cases = [
            ([8] * 64, 16, 64),    # tokenSize 1024, loopTimes 4
            ([8] * 128, 16, 128),  # tokenSize 2048, loopTimes 8
            ([8] * 256, 8, 256),   # tokenSize 2048, loopTimes 8
        ]
        pending = []
        for start_pos, seq_len, batch in cases:
            p = _make_inputs(start_pos, seq_len, coff, ratio, head_dim, hidden, 2,
                             "TH", torch.bfloat16, batch, 16, seed=53000)
            kv = p["kv_state"].clone()
            sc = p["score_state"].clone()
            ref, mask = _reference_compressor(
                p["x"], p["wkv"], p["wgate"], kv, sc,
                torch.zeros_like(kv, dtype=torch.bool),
                torch.zeros_like(sc, dtype=torch.bool),
                p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
                block_table=p["block_table"],
                cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
                seqused=p["seqused"], start_pos=p["start_pos"],
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=2)
            mask_t = torch.from_numpy(np.asarray(mask))
            state_npu = p["state_cache"].clone().npu()
            out = torch.ops.npu.compressor(
                p["x"].npu(), p["wkv"].npu(), p["wgate"].npu(), state_npu,
                p["ape"].npu(), p["norm_weight"].npu(), p["rope_sin"].npu(), p["rope_cos"].npu(),
                state_block_table=p["block_table"].npu(),
                cu_seqlens=p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None,
                seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
                start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
            pending.append((out, ref, mask_t))
        torch_npu.npu.synchronize()
        for out, ref, mask_t in pending:
            if mask_t.numel() == 0:
                continue
            d = (out.cpu() - ref).abs()[mask_t]
            self._assert_ok(d.max().item() if d.numel() > 0 else 0.0)

    def test_copyto_multi_stream(self):
        # Explicitly force the multi-stream ordering the raw aclrtMemcpy (no stream)
        # gets wrong: it uploads on the DEVICE DEFAULT stream, while the compressor
        # kernel runs on the torch_npu CURRENT stream. We hold the default stream
        # busy with a long op, then drive the compressor from a NON-default current
        # stream, so the default-stream H2D and the current-stream kernel race with
        # no ordering dependency. aclrtMemcpyAsync (upload on the current stream) is
        # ordered with the kernel and must pass; the raw default-stream copy can leave
        # the kernel reading a stale/partial tiling.
        coff, ratio, head_dim, hidden = 2, 4, 512, 1024
        cases = [([8] * 64, 16, 64), ([8] * 128, 16, 128)]
        pending = []
        for start_pos, seq_len, batch in cases:
            p = _make_inputs(start_pos, seq_len, coff, ratio, head_dim, hidden, 2,
                             "TH", torch.bfloat16, batch, 16, seed=55000)
            kv = p["kv_state"].clone()
            sc = p["score_state"].clone()
            ref, mask = _reference_compressor(
                p["x"], p["wkv"], p["wgate"], kv, sc,
                torch.zeros_like(kv, dtype=torch.bool),
                torch.zeros_like(sc, dtype=torch.bool),
                p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
                block_table=p["block_table"],
                cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
                seqused=p["seqused"], start_pos=p["start_pos"],
                rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                rotary_mode=2, cache_mode=2)
            mask_t = torch.from_numpy(np.asarray(mask))
            x_n = p["x"].npu()
            wkv_n = p["wkv"].npu()
            wgate_n = p["wgate"].npu()
            ape_n = p["ape"].npu()
            norm_n = p["norm_weight"].npu()
            sine_n = p["rope_sin"].npu()
            cose_n = p["rope_cos"].npu()
            tbl_n = p["block_table"].npu()
            cu_n = p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None
            used_n = torch.tensor(p["seqused"], dtype=torch.int32).npu()
            start_n = torch.tensor(p["start_pos"], dtype=torch.int32).npu()
            state_npu = p["state_cache"].clone().npu()
            torch_npu.npu.synchronize()  # inputs stable on the default stream
            big = torch.randn(4096, 4096, device="npu")
            _ = big @ big  # long op on the default stream, host does not wait
            s = torch_npu.npu.Stream()
            with torch_npu.npu.stream(s):
                out = torch.ops.npu.compressor(
                    x_n, wkv_n, wgate_n, state_npu, ape_n, norm_n, sine_n, cose_n,
                    state_block_table=tbl_n, cu_seqlens=cu_n, seqused=used_n,
                    start_pos=start_n,
                    rope_head_dim=64, cmp_ratio=ratio, coff=coff, norm_eps=1e-6,
                    rotary_mode=2, cache_mode=2, state_cache_stride_dim0=0)
            pending.append((out, ref, mask_t))
        torch_npu.npu.synchronize()
        for out, ref, mask_t in pending:
            if mask_t.numel() == 0:
                continue
            d = (out.cpu() - ref).abs()[mask_t]
            self._assert_ok(d.max().item() if d.numel() > 0 else 0.0)

    def test_npu_graph_capture(self):
        # Graph-capture support: warmup (fills the device-resident tiling cache),
        # capture, then replay multiple times against mutated inputs to verify the
        # graph reads the *current* contents of the captured tensors.
        p = _make_inputs([200], 129, 1, 128, 512, 1024, 2, "TH", torch.bfloat16, 1, 16)

        x_n = p["x"].clone().npu()
        wkv_n = p["wkv"].clone().npu()
        wgate_n = p["wgate"].clone().npu()
        ape_n = p["ape"].clone().npu()
        norm_n = p["norm_weight"].clone().npu()
        sine_n = p["rope_sin"].clone().npu()
        cose_n = p["rope_cos"].clone().npu()
        tbl_n = p["block_table"].clone().npu()
        cu_n = p["cu_seqlens"].clone().npu()
        used_n = torch.tensor(p["seqused"], dtype=torch.int32).npu()
        start_n = torch.tensor(p["start_pos"], dtype=torch.int32).npu()
        kw = dict(
            rope_head_dim=64,
            cmp_ratio=128,
            coff=1,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=2,
            state_cache_stride_dim0=0,
        )

        def _call(state_n):
            return torch.ops.npu.compressor(
                x_n,
                wkv_n,
                wgate_n,
                state_n,
                ape_n,
                norm_n,
                sine_n,
                cose_n,
                state_block_table=tbl_n,
                cu_seqlens=cu_n,
                seqused=used_n,
                start_pos=start_n,
                **kw
            )

        # valid mask (matches eager CPU reference in _run_case)
        _, mask = _reference_compressor(
            p["x"],
            p["wkv"],
            p["wgate"],
            p["kv_state"].clone(),
            p["score_state"].clone(),
            torch.zeros_like(p["kv_state"], dtype=torch.bool),
            torch.zeros_like(p["score_state"], dtype=torch.bool),
            p["ape"],
            p["norm_weight"],
            p["rope_sin"],
            p["rope_cos"],
            block_table=p["block_table"],
            cu_seqlens=p["cu_seqlens"].tolist(),
            seqused=p["seqused"],
            start_pos=p["start_pos"],
            rope_head_dim=64,
            cmp_ratio=128,
            coff=1,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=2,
        )
        mask_t = torch.from_numpy(np.asarray(mask)).bool()
        if not mask_t.any():
            return

        # eager reference
        state2 = p["state_cache"].clone().npu()
        out_eager = _call(state2)
        torch_npu.npu.synchronize()

        # warmup: fill the tiling cache before capture (no host memcpy inside capture)
        _call(p["state_cache"].clone().npu())
        torch_npu.npu.synchronize()

        # reset state so eager and graph see identical input
        state2.copy_(p["state_cache"])
        torch_npu.npu.synchronize()

        g = torch.npu.NPUGraph()
        capture_stream = torch_npu.npu.Stream()
        with torch_npu.npu.graph(g, stream=capture_stream, auto_dispatch_capture=True):
            out_graph = _call(state2)
        torch_npu.npu.synchronize()
        g.replay()
        torch_npu.npu.synchronize()

        eg = out_eager.cpu().float()[mask_t]
        og = out_graph.cpu().float()[mask_t]
        print(
            f"[diag] eager nan={eg.isnan().sum().item()} graph nan={og.isnan().sum().item()} "
            f"eager inf={eg.isinf().sum().item()} graph inf={og.isinf().sum().item()} "
            f"eager range=[{eg.min().item():.3e},{eg.max().item():.3e}] "
            f"graph range=[{og.min().item():.3e},{og.max().item():.3e}] "
            f"mask_count={mask_t.sum().item()}"
        )
        d = ((og - eg).abs()).max().item() if not eg.isnan().any() and not og.isnan().any() else float("nan")
        self._assert_ok(d)

        # replay against mutated input: the graph must read the current x contents
        for seed, offset in ((20260813, -0.03), (20260814, 0.08)):
            gen = torch.Generator().manual_seed(seed)
            x_n.copy_(
                (torch.randn(p["x"].shape, generator=gen) * 0.02 + offset).to(
                    p["x"].dtype
                )
            )
            state2.copy_(
                p["state_cache"]
            )  # reset the in/out state so graph and ref see identical input
            torch_npu.npu.synchronize()
            g.replay()
            torch_npu.npu.synchronize()
            ref, mask2 = _reference_compressor(
                x_n.cpu(),
                p["wkv"],
                p["wgate"],
                p["kv_state"].clone(),
                p["score_state"].clone(),
                torch.zeros_like(p["kv_state"], dtype=torch.bool),
                torch.zeros_like(p["score_state"], dtype=torch.bool),
                p["ape"],
                p["norm_weight"],
                p["rope_sin"],
                p["rope_cos"],
                block_table=p["block_table"],
                cu_seqlens=p["cu_seqlens"].tolist(),
                seqused=p["seqused"],
                start_pos=p["start_pos"],
                rope_head_dim=64,
                cmp_ratio=128,
                coff=1,
                norm_eps=1e-6,
                rotary_mode=2,
                cache_mode=2,
            )
            mask2_t = torch.from_numpy(np.asarray(mask2)).bool()
            if mask2_t.any():
                d2 = (
                    (
                        (out_graph.cpu().float() - torch.as_tensor(ref).float()).abs()
                        * mask2_t
                    )
                    .max()
                    .item()
                )
                self._assert_ok(d2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

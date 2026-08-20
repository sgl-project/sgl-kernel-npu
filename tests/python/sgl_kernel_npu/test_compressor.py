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
        capacities = [seq_len] * batch
        history = coff * cmp_ratio
        state_block_size = max(block_size, history + max(capacities, default=1) - 1, 1)
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

    def test_continuous_c4li(self):
        p = _make_inputs([13], 9, 2, 4, 128, 1024, 1, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 2, 4, 128, 1, torch.bfloat16))

    def test_continuous_c4a(self):
        p = _make_inputs([13], 9, 2, 4, 512, 1024, 1, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 2, 4, 512, 1, torch.bfloat16))

    def test_continuous_c128a(self):
        p = _make_inputs([200], 129, 1, 128, 512, 1024, 1, "TH", torch.bfloat16, 1, 16)
        self._assert_ok(_run_case(p, 1, 128, 512, 1, torch.bfloat16))

    def test_prefill_8192_c4a(self):
        p = _make_inputs([0], 8192, 2, 4, 512, 4096, 1, "TH", torch.bfloat16, 1, 128)
        self._assert_ok(_run_case(p, 2, 4, 512, 1, torch.bfloat16))

    def test_prefill_8192_c128a(self):
        p = _make_inputs([0], 8192, 1, 128, 512, 4096, 1, "TH", torch.bfloat16, 1, 128)
        self._assert_ok(_run_case(p, 1, 128, 512, 1, torch.bfloat16))

    def test_decode_c4a(self):
        p = _make_inputs([8192], 1, 2, 4, 512, 4096, 1, "TH", torch.bfloat16, 1, 128)
        self._assert_ok(_run_case(p, 2, 4, 512, 1, torch.bfloat16))

    def test_decode_c128a(self):
        p = _make_inputs([8192], 1, 1, 128, 512, 4096, 1, "TH", torch.bfloat16, 1, 128)
        self._assert_ok(_run_case(p, 1, 128, 512, 1, torch.bfloat16))

    def test_fp16_c4a(self):
        p = _make_inputs([0], 8192, 2, 4, 512, 4096, 1, "TH", torch.float16, 1, 128)
        self._assert_ok(_run_case(p, 2, 4, 512, 1, torch.float16))

    def test_fp16_c128a(self):
        p = _make_inputs([0], 8192, 1, 128, 512, 4096, 1, "TH", torch.float16, 1, 128)
        self._assert_ok(_run_case(p, 1, 128, 512, 1, torch.float16))

    def test_multi_batch_continuous(self):
        p = _make_inputs([0, 128], 128, 2, 4, 512, 1024, 1, "TH", torch.bfloat16, 2, 16)
        self._assert_ok(_run_case(p, 2, 4, 512, 1, torch.bfloat16))

    def test_multi_batch_ring(self):
        p = _make_inputs(
            [200, 328], 129, 1, 128, 512, 1024, 2, "TH", torch.bfloat16, 2, 16
        )
        self._assert_ok(_run_case(p, 1, 128, 512, 2, torch.bfloat16))

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

        d = (
            ((out_graph.cpu().float() - out_eager.cpu().float()).abs() * mask_t)
            .max()
            .item()
        )
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

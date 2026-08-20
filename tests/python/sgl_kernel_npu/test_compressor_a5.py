# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

import unittest

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu


def _is_ascend950():
    try:
        name = torch.npu.get_device_name(0).replace(" ", "").lower()
        return "ascend950" in name
    except Exception:
        return False


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


def _cycle_slot(bank_id, absolute_pos, block_size):
    return int(bank_id), int(absolute_pos) % int(block_size)


def _read_cycle_state(state, bank_id, absolute_start, absolute_end):
    block_size = state.shape[1]
    return torch.stack(
        [
            state[_cycle_slot(bank_id, pos, block_size)]
            for pos in range(absolute_start, absolute_end)
        ],
        dim=0,
    )


def _write_cycle_state(state, bank_id, absolute_pos, value):
    state[_cycle_slot(bank_id, absolute_pos, state.shape[1])] = value


def _make_a5_inputs(
    start_pos,
    seq_len,
    coff,
    cmp_ratio,
    head_dim,
    hidden,
    cache_mode,
    layout,
    dtype,
    batch,
    block_size,
    noncontiguous_dim0,
    seed,
):
    assert cache_mode == 2
    gen = torch.Generator().manual_seed(seed)
    width = coff * head_dim
    if layout == "TH":
        x = (torch.randn(batch * seq_len, hidden, generator=gen) * 0.02).to(dtype)
        cu_seqlens = torch.arange(0, batch * seq_len + 1, seq_len, dtype=torch.int32)
    else:
        x = (torch.randn(batch, seq_len, hidden, generator=gen) * 0.02).to(dtype)
        cu_seqlens = None
    wkv = (torch.randn(width, hidden, generator=gen) * 0.02).to(dtype)
    wgate = (torch.randn(width, hidden, generator=gen) * 0.02).to(dtype)
    ape = torch.randn(cmp_ratio, width, generator=gen).float() * 0.01
    norm_weight = torch.randn(head_dim, generator=gen).float() * 0.02 + 1.0
    rope_rows = min(batch * seq_len, batch * seq_len // cmp_ratio + batch)
    rope_sin = torch.randn(rope_rows, 64, generator=gen).float() * 0.01
    rope_cos = (
        torch.ones(rope_rows, 64)
        + torch.randn(rope_rows, 64, generator=gen).float() * 0.01
    )
    bank_ids = torch.arange(batch, dtype=torch.int32)
    block_num = batch + 1
    if noncontiguous_dim0:
        backing = torch.empty(
            (block_num * 2, block_size, 2 * width), dtype=torch.float32
        )
        state_cache = backing[::2]
        state_cache.fill_(17.0)
        assert state_cache.stride(1) == 2 * width
        assert state_cache.stride(0) > block_size * 2 * width
    else:
        state_cache = torch.full(
            (block_num, block_size, 2 * width), 17.0, dtype=torch.float32
        )
    return dict(
        x=x,
        wkv=wkv,
        wgate=wgate,
        state_cache=state_cache,
        ape=ape,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        state_block_table=bank_ids,
        cu_seqlens=cu_seqlens,
        seqused=[seq_len] * batch,
        start_pos=start_pos,
        rope_head_dim=64,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=cache_mode,
        layout=layout,
        dtype=dtype,
    )


def _reference_compressor_a5(inputs):
    x_dtype = inputs["x"].dtype
    x = inputs["x"].float().numpy()
    wkv = inputs["wkv"].float().numpy()
    wgate = inputs["wgate"].float().numpy()
    state = inputs["state_cache"].clone()
    written = torch.zeros_like(state, dtype=torch.bool)
    ape = inputs["ape"].numpy()
    norm_weight = inputs["norm_weight"].float().numpy()
    rope_sin = inputs["rope_sin"].float().numpy()
    rope_cos = inputs["rope_cos"].float().numpy()
    start_pos, seqused = inputs["start_pos"], inputs["seqused"]
    coff, cmp_ratio, rope_head_dim = (
        inputs["coff"],
        inputs["cmp_ratio"],
        inputs["rope_head_dim"],
    )
    width, head_dim = wkv.shape[0], wkv.shape[0] // coff
    new_kv = np.matmul(x, wkv.T, dtype=np.float32)
    new_score = np.matmul(x, wgate.T, dtype=np.float32)
    combine = inputs["cu_seqlens"] is not None
    if not combine:
        new_kv = new_kv.reshape(len(start_pos) * x.shape[1], width)
        new_score = new_score.reshape(len(start_pos) * x.shape[1], width)
    if combine:
        output = np.zeros(
            (min(x.shape[0], x.shape[0] // cmp_ratio + len(start_pos)), head_dim),
            dtype=np.float32,
        )
    else:
        output = np.zeros(
            (len(start_pos), (x.shape[1] + cmp_ratio - 1) // cmp_ratio, head_dim),
            dtype=np.float32,
        )
    out_index = 0
    for batch_index, batch_start in enumerate(start_pos):
        batch_out = 0
        batch_used = seqused[batch_index]
        compress_until = (batch_start + batch_used) // cmp_ratio * cmp_ratio
        seq_index = 0
        bank_id = inputs["state_block_table"][batch_index].item()
        while seq_index < batch_used:
            absolute_start = batch_start + seq_index
            absolute_end = min(
                absolute_start // cmp_ratio * cmp_ratio + cmp_ratio,
                batch_start + batch_used,
            )
            base = (
                inputs["cu_seqlens"][batch_index].item()
                if combine
                else batch_index * x.shape[1]
            )
            start_offset, end_offset = (
                base + seq_index,
                base + absolute_end - batch_start,
            )
            within = absolute_start % cmp_ratio
            new_score[start_offset:end_offset] += ape[
                within : within + end_offset - start_offset
            ]
            if absolute_start >= compress_until - (coff - 1) * cmp_ratio:
                for offset, absolute_pos in enumerate(
                    range(absolute_start, absolute_end)
                ):
                    value = torch.from_numpy(
                        np.concatenate(
                            (
                                new_kv[start_offset + offset],
                                new_score[start_offset + offset],
                            )
                        )
                    ).float()
                    _write_cycle_state(state, bank_id, absolute_pos, value)
                    _write_cycle_state(
                        written,
                        bank_id,
                        absolute_pos,
                        torch.ones_like(value, dtype=torch.bool),
                    )
            if absolute_start < compress_until:
                kv_groups = np.zeros((coff, cmp_ratio, head_dim), dtype=np.float32)
                score_groups = np.full(
                    (coff, cmp_ratio, head_dim), -float("inf"), dtype=np.float32
                )
                coff_id = coff - 1
                d_start, d_end = coff_id * head_dim, (coff_id + 1) * head_dim
                cnt_from_state = 0
                if batch_start == absolute_start:
                    cnt_from_state = batch_start % cmp_ratio
                    if cnt_from_state > 0:
                        history = _read_cycle_state(
                            state, bank_id, batch_start - cnt_from_state, batch_start
                        )
                        kv_groups[coff_id, :cnt_from_state] = history[
                            :, d_start:d_end
                        ].numpy()
                        score_groups[coff_id, :cnt_from_state] = history[
                            :, width + d_start : width + d_end
                        ].numpy()
                kv_groups[coff_id, cnt_from_state:cmp_ratio] = new_kv[
                    start_offset:end_offset, d_start:d_end
                ]
                score_groups[coff_id, cnt_from_state:cmp_ratio] = new_score[
                    start_offset:end_offset, d_start:d_end
                ]
                if coff == 2:
                    coff_id = 0
                    d_start, d_end = 0, head_dim
                    cnt_from_state = 0
                    if batch_start == absolute_start:
                        cnt_from_state = cmp_ratio
                        if batch_start >= cmp_ratio:
                            copy_start = (
                                batch_start - batch_start % cmp_ratio - cmp_ratio
                            )
                            history = _read_cycle_state(
                                state, bank_id, copy_start, copy_start + cnt_from_state
                            )
                            kv_groups[coff_id, :cnt_from_state] = history[
                                :, d_start:d_end
                            ].numpy()
                            score_groups[coff_id, :cnt_from_state] = history[
                                :, width + d_start : width + d_end
                            ].numpy()
                    elif absolute_start - cmp_ratio < batch_start:
                        cnt_from_state = batch_start % cmp_ratio
                        if cnt_from_state > 0:
                            copy_start = batch_start - batch_start % cmp_ratio
                            history = _read_cycle_state(
                                state, bank_id, copy_start, batch_start
                            )
                            kv_groups[coff_id, :cnt_from_state] = history[
                                :, d_start:d_end
                            ].numpy()
                            score_groups[coff_id, :cnt_from_state] = history[
                                :, width + d_start : width + d_end
                            ].numpy()
                    if cnt_from_state < cmp_ratio:
                        previous = new_kv[
                            start_offset - (cmp_ratio - cnt_from_state) : start_offset,
                            d_start:d_end,
                        ]
                        previous_scores = new_score[
                            start_offset - (cmp_ratio - cnt_from_state) : start_offset,
                            d_start:d_end,
                        ]
                        kv_groups[coff_id, cnt_from_state:cmp_ratio] = previous
                        score_groups[coff_id, cnt_from_state:cmp_ratio] = (
                            previous_scores
                        )
                compressed = np.sum(
                    kv_groups.reshape(-1, head_dim)
                    * _softmax_columns(score_groups.reshape(-1, head_dim)),
                    axis=0,
                    keepdims=True,
                )
                compressed = _rms_norm(compressed, norm_weight, inputs["norm_eps"])
                compressed[:, -rope_head_dim:] = _rotary_emb(
                    compressed[:, -rope_head_dim:],
                    rope_sin[out_index],
                    rope_cos[out_index],
                    inputs["rotary_mode"],
                )
                if combine:
                    output[out_index] = compressed
                else:
                    output[batch_index, batch_out] = compressed
                batch_out += 1
                out_index += 1
            seq_index = absolute_end - batch_start
    return torch.tensor(output).to(x_dtype), state, written


def _call_npu_compressor(inputs, state_cache, rotary_mode=2, state_cache_stride_dim0=0):
    return torch.ops.npu.compressor(
        inputs["x"].npu(),
        inputs["wkv"].npu(),
        inputs["wgate"].npu(),
        state_cache,
        inputs["ape"].npu(),
        inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(),
        inputs["rope_cos"].npu(),
        state_block_table=inputs["state_block_table"].npu(),
        cu_seqlens=(
            inputs["cu_seqlens"].npu() if inputs["cu_seqlens"] is not None else None
        ),
        seqused=torch.tensor(inputs["seqused"], dtype=torch.int32).npu(),
        start_pos=torch.tensor(inputs["start_pos"], dtype=torch.int32).npu(),
        rope_head_dim=inputs["rope_head_dim"],
        cmp_ratio=inputs["cmp_ratio"],
        coff=inputs["coff"],
        norm_eps=inputs["norm_eps"],
        rotary_mode=rotary_mode,
        cache_mode=inputs["cache_mode"],
        state_cache_stride_dim0=state_cache_stride_dim0,
    )


def _call_native_compressor(inputs, state_cache):
    return torch.ops.custom.compressor(
        inputs["x"].npu(),
        inputs["wkv"].npu(),
        inputs["wgate"].npu(),
        state_cache,
        inputs["ape"].npu(),
        inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(),
        inputs["rope_cos"].npu(),
        state_block_table=inputs["state_block_table"].npu(),
        cu_seqlens=(
            inputs["cu_seqlens"].npu() if inputs["cu_seqlens"] is not None else None
        ),
        seqused=torch.tensor(inputs["seqused"], dtype=torch.int32).npu(),
        start_pos=torch.tensor(inputs["start_pos"], dtype=torch.int32).npu(),
        rope_head_dim=inputs["rope_head_dim"],
        cmp_ratio=inputs["cmp_ratio"],
        coff=inputs["coff"],
        norm_eps=inputs["norm_eps"],
        rotary_mode=2,
        cache_mode=inputs["cache_mode"],
        state_cache_stride_dim0=0,
    )


def _small_cycle_case(**overrides):
    params = dict(
        start_pos=[13],
        seq_len=9,
        coff=2,
        cmp_ratio=4,
        head_dim=512,
        hidden=1024,
        cache_mode=2,
        layout="TH",
        dtype=torch.bfloat16,
        batch=1,
        block_size=16,
        noncontiguous_dim0=False,
        seed=20260820,
    )
    params.update(overrides)
    return _make_a5_inputs(**params)


@unittest.skipUnless(_is_ascend950(), "A5 Compressor tests require Ascend950")
class TestCompressorA5(unittest.TestCase):
    def _assert_precision(self, actual, expected, dtype):
        rtol, atol = (0.0078125, 1e-4) if dtype == torch.bfloat16 else (0.005, 2.5e-5)
        actual, expected = actual.float().cpu(), expected.float().cpu()
        self.assertTrue(torch.isfinite(expected).all().item())
        self.assertTrue(torch.isfinite(actual).all().item())
        passed = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        self.assertGreaterEqual(passed.float().mean().item(), 0.995)

    def test_cycle_table_is_one_dimensional(self):
        inputs = _small_cycle_case()
        self.assertEqual(inputs["state_block_table"].dim(), 1)
        state = inputs["state_cache"].clone().npu()
        out = _call_npu_compressor(inputs, state)
        torch.npu.synchronize()
        self.assertEqual(out.dim(), 2)

    def test_cycle_wrap_updates_expected_slot(self):
        inputs = _small_cycle_case(start_pos=[16])
        expected_out, expected_state, written = _reference_compressor_a5(inputs)
        state = inputs["state_cache"].clone().npu()
        actual_out = _call_npu_compressor(inputs, state)
        torch.npu.synchronize()
        self._assert_precision(actual_out, expected_out, inputs["dtype"])
        self._assert_precision(
            state.cpu()[written], expected_state[written], inputs["dtype"]
        )
        self.assertTrue(torch.equal(state.cpu()[~written], expected_state[~written]))

    def test_native_ab_c4a_bf16_th_cycle(self):
        inputs = _small_cycle_case()
        if not hasattr(torch.ops.custom, "compressor"):
            self.skipTest("torch.ops.custom.compressor is required for native A/B")
        native_state, migrated_state = (
            inputs["state_cache"].clone().npu(),
            inputs["state_cache"].clone().npu(),
        )
        native_out = _call_native_compressor(inputs, native_state)
        migrated_out = _call_npu_compressor(inputs, migrated_state)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(native_out, migrated_out))
        self.assertTrue(torch.equal(native_state, migrated_state))

    def test_noncontiguous_dim0_stride(self):
        inputs = _small_cycle_case(noncontiguous_dim0=True)
        expected_out, expected_state, written = _reference_compressor_a5(inputs)
        shape = inputs["state_cache"].shape
        backing = torch.empty(
            (shape[0] * 2, shape[1], shape[2]), dtype=torch.float32, device="npu"
        )
        state = backing[::2]
        state.copy_(inputs["state_cache"].contiguous().npu())
        self.assertGreater(state.stride(0), shape[1] * shape[2])
        actual_out = _call_npu_compressor(inputs, state)
        torch.npu.synchronize()
        self._assert_precision(actual_out, expected_out, inputs["dtype"])
        self._assert_precision(
            state.cpu()[written], expected_state[written], inputs["dtype"]
        )
        self.assertTrue(torch.equal(state.cpu()[~written], expected_state[~written]))

    def test_explicit_stride_mismatch_rejected(self):
        inputs = _small_cycle_case()
        state = inputs["state_cache"].clone().npu()
        with self.assertRaisesRegex(RuntimeError, "state_cache_stride_dim0"):
            _call_npu_compressor(
                inputs, state, state_cache_stride_dim0=state.stride(0) + 1
            )

    def test_rotary_mode_one_rejected(self):
        inputs = _small_cycle_case()
        state = inputs["state_cache"].clone().npu()
        with self.assertRaisesRegex(RuntimeError, "rotary_mode=2"):
            _call_npu_compressor(inputs, state, rotary_mode=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

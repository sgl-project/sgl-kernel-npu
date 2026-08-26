# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

import unittest

import numpy as np
import torch

try:
    import sgl_kernel_npu
    import torch_npu
except ModuleNotFoundError:
    pass

SCENARIOS = {
    "c4a": dict(cmp_ratio=4, coff=2, head_dim=512),
    "c4li": dict(cmp_ratio=4, coff=2, head_dim=128),
    "c128a": dict(cmp_ratio=128, coff=1, head_dim=512),
}
DTYPES = (torch.bfloat16, torch.float16)
LAYOUTS = ("TH", "BSH")
CACHE_MODES = (1, 2)
STATE_SENTINEL = 17.0


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


def _state_slot(inputs, batch_index, absolute_pos):
    block_size = inputs["state_cache"].shape[1]
    if inputs["cache_mode"] == 1:
        bank_id = inputs["state_block_table"][batch_index, absolute_pos // block_size]
    else:
        bank_id = inputs["state_block_table"][batch_index]
    return int(bank_id), int(absolute_pos) % int(block_size)


def _read_state(state, inputs, batch_index, absolute_start, absolute_end):
    return torch.stack(
        [
            state[_state_slot(inputs, batch_index, pos)]
            for pos in range(absolute_start, absolute_end)
        ],
        dim=0,
    )


def _write_state(state, inputs, batch_index, absolute_pos, value):
    state[_state_slot(inputs, batch_index, absolute_pos)] = value


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
    rope_shape = (
        (rope_rows, 64)
        if layout == "TH"
        else (batch, (seq_len + cmp_ratio - 1) // cmp_ratio, 64)
    )
    rope_sin = torch.randn(rope_shape, generator=gen).float() * 0.01
    rope_cos = (
        torch.ones(rope_shape) + torch.randn(rope_shape, generator=gen).float() * 0.01
    )
    if cache_mode == 1:
        table_width = (max(start_pos) + seq_len + block_size - 1) // block_size
        state_block_table = torch.arange(
            1, batch * table_width + 1, dtype=torch.int32
        ).reshape(batch, table_width)
        block_num = batch * table_width + 2
    else:
        state_block_table = torch.arange(1, batch + 1, dtype=torch.int32)
        block_size = (
            2 * cmp_ratio + seq_len - 1 if coff == 2 else cmp_ratio + seq_len - 1
        )
        block_num = batch + 2
    if noncontiguous_dim0:
        backing = torch.empty(
            (block_num * 2, block_size, 2 * width), dtype=torch.float32
        )
        state_cache = backing[::2]
        state_cache.fill_(STATE_SENTINEL)
        assert state_cache.stride(1) == 2 * width
        assert state_cache.stride(0) > block_size * 2 * width
    else:
        state_cache = torch.full(
            (block_num, block_size, 2 * width), STATE_SENTINEL, dtype=torch.float32
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
        state_block_table=state_block_table,
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
        noncontiguous_dim0=noncontiguous_dim0,
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
    rope_sin = inputs["rope_sin"].float().numpy().reshape(-1, inputs["rope_head_dim"])
    rope_cos = inputs["rope_cos"].float().numpy().reshape(-1, inputs["rope_head_dim"])
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
        output_written = torch.zeros(output.shape, dtype=torch.bool)
    else:
        output = np.zeros(
            (len(start_pos), (x.shape[1] + cmp_ratio - 1) // cmp_ratio, head_dim),
            dtype=np.float32,
        )
        output_written = torch.zeros(output.shape, dtype=torch.bool)
    out_index = 0
    for batch_index, batch_start in enumerate(start_pos):
        batch_out = 0
        batch_used = seqused[batch_index]
        compress_until = (batch_start + batch_used) // cmp_ratio * cmp_ratio
        seq_index = 0
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
            if (
                inputs["cache_mode"] == 1
                or absolute_start >= compress_until - (coff - 1) * cmp_ratio
            ):
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
                    _write_state(state, inputs, batch_index, absolute_pos, value)
                    _write_state(
                        written,
                        inputs,
                        batch_index,
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
                        history = _read_state(
                            state,
                            inputs,
                            batch_index,
                            batch_start - cnt_from_state,
                            batch_start,
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
                            history = _read_state(
                                state,
                                inputs,
                                batch_index,
                                copy_start,
                                copy_start + cnt_from_state,
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
                            history = _read_state(
                                state, inputs, batch_index, copy_start, batch_start
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
                    output_written[out_index] = True
                else:
                    output[batch_index, batch_out] = compressed
                    output_written[batch_index, batch_out] = True
                batch_out += 1
                out_index += 1
            seq_index = absolute_end - batch_start
    return torch.tensor(output).to(x_dtype), output_written, state, written


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
    )


def _make_a5_device_inputs(inputs):
    """Move fixed compressor inputs and scalar metadata to NPU once."""
    device_inputs = dict(inputs)
    for name in (
        "x",
        "wkv",
        "wgate",
        "ape",
        "norm_weight",
        "rope_sin",
        "rope_cos",
        "state_block_table",
    ):
        device_inputs[name] = inputs[name].npu()
    device_inputs["cu_seqlens"] = (
        inputs["cu_seqlens"].npu() if inputs["cu_seqlens"] is not None else None
    )
    device_inputs["seqused"] = torch.tensor(
        inputs["seqused"], dtype=torch.int32, device="npu"
    )
    device_inputs["start_pos"] = torch.tensor(
        inputs["start_pos"], dtype=torch.int32, device="npu"
    )
    return device_inputs


def _call_a5_device_compressor(operator, inputs, state_cache):
    return operator(
        inputs["x"],
        inputs["wkv"],
        inputs["wgate"],
        state_cache,
        inputs["ape"],
        inputs["norm_weight"],
        inputs["rope_sin"],
        inputs["rope_cos"],
        state_block_table=inputs["state_block_table"],
        cu_seqlens=inputs["cu_seqlens"],
        seqused=inputs["seqused"],
        start_pos=inputs["start_pos"],
        rope_head_dim=inputs["rope_head_dim"],
        cmp_ratio=inputs["cmp_ratio"],
        coff=inputs["coff"],
        norm_eps=inputs["norm_eps"],
        rotary_mode=inputs["rotary_mode"],
        cache_mode=inputs["cache_mode"],
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


class TestCompressorA5Reference(unittest.TestCase):
    def test_device_call_uses_native_compressor_argument_contract(self):
        inputs = _small_cycle_case()
        call = {}

        def native_compressor_schema(*args, **kwargs):
            call.update(args=args, kwargs=kwargs)
            return args[0]

        self.assertIs(
            _call_a5_device_compressor(
                native_compressor_schema, inputs, inputs["state_cache"]
            ),
            inputs["x"],
        )
        self.assertEqual(len(call["args"]), 8)
        self.assertEqual(
            set(call["kwargs"]),
            {
                "state_block_table",
                "cu_seqlens",
                "seqused",
                "start_pos",
                "rope_head_dim",
                "cmp_ratio",
                "coff",
                "norm_eps",
                "rotary_mode",
                "cache_mode",
            },
        )

    def test_continuous_state_uses_distinct_physical_blocks(self):
        inputs = _make_a5_inputs(
            start_pos=[127, 255],
            seq_len=129,
            coff=1,
            cmp_ratio=128,
            head_dim=512,
            hidden=1024,
            cache_mode=1,
            layout="TH",
            dtype=torch.float16,
            batch=2,
            block_size=16,
            noncontiguous_dim0=False,
            seed=20260820,
        )
        block_table = inputs["state_block_table"]

        self.assertEqual(block_table.unique().numel(), block_table.numel())
        self.assertGreater(inputs["state_cache"].shape[0], block_table.max().item())

    def test_cycle_state_uses_required_history_capacity(self):
        inputs = _make_a5_inputs(
            start_pos=[15],
            seq_len=129,
            coff=1,
            cmp_ratio=128,
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

        self.assertEqual(inputs["state_cache"].shape[1], 256)

    def test_continuous_c4a_writes_every_valid_token(self):
        inputs = _make_a5_inputs(
            start_pos=[3],
            seq_len=5,
            coff=2,
            cmp_ratio=4,
            head_dim=512,
            hidden=1024,
            cache_mode=1,
            layout="TH",
            dtype=torch.bfloat16,
            batch=1,
            block_size=16,
            noncontiguous_dim0=False,
            seed=20260820,
        )
        original_state = inputs["state_cache"].clone()
        _, _, state, written = _reference_compressor_a5(inputs)

        for absolute_pos in range(3, 8):
            bank_id, slot = _state_slot(inputs, 0, absolute_pos)
            self.assertTrue(written[bank_id, slot].all().item())
            self.assertFalse(
                torch.equal(state[bank_id, slot], original_state[bank_id, slot])
            )
        self.assertTrue(torch.equal(state[~written], original_state[~written]))


@unittest.skipUnless(_is_ascend950(), "A5 Compressor tests require Ascend950")
class TestCompressorA5(unittest.TestCase):
    def _assert_precision(self, actual, expected, dtype):
        rtol, atol = (0.0078125, 1e-4) if dtype == torch.bfloat16 else (0.005, 2.5e-5)
        actual, expected = actual.float().cpu(), expected.float().cpu()
        self.assertEqual(actual.numel(), expected.numel())
        if actual.numel() == 0:
            return
        self.assertTrue(torch.isfinite(expected).all().item())
        self.assertTrue(torch.isfinite(actual).all().item())
        passed = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        pass_rate = passed.float().mean().item()
        max_abs_error = (actual - expected).abs().max().item()
        bad_indices = (~passed).flatten().nonzero().flatten()
        sample_indices = bad_indices[:8]
        sample_actual = actual.flatten()[sample_indices].tolist()
        sample_expected = expected.flatten()[sample_indices].tolist()
        self.assertGreaterEqual(
            pass_rate,
            0.995,
            msg=(
                f"pass_rate={pass_rate}, max_abs_error={max_abs_error}, "
                f"numel={actual.numel()}, dtype={dtype}, "
                f"bad_indices={sample_indices.tolist()}, "
                f"actual={sample_actual}, expected={sample_expected}"
            ),
        )

    def _new_npu_state(self, inputs):
        if not inputs["noncontiguous_dim0"]:
            return inputs["state_cache"].clone().npu(), None, None
        shape = inputs["state_cache"].shape
        backing = torch.full(
            (shape[0] * 2, shape[1], shape[2]),
            STATE_SENTINEL,
            dtype=torch.float32,
            device="npu",
        )
        state = backing[::2]
        state.copy_(inputs["state_cache"].npu())
        self.assertGreater(state.stride(0), shape[1] * shape[2])
        original_padding = backing[1::2].cpu()
        return state, backing, original_padding

    def _assert_padding_unchanged(self, backing, original_padding):
        if backing is None:
            return
        self.assertTrue(torch.all(original_padding == STATE_SENTINEL).item())
        self.assertTrue(torch.equal(backing[1::2].cpu(), original_padding))

    def _assert_untouched_state(self, actual_state, original_state, written):
        actual_state = actual_state.cpu()
        self.assertTrue(torch.equal(actual_state[~written], original_state[~written]))
        self.assertTrue(torch.equal(actual_state[-1], original_state[-1]))
        self.assertTrue(torch.all(original_state == STATE_SENTINEL).item())

    def _assert_golden_case(self, inputs):
        expected_out, output_written, expected_state, state_written = (
            _reference_compressor_a5(inputs)
        )
        original_state = inputs["state_cache"].clone()
        actual_state, actual_backing, original_padding = self._new_npu_state(inputs)
        actual_out = _call_npu_compressor(inputs, actual_state)
        torch.npu.synchronize()
        self._assert_precision(
            actual_out[output_written], expected_out[output_written], inputs["dtype"]
        )
        self._assert_precision(
            actual_state.cpu()[state_written],
            expected_state[state_written],
            inputs["dtype"],
        )
        self._assert_untouched_state(actual_state, original_state, state_written)
        self._assert_padding_unchanged(actual_backing, original_padding)

    def _assert_native_ab_case(self, inputs):
        if not hasattr(torch.ops.custom, "compressor"):
            self.skipTest("torch.ops.custom.compressor is required for native A/B")
        _, _, _, state_written = _reference_compressor_a5(inputs)
        original_state = inputs["state_cache"].clone()
        native_state, native_backing, native_padding = self._new_npu_state(inputs)
        migrated_state, migrated_backing, migrated_padding = self._new_npu_state(inputs)
        native_out = _call_native_compressor(inputs, native_state)
        migrated_out = _call_npu_compressor(inputs, migrated_state)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(native_out, migrated_out))
        self.assertTrue(torch.equal(native_state, migrated_state))
        self._assert_untouched_state(native_state, original_state, state_written)
        self._assert_untouched_state(migrated_state, original_state, state_written)
        self._assert_padding_unchanged(native_backing, native_padding)
        self._assert_padding_unchanged(migrated_backing, migrated_padding)

    def _matrix_inputs(
        self, scenario, dtype, layout, cache_mode, hidden=1024, **overrides
    ):
        params = dict(
            start_pos=[scenario["cmp_ratio"] - 1, 2 * scenario["cmp_ratio"] - 1],
            seq_len=scenario["cmp_ratio"] + 1,
            hidden=hidden,
            cache_mode=cache_mode,
            layout=layout,
            dtype=dtype,
            batch=2,
            block_size=16,
            noncontiguous_dim0=False,
            seed=20260820,
        )
        params.update(scenario)
        params.update(overrides)
        return _make_a5_inputs(**params)

    def test_full_correctness_matrix(self):
        for scenario_name, scenario in SCENARIOS.items():
            for dtype in DTYPES:
                for layout in LAYOUTS:
                    for cache_mode in CACHE_MODES:
                        with self.subTest(
                            scenario=scenario_name,
                            dtype=dtype,
                            layout=layout,
                            cache_mode=cache_mode,
                        ):
                            self._assert_golden_case(
                                self._matrix_inputs(scenario, dtype, layout, cache_mode)
                            )

    def test_native_ab_full_matrix(self):
        if not hasattr(torch.ops.custom, "compressor"):
            self.skipTest("torch.ops.custom.compressor is required for native A/B")
        for scenario_name, scenario in SCENARIOS.items():
            for dtype in DTYPES:
                for layout in LAYOUTS:
                    for cache_mode in CACHE_MODES:
                        with self.subTest(
                            scenario=scenario_name,
                            dtype=dtype,
                            layout=layout,
                            cache_mode=cache_mode,
                        ):
                            self._assert_native_ab_case(
                                self._matrix_inputs(scenario, dtype, layout, cache_mode)
                            )

    def test_hidden_7168_golden_cases(self):
        for scenario_name, scenario in SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                self._assert_golden_case(
                    self._matrix_inputs(scenario, torch.bfloat16, "TH", 1, hidden=7168)
                )

    def test_cycle_position_boundaries_and_bank_isolation(self):
        for scenario_name, scenario in SCENARIOS.items():
            seq_len = scenario["cmp_ratio"] + 1
            cycle_size = (
                2 * scenario["cmp_ratio"] + seq_len - 1
                if scenario["coff"] == 2
                else scenario["cmp_ratio"] + seq_len - 1
            )
            positions = (
                0,
                scenario["cmp_ratio"] - 1,
                scenario["cmp_ratio"],
                8192,
                cycle_size - 1,
                cycle_size,
                cycle_size + 1,
            )
            for position in positions:
                with self.subTest(scenario=scenario_name, position=position):
                    inputs = self._matrix_inputs(
                        scenario,
                        torch.bfloat16,
                        "TH",
                        2,
                        start_pos=[position, position + scenario["cmp_ratio"]],
                    )
                    self.assertEqual(inputs["state_block_table"].dim(), 1)
                    self.assertNotEqual(
                        inputs["state_block_table"][0].item(),
                        inputs["state_block_table"][1].item(),
                    )
                    self._assert_golden_case(inputs)

    def test_multi_bank_isolation(self):
        for scenario_name, scenario in SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                inputs = self._matrix_inputs(
                    scenario,
                    torch.bfloat16,
                    "TH",
                    2,
                    start_pos=[0, 0],
                )
                perturbed_inputs = dict(inputs)
                perturbed_inputs["x"] = inputs["x"].clone()
                perturbed_inputs["x"][: inputs["seqused"][0]].add_(0.1)
                baseline_state, _, _ = self._new_npu_state(inputs)
                perturbed_state, _, _ = self._new_npu_state(inputs)
                _call_npu_compressor(inputs, baseline_state)
                _call_npu_compressor(perturbed_inputs, perturbed_state)
                torch.npu.synchronize()
                changed_bank = inputs["state_block_table"][0].item()
                other_bank = inputs["state_block_table"][1].item()
                self.assertFalse(
                    torch.equal(
                        baseline_state[changed_bank].cpu(),
                        perturbed_state[changed_bank].cpu(),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        baseline_state[other_bank].cpu(),
                        perturbed_state[other_bank].cpu(),
                    )
                )

    def test_noncontiguous_dim0_golden_cases(self):
        for scenario_name, scenario in SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                self._assert_golden_case(
                    self._matrix_inputs(
                        scenario,
                        torch.float16,
                        "BSH",
                        2,
                        noncontiguous_dim0=True,
                    )
                )

    def test_native_ab_noncontiguous_dim0(self):
        self._assert_native_ab_case(
            self._matrix_inputs(
                SCENARIOS["c4a"],
                torch.bfloat16,
                "TH",
                2,
                noncontiguous_dim0=True,
            )
        )

    def test_cycle_graph_replay_matches_eager_without_allocations(self):
        """A missing capture-safe tiling path changes output/state or allocates on replay."""
        inputs = self._matrix_inputs(
            SCENARIOS["c4a"],
            torch.bfloat16,
            "TH",
            2,
            batch=1,
            start_pos=[13],
        )
        device_inputs = _make_a5_device_inputs(inputs)
        initial_state, _, _ = self._new_npu_state(inputs)
        warmup_state = initial_state.clone()

        # Warm the exact shape before capture so host tiling is not captured.
        _call_a5_device_compressor(
            torch.ops.npu.compressor, device_inputs, warmup_state
        )
        torch.npu.synchronize()

        eager_state = initial_state.clone()
        eager_results = []
        for _ in range(3):
            eager_output = _call_a5_device_compressor(
                torch.ops.npu.compressor, device_inputs, eager_state
            )
            torch.npu.synchronize()
            eager_results.append((eager_output.cpu(), eager_state.cpu()))

        captured_state = initial_state.clone()
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
            captured_output = _call_a5_device_compressor(
                torch.ops.npu.compressor, device_inputs, captured_state
            )
        torch.npu.synchronize()

        # The capture launch mutates state; reset before the eager-matched replay sequence.
        captured_state.copy_(initial_state)
        torch.npu.synchronize()
        allocated_before = torch.npu.memory_allocated()
        for expected_output, expected_state in eager_results:
            graph.replay()
            torch.npu.synchronize()
            self.assertTrue(torch.equal(captured_output.cpu(), expected_output))
            self.assertTrue(torch.equal(captured_state.cpu(), expected_state))
        allocated_after = torch.npu.memory_allocated()
        self.assertEqual(allocated_before, allocated_after)

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

    def test_unsupported_scenario_rejected(self):
        inputs = _small_cycle_case(coff=1)
        state = inputs["state_cache"].clone().npu()
        with self.assertRaisesRegex(
            RuntimeError, r"supported \(cmp_ratio, coff, head_dim\)"
        ):
            _call_npu_compressor(inputs, state)


if __name__ == "__main__":
    unittest.main(verbosity=2)

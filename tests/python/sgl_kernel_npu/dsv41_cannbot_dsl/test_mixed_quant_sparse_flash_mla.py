# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Self-contained MQSMLA precision tests: local inputs, CPU golden and samples."""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("cannbotdsl")
import importlib

import torch
from sgl_kernel_npu.operators.attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

MQSMLA = importlib.import_module(
    "sgl_kernel_npu.implementation.cannbot-dsl.attention.mixed_quant_sparse_flash_mla"
)
METADATA = importlib.import_module(
    "sgl_kernel_npu.implementation.cannbot-dsl.attention.mixed_quant_sparse_flash_mla_metadata"
)
from mixed_quant_sparse_flash_mla_paramset import ENABLED_PARAMS, TEST_PARAMS

LOGGER = logging.getLogger(__name__)

D, N1 = 512, 64
KV_ROW_BYTES_ORI, KV_ROW_BYTES_CMP, KV_PAGE_PADDING_BYTES = 544, 320, 64


def _selected_cases():
    selection = os.environ.get("MQSMLA_CASES", "").strip()
    names = (
        list(TEST_PARAMS)
        if selection == "all"
        else (
            [name.strip() for name in selection.split(",")]
            if selection
            else ENABLED_PARAMS
        )
    )
    unknown = set(names) - TEST_PARAMS.keys()
    if unknown:
        raise ValueError(f"Unknown MQSMLA_CASES: {sorted(unknown)}")
    return names


def _make_kv(batch, tokens, block_size, fp4, generator):
    """Encode logical KV, then shuffle physical pages independently of the golden.

    Features are nope[448], rope[64], followed by BF16 group scales.
    FP4 stores the earlier feature in the low nibble. The CPU reference keeps
    dequantized logical rows, so it never shares the kernel's page addressing.
    """
    pages = (tokens + block_size - 1) // block_size
    shape = (batch, pages * block_size, D)
    group = 16 if fp4 else 32
    scales = (
        0.125 + torch.rand((*shape[:-1], D // group), generator=generator) * 0.375
    ).bfloat16()
    if fp4:
        codes = torch.randint(0, 16, shape, generator=generator, dtype=torch.uint8)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        values = lut[codes.long()]
        encoded = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        values = (torch.randn(shape, generator=generator) * 2).to(torch.float8_e4m3fn)
        encoded = values.view(torch.uint8)
        values = values.float()
    logical = (values * scales.float().repeat_interleave(group, dim=-1)).bfloat16()
    packed = torch.cat((encoded, scales.view(torch.uint8)), dim=-1)
    table = (
        torch.randperm(batch * pages, generator=generator).reshape(batch, pages).int()
    )
    physical = torch.empty_like(packed.reshape(batch * pages, block_size, -1))
    physical[table.long().flatten()] = packed.reshape_as(physical)
    return {
        "phys": {"row": physical},
        "num_phys_blocks": batch * pages,
        "block_size": block_size,
        "block_table": table,
    }, logical


def gen_data(params):
    """Generate a bounded deterministic case without files or external test code."""
    p = dict(params)
    generator = torch.Generator().manual_seed(p.get("seed", 42))
    spans = p["q_lengths"]
    batch, rows = len(spans), sum(spans)
    has_cmp = p.get("has_cmp", True)
    q = (torch.randn((rows, N1, D), generator=generator) * 0.25).bfloat16()
    sinks = torch.randn(N1, generator=generator)
    cu = torch.tensor([0, *np.cumsum(spans).tolist()], dtype=torch.int32)
    case = {
        "q": q,
        "sinks": sinks,
        "cu_seqlens_q": cu,
        "seqused_q": torch.tensor(spans, dtype=torch.int32),
        "b": batch,
        "t1": rows,
        "quant_mode": 1,
        "mode": "ORI_CMP_SPARSE" if has_cmp else "ORI_SPARSE",
        "scale": D**-0.5,
    }
    logical = {}
    for side, fp4 in (("ori", False), ("cmp", True)):
        width = p[f"k_{side}"]
        pa, logical[side] = _make_kv(
            batch, p[f"s_{side}"], p[f"block_size_{side}"], fp4, generator
        )
        case["pa" if side == "ori" else "pa_cmp"] = pa
        indices = torch.randint(
            p[f"s_{side}"], (rows, width), generator=generator, dtype=torch.int32
        )
        lengths = torch.full((rows,), width, dtype=torch.int32)
        if p.get("vary_lengths", False):
            lengths = torch.tensor(
                ([0, 1, min(127, width), width] * ((rows + 3) // 4))[:rows],
                dtype=torch.int32,
            )
        if (side == "cmp" and not has_cmp) or (
            side == "ori" and p.get("zero_ori", False)
        ):
            lengths.zero_()
        if p.get(f"omit_{side}_topk_length", False):
            lengths.fill_(width)
        # Invalid unused indices catch reads beyond each valid prefix.
        indices[torch.arange(width)[None, :] >= lengths[:, None]] = -1
        case[f"idx_{side}"] = indices
        case[f"len_{side}"] = lengths
    output = torch.zeros_like(q)
    lse = torch.empty((1, rows, N1), dtype=torch.float32)
    for b in range(batch):
        for row in range(int(cu[b]), int(cu[b + 1])):
            sides = [
                logical[side][
                    b, case[f"idx_{side}"][row, : int(case[f"len_{side}"][row])].long()
                ]
                for side in ("ori", "cmp")
            ]
            # Dense FP64 logsumexp independently checks LSE and sink semantics.
            kv = torch.cat(sides).double()
            logits = q[row].double() @ kv.T * case["scale"]
            all_logits = torch.cat((logits, sinks.double()[:, None]), dim=1)
            lse[0, row] = torch.logsumexp(all_logits, dim=1).float()
            # Match sparse_flash_mla's CPU reference arithmetic: online softmax
            # over 128-token tiles, BF16 probabilities for PV, FP32 accumulation.
            # Each side starts a new tile sequence, including its own tail.
            row_max = sinks.clone()
            row_sum = torch.ones(N1)
            accumulator = torch.zeros(N1, D)
            for values in sides:
                for start in range(0, len(values), 128):
                    tile = values[start : start + 128].float()
                    scores = q[row].float() @ tile.T * case["scale"]
                    new_max = torch.maximum(row_max, scores.max(dim=1).values)
                    correction = torch.exp(row_max - new_max)
                    weights = torch.exp(scores - new_max[:, None])
                    row_sum = row_sum * correction + weights.sum(dim=1)
                    accumulator = (
                        accumulator * correction[:, None]
                        + weights.bfloat16().float() @ tile
                    )
                    row_max = new_max
            output[row] = (accumulator / row_sum[:, None]).bfloat16()
    return p, case, output.reshape(-1, D), lse


def check_result(expect, npu_result, chunk_elements=1 << 20):
    """与原BF16比较规则一致，跨块累计全局合格率及前10000000个错误的最大相对误差。"""
    expected = expect.reshape(-1)
    actual = npu_result.reshape(-1)
    total = actual.numel()
    if total != expected.numel():
        LOGGER.error("Error, the sizes of NPU output and benchmark differ")
        return "Failed", 0.0
    errors = 0
    sampled_errors = 0
    max_error = 0.0
    for start in range(0, total, chunk_elements):
        real = actual[start : start + chunk_elements].cpu().float().numpy()
        golden = expected[start : start + chunk_elements].cpu().float().numpy()
        close = np.isclose(real, golden, rtol=0.0078125, atol=0.0001, equal_nan=True)
        bad = np.flatnonzero(~close)
        errors += bad.size
        if bad.size and sampled_errors < 10000000:
            take = min(bad.size, 10000000 - sampled_errors)
            # 保持原比较器的float32运算顺序与max规则，包括NaN行为。
            diff_abs = abs(golden - real)
            b1 = np.maximum(np.abs(real), np.abs(golden))
            b2 = float((1.0 / (1 << 14)) / 0.005)
            b = np.add(np.maximum(b1, b2), 10e-10)
            relative = (diff_abs / (b + 10e-10))[bad[:take]]
            for value in relative:
                if sampled_errors == 0 or value > max_error:
                    max_error = value
                sampled_errors += 1
    percent = float(total - errors) / float(total) * 100.0
    result = "Pass" if percent >= (1 - 0.005) * 100.0 else "Failed"
    if sampled_errors and max_error >= 10:
        result = "Failed"
    LOGGER.info(
        "BF16 chunked compare: count=%s, errors=%s, pass=%.6f%%, "
        "rtol=0.0078125, atol=0.0001, max_relative=%s, result=%s",
        total,
        errors,
        percent,
        max_error,
        result,
    )
    return result, percent


def _pa_tensor(pa, row_bytes, *, axis0_noncontiguous=False):
    """先搬完整含 padding 的 storage，再在 NPU 上切视图，保留真实页距。"""
    row = pa["phys"]["row"]
    shape = (pa["num_phys_blocks"], pa["block_size"], 1, row_bytes)
    if not axis0_noncontiguous:
        return row.reshape(shape).npu()
    page_bytes = pa["block_size"] * row_bytes
    stride0 = page_bytes + KV_PAGE_PADDING_BYTES
    strides = (stride0, row_bytes, row_bytes, 1)
    # 0xA5 毒化页间无用数据；golden 保留原始逻辑 KV，不读取这份 storage。
    backing = torch.full((shape[0] * stride0,), 0xA5, dtype=torch.uint8)
    cpu_view = backing.as_strided(shape, strides)
    cpu_view.copy_(row.reshape(shape))
    result = backing.npu().as_strided(shape, strides)
    assert result.stride() == strides and result.storage_offset() == 0
    if shape[0] > 1:
        assert not result.is_contiguous()
    return result


def _call_public(kwargs, mode, t1, *, omit_ori_length=False, omit_cmp_length=False):
    """Run this checkout’s metadata and AICore public entries."""
    kwargs = dict(kwargs)
    if kwargs.get("metadata") is None:
        # Metadata always requires both length tensors; an absent side (cmp in
        # ORI_SPARSE, ori in CMP_SPARSE) passes a zero-length tensor.
        ori_lengths = kwargs["ori_topk_length"]
        if ori_lengths is None:
            ori_lengths = torch.zeros((t1, 1), dtype=torch.int32).npu()
        cmp_lengths = kwargs["cmp_topk_length"]
        if cmp_lengths is None:
            cmp_lengths = torch.zeros((t1, 1), dtype=torch.int32).npu()
        kwargs["metadata"] = mixed_quant_sparse_flash_mla_metadata(
            ori_lengths,
            cmp_lengths,
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            num_heads_q=N1,
            num_heads_kv=1,
            head_dim=D,
            quant_mode=kwargs["quant_mode"],
            has_cmp_kv=(mode != "ORI_SPARSE"),
        )
    # Metadata keeps its required length inputs; attention can omit either side.
    if omit_ori_length:
        kwargs.pop("ori_topk_length")
    if omit_cmp_length:
        kwargs.pop("cmp_topk_length")
    want_lse = kwargs.get("return_softmax_lse", False)
    out = torch.empty_like(kwargs["q"])
    lse = torch.empty(
        (1, t1, N1) if want_lse else (0,),
        dtype=torch.float32,
        device=kwargs["q"].device,
    )
    mixed_quant_sparse_flash_mla(**kwargs, out=out, lse=lse)
    output = out.cpu().reshape(-1, D)
    lse = lse.cpu()
    assert torch.isfinite(output).all(), "attention output must be finite"
    if want_lse:
        assert torch.isfinite(lse).all()
    else:
        assert lse.shape == (0,) and lse.dtype == torch.float32
    return output, lse


def run_prepared(
    p,
    case,
    expected,
    expected_lse,
):
    """Execute locally generated inputs against the independent CPU reference."""
    mode = case["mode"]
    quant_mode = case["quant_mode"]
    pa_ori = case["pa"]
    has_ori = mode != "CMP_SPARSE"
    has_cmp = mode != "ORI_SPARSE"
    ori_kv = (
        _pa_tensor(
            pa_ori, KV_ROW_BYTES_ORI, axis0_noncontiguous=p["kv_axis0_noncontiguous"]
        )
        if has_ori
        else None
    )
    pa_cmp = case["pa_cmp"]
    kwargs = {
        "q": case["q"].npu(),
        "ori_kv": ori_kv,
        "cmp_kv": (
            _pa_tensor(
                pa_cmp,
                KV_ROW_BYTES_CMP,
                axis0_noncontiguous=p["kv_axis0_noncontiguous"],
            )
            if has_cmp
            else None
        ),
        "ori_sparse_indices": (
            case["idx_ori"].reshape(case["t1"], 1, -1).npu() if has_ori else None
        ),
        "cmp_sparse_indices": (
            case["idx_cmp"].reshape(case["t1"], 1, -1).npu() if has_cmp else None
        ),
        "ori_block_table": pa_ori["block_table"].npu() if has_ori else None,
        "cmp_block_table": pa_cmp["block_table"].npu() if has_cmp else None,
        "cu_seqlens_q": case["cu_seqlens_q"].npu(),
        "seqused_q": (
            case["seqused_q"].npu() if case.get("seqused_q") is not None else None
        ),
        "seqused_ori_kv": None,
        "seqused_cmp_kv": None,
        "ori_topk_length": (
            case["len_ori"].reshape(case["t1"], 1).npu() if has_ori else None
        ),
        "cmp_topk_length": (
            case["len_cmp"].reshape(case["t1"], 1).npu() if has_cmp else None
        ),
        "sinks": case["sinks"].npu(),
        "metadata": None,
        "quant_mode": quant_mode,
        "softmax_scale": case["scale"],
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "return_softmax_lse": p.get("return_softmax_lse", False),
    }
    for side in ("ori", "cmp"):
        kv = kwargs[f"{side}_kv"]
        if kv is not None:
            LOGGER.info(
                "%s_kv shape=%s stride=%s storage_offset=%s",
                side,
                tuple(kv.shape),
                kv.stride(),
                kv.storage_offset(),
            )
    actual, actual_lse = _call_public(
        kwargs,
        mode,
        case["t1"],
        omit_ori_length=p.get("omit_ori_topk_length", False),
        omit_cmp_length=p.get("omit_cmp_topk_length", False),
    )
    if kwargs["return_softmax_lse"]:
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    if has_ori:
        empty_rows = case["len_ori"].reshape(-1) == 0
        if has_cmp:
            empty_rows &= case["len_cmp"].reshape(-1) == 0
    else:
        empty_rows = case["len_cmp"].reshape(-1) == 0
    if bool(empty_rows.any()):
        empty_out = actual.reshape(case["t1"], N1, D)[empty_rows]
        assert (
            torch.count_nonzero(empty_out) == 0
        ), "empty queries must produce exact zero"
        if kwargs["return_softmax_lse"]:
            torch.testing.assert_close(
                actual_lse[0, empty_rows],
                case["sinks"].expand(int(empty_rows.sum()), N1),
                rtol=0,
                atol=0,
            )
    result, percent = check_result(expected, actual)
    assert result == "Pass", f"strict BF16 comparator failed: {percent:.6f}%"
    # 大prefill输出可达16GiB，误差统计不要同时物化三份全量FP32输出。
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    max_abs = 0.0
    for start in range(0, actual_flat.numel(), 1 << 20):
        diff = (
            actual_flat[start : start + (1 << 20)].float()
            - expected_flat[start : start + (1 << 20)].float()
        ).abs()
        chunk_max = diff.max().item()
        max_abs = chunk_max if chunk_max != chunk_max else max(max_abs, chunk_max)
    LOGGER.info(
        "[mqsmla test_case] NPU %s quant%s B=%s T1=%s OK: max_abs=%.6g",
        mode,
        quant_mode,
        case["b"],
        case["t1"],
        max_abs,
    )


@pytest.mark.npu
@pytest.mark.parametrize("case_name", _selected_cases())
def test_mixed_quant_sparse_flash_mla(case_name):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(0)
    run_prepared(*gen_data(TEST_PARAMS[case_name]))

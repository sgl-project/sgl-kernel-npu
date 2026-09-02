# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from utils import require_npu_op

pytestmark = require_npu_op("lightning_indexer_v2")

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def _get_data_from_pa_cache(key, block_table, act_s2):
    block_num, block_size, n2, d = key.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_block_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_block_num * block_size
    out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
    for i in range(need_block_num):
        out[i * block_size : (i + 1) * block_size, :] = key[
            block_table[i], ...
        ].reshape(block_size, d)

    return out[:act_s2, :]


def _lightning_indexer_ref(
    query,
    key,
    weights,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    layout_query="BSND",
    sparse_count=2048,
    sparse_mode=3,
):
    """CPU reference. Returns (indices, values), values aligned with indices."""
    batch_size = query.shape[0]
    if layout_query == "TND":
        batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = key.shape[2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2

    idx_out = (
        torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(
            -1, n2, sparse_count
        )
        - 1
    )
    val_out = torch.full(
        (idx_out.shape[0], n2, sparse_count),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )

    process_q_len = 0
    for batch_id in range(batch_size):
        if actual_seq_lengths_query is None:
            act_s1 = query.shape[1]
        elif layout_query == "TND":  # prefix sums
            act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
        else:
            act_s1 = actual_seq_lengths_query[batch_id]
        act_s2 = actual_seq_lengths_key[batch_id]

        now_q = (
            query.reshape(-1, n1, d)[process_q_len : process_q_len + act_s1, :, :]
            .transpose(0, 1)
            .to(torch.float32)
        )
        now_weights = (
            weights.reshape(-1, n1, 1)[process_q_len : process_q_len + act_s1, :, :]
            .transpose(0, 1)
            .to(torch.float32)
        )
        process_q_len += act_s1
        now_block_table = block_table[batch_id, :]
        now_k = (
            _get_data_from_pa_cache(key, now_block_table, act_s2)
            .transpose(0, 1)
            .to(torch.float32)
        )
        # n1,s1,d @ d,s2 -> n1,s1,s2
        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        weight_out = relu_out * now_weights
        # n1,s1,s2 -> s1,s2
        reduce_out = torch.sum(weight_out, dim=0)

        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]
        if sparse_mode == 3:
            for i in range(tmp_s1):
                reduce_out[-1 - i, tmp_s2 - i :] = float("-inf")
        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
        return_s2 = min(sparse_count, tmp_s2)
        lo, hi = process_q_len - act_s1, process_q_len
        idx_out[lo:hi, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        val_out[lo:hi, 0, :return_s2] = sorted_value[:, :return_s2]

    return idx_out.reshape(out_shape), val_out.reshape(out_shape)


def _make_tnd_inputs(dtype, b=3, t=5, s2=8192, n1=64, n2=1, d=128, block_size=256):
    np.random.seed(3)
    query = torch.tensor(np.random.uniform(-10, 10, (t, n1, d))).to(dtype)
    key = torch.tensor(
        np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, d))
    ).to(dtype)
    weights = torch.tensor(np.random.uniform(-1, 1, (t, n1))).to(dtype)
    # TND: actual_seq_lengths_query is a prefix sum
    actual_seq_lengths_query = torch.tensor([1, 3, 5]).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(
        torch.int32
    )
    block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(
        b, -1
    )
    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table


def _to_npu(*tensors):
    return [x.to("npu:%s" % DEVICE_ID) for x in tensors]


def _assert_index_sets_equal(testcase, npu_out, cpu_out, sparse_count):
    """The kernel does not guarantee ordering among equal scores, so compare as sets."""
    npu_out = npu_out.reshape(-1, sparse_count).cpu()
    cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
    for i in range(npu_out.shape[0]):
        testcase.assertEqual(sorted(npu_out[i].tolist()), sorted(cpu_out[i].tolist()))


class TestLightningIndexerV2(unittest.TestCase):
    def test_tnd_pa_eager(self):
        sparse_count, sparse_mode = 2048, 3
        for dtype in [torch.bfloat16, torch.float16]:
            (
                query,
                key,
                weights,
                asl_q,
                asl_k,
                block_table,
            ) = _make_tnd_inputs(dtype)

            cpu_idx, _ = _lightning_indexer_ref(
                query,
                key,
                weights,
                asl_q,
                asl_k,
                block_table,
                layout_query="TND",
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
            )

            q, k, w, aq, ak, bt = _to_npu(
                query, key, weights, asl_q, asl_k, block_table
            )
            npu_idx, npu_val = torch.ops.npu.lightning_indexer_v2(
                q,
                k,
                w,
                actual_seq_lengths_query=aq,
                actual_seq_lengths_key=ak,
                block_table=bt,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
            )

            _assert_index_sets_equal(self, npu_idx, cpu_idx, sparse_count)
            # return_values defaults to False -> empty second output
            self.assertEqual(npu_val.numel(), 0)

    def test_tnd_pa_return_values(self):
        sparse_count, sparse_mode = 2048, 3
        dtype = torch.bfloat16
        (
            query,
            key,
            weights,
            asl_q,
            asl_k,
            block_table,
        ) = _make_tnd_inputs(dtype)

        cpu_idx, cpu_val = _lightning_indexer_ref(
            query,
            key,
            weights,
            asl_q,
            asl_k,
            block_table,
            layout_query="TND",
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
        )

        q, k, w, aq, ak, bt = _to_npu(query, key, weights, asl_q, asl_k, block_table)
        npu_idx, npu_val = torch.ops.npu.lightning_indexer_v2(
            q,
            k,
            w,
            actual_seq_lengths_query=aq,
            actual_seq_lengths_key=ak,
            block_table=bt,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
            return_values=True,
        )

        _assert_index_sets_equal(self, npu_idx, cpu_idx, sparse_count)
        self.assertEqual(tuple(npu_val.shape), tuple(npu_idx.shape))
        self.assertEqual(npu_val.dtype, dtype)
        # Scores are sorted descending, so comparing the sorted vectors is
        # order-independent w.r.t. ties.
        npu_val_f = npu_val.float().reshape(-1, sparse_count).cpu()
        cpu_val_f = cpu_val.float().reshape(-1, sparse_count).cpu()
        finite = torch.isfinite(cpu_val_f)
        torch.testing.assert_close(
            npu_val_f[finite], cpu_val_f[finite], rtol=4e-2, atol=4e-2
        )

    def test_fp32_weights(self):
        """weights may be fp32 while query/key stay fp16/bf16 (upstream DT_W_FLAG)."""
        sparse_count, sparse_mode = 2048, 3
        dtype = torch.bfloat16
        (
            query,
            key,
            weights,
            asl_q,
            asl_k,
            block_table,
        ) = _make_tnd_inputs(dtype)

        cpu_idx, _ = _lightning_indexer_ref(
            query,
            key,
            weights.float(),
            asl_q,
            asl_k,
            block_table,
            layout_query="TND",
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
        )

        q, k, w, aq, ak, bt = _to_npu(
            query, key, weights.float(), asl_q, asl_k, block_table
        )
        npu_idx, _ = torch.ops.npu.lightning_indexer_v2(
            q,
            k,
            w,
            actual_seq_lengths_query=aq,
            actual_seq_lengths_key=ak,
            block_table=bt,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
        )

        _assert_index_sets_equal(self, npu_idx, cpu_idx, sparse_count)


if __name__ == "__main__":
    unittest.main()

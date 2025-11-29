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
import torch.nn as nn
import torch_npu
import torchair

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def _get_data_from_pa_cache(key, block_table, act_s2):
    block_num, block_size, n2, d = key.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
    for i in range(need_blcok_num):
        out[i * block_size : (i + 1) * block_size, :] = key[
            block_table[i], ...
        ].reshape(block_size, d)

    return out[:act_s2, :]


def _lightning_indexer(
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
    batch_size = query.shape[0]
    if layout_query == "TND":
        batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = key.shape[2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # 初始化为全-1
    out = (
        torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(
            -1, n2, sparse_count
        )
        - 1
    )
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    for batch_id in range(batch_size):
        if actual_seq_lengths_query is None:
            # 只能为BSND格式
            act_s1 = query.shape[1]
        else:
            if layout_query == "TND":  # TND格式时actual_seq_lengths_query为前缀和
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
        # sparse场景下三角置为-inf
        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]
        if sparse_mode == 3:
            for i in range(tmp_s1):
                reduce_out[-1 - i, tmp_s2 - i :] = float("-inf")
        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
        return_s2 = min(sparse_count, tmp_s2)
        out[process_q_len - act_s1 : process_q_len, 0, :return_s2] = sorted_indices.to(
            torch.int32
        )[:, :return_s2]

    out = out.reshape(out_shape)
    return out


class TestCustomLightningIndexer(unittest.TestCase):
    def test_tnd_lightning_indexer_eager(self):
        b = 3
        t = 5
        s2 = 8192
        n1 = 64
        n2 = 1
        d = 128
        block_size = 256
        layout_query = "TND"

        for dtype in [torch.bfloat16, torch.float16]:
            np.random.seed(3)
            query = torch.tensor(np.random.uniform(-10, 10, (t, n1, d))).to(dtype)
            key = torch.tensor(
                np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, d))
            ).to(dtype)
            weights = torch.tensor(np.random.uniform(-1, 1, (t, n1))).to(dtype)
            # TND格式下，actual_seq_lengths_query为前缀和表示
            actual_seq_lengths_query = torch.tensor([1, 3, 5]).to(torch.int32)
            actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(
                torch.int32
            )
            block_table = torch.tensor(
                [range(b * s2 // block_size)], dtype=torch.int32
            ).reshape(b, -1)
            layout_key = "PA_BSND"
            sparse_count = 2048
            sparse_mode = 3
            cpuout = _lightning_indexer(
                query,
                key,
                weights,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table,
                layout_query,
                sparse_count,
                sparse_mode,
            )

            torch_npu.npu.set_device(int(DEVICE_ID))
            query = query.to("npu:%s" % DEVICE_ID)
            key = key.to("npu:%s" % DEVICE_ID)
            weights = weights.to("npu:%s" % DEVICE_ID)
            actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
            actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)
            block_table = block_table.to("npu:%s" % DEVICE_ID)

            # start run custom ops
            print(
                f"======================== PTA eager BEGIN {layout_query=}, {dtype=}========================"
            )
            npu_out = torch.ops.npu.lightning_indexer(
                query,
                key,
                weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query=layout_query,
                layout_key=layout_key,
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
            )

            # compare result
            npu_out = npu_out.reshape(-1, sparse_count).cpu()
            cpuout = cpuout.reshape(-1, sparse_count).cpu()
            t = npu_out.shape[0]
            for i in range(t):
                tmp_npu_out = sorted(npu_out[i])
                tmp_cpu_out = sorted(cpuout[i])
                for j in range(sparse_count):
                    self.assertEqual(tmp_npu_out[j], tmp_cpu_out[j])
            print(
                f"======================== PTA eager FINISH {layout_query=}, {dtype=}========================"
            )

    def test_bsnd_lightning_indexer_eager(self):
        b = 1
        s1 = 1
        s2 = 8192
        n1 = 64
        n2 = 1
        d = 128
        block_size = 256
        t = 8192
        layout_query = "BSND"

        for dtype in [torch.bfloat16, torch.float16]:
            np.random.seed(0)
            query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(dtype)
            key = torch.tensor(
                np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, d))
            ).to(dtype)
            weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(dtype)
            actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(
                torch.int32
            )
            actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(
                torch.int32
            )
            block_table = torch.tensor(
                [range(b * s2 // block_size)], dtype=torch.int32
            ).reshape(b, -1)
            layout_key = "PA_BSND"
            sparse_count = 2048
            sparse_mode = 3
            cpuout = _lightning_indexer(
                query,
                key,
                weights,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table,
                layout_query,
                sparse_count,
                sparse_mode,
            )

            torch_npu.npu.set_device(int(DEVICE_ID))
            query = query.to("npu:%s" % DEVICE_ID)
            key = key.to("npu:%s" % DEVICE_ID)
            weights = weights.to("npu:%s" % DEVICE_ID)
            actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
            actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)
            block_table = block_table.to("npu:%s" % DEVICE_ID)

            # start run custom ops
            print(
                f"======================== PTA eager BEGIN {layout_query=}, {dtype=}========================"
            )
            npu_out = torch.ops.npu.lightning_indexer(
                query,
                key,
                weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query=layout_query,
                layout_key=layout_key,
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
            )

            # compare result
            npu_out = npu_out.reshape(-1, sparse_count).cpu()
            cpuout = cpuout.reshape(-1, sparse_count).cpu()
            t = npu_out.shape[0]
            for i in range(t):
                tmp_npu_out = sorted(npu_out[i])
                tmp_cpu_out = sorted(cpuout[i])
                for j in range(sparse_count):
                    self.assertEqual(tmp_npu_out[j], tmp_cpu_out[j])
            print(
                f"======================== PTA eager FINISH {layout_query=}, {dtype=}========================"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

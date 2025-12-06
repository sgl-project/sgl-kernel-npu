# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import random
import torch
import torch_npu
import torchair
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def gather_kv(k_tensor, v_tensor, sparse_indices, sparse_block_size, sparse_count, batch, n2_idx, s1_idx,
             cur_actual_seq_lengths_kv):
    s2_sparse = list()
    for sparse_id in sparse_indices:
        if sparse_id == -1: 
            break
        begin_idx = sparse_id * sparse_block_size
        end_idx = begin_idx + sparse_block_size \
                if begin_idx + sparse_block_size <= cur_actual_seq_lengths_kv else cur_actual_seq_lengths_kv
        s2_sparse.extend(np.arange(begin_idx, end_idx))

    k_sparse, v_sparse = k_tensor[batch, n2_idx, s2_sparse, :], v_tensor[batch, n2_idx, s2_sparse, :]

    return k_sparse, v_sparse


def mask(res, cur_actual_seq_q, cur_actual_seq, topk_indices, s1_idx, sparse_blocksize):
    # 求尾块ID和尾块长度
    sparse_tail_idx = np.ceil(cur_actual_seq / sparse_blocksize)
    sparse_tail_seq_len = cur_actual_seq % sparse_blocksize
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = sparse_blocksize

    delta_s = cur_actual_seq - cur_actual_seq_q
    threshold = delta_s + s1_idx + 1

    s_idx = 0
    for _, sparse_id in enumerate(topk_indices):
        if sparse_id == -1:
            break
        begin_idx = sparse_id * sparse_blocksize
        block_len = sparse_blocksize if sparse_id != sparse_tail_idx - 1 else sparse_tail_seq_len
        end_idx = begin_idx + block_len
        if begin_idx < threshold and end_idx <= threshold:
            s_idx += block_len
            continue
        elif end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len

            res[:, mask_begin: mask_end] = -1e12
        s_idx += block_len

    return res


def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans


def pa_to_bsnd(pa_in, block_table, actual_seq_lengths):
    block_num, block_size, n, d = pa_in.shape
    b = len(actual_seq_lengths)
    output = torch.zeros((b, block_num * block_size // b, 1, d)).to(pa_in.dtype)
    for i in range(b):
        for j in range(actual_seq_lengths[i] // block_size):
            output[i, j * block_size: (j + 1) * block_size, 0, :] = \
                pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
    return output


def trans_tnd_to_bsnd(tensor, shape, act_seq):
    t = shape[0]
    n = shape[1]
    d = shape[2]
    b = len(act_seq)
    s = max(act_seq)
    output = np.zeros((b, s, n, d), dtype=tensor.dtype)
    t_start = 0
    for b_idx in range(b):
        act_s = act_seq[b_idx]
        t_end = t_start + act_s
        if act_s == 0:
            continue
        for n_idx in range(n):
            output[b_idx, 0:act_s, n_idx, :] = tensor[t_start:t_end, n_idx, :]
        t_start += act_s
    return output


def trans_bnsd_to_tnd(tensor, shape, act_seq):
    t = sum(act_seq)
    b = tensor.shape[0]
    n = tensor.shape[1]
    d = tensor.shape[3]
    output = torch.full(size=(t, n, d), fill_value=-1, dtype=tensor.dtype)
    t_start = 0
    for b_idx in range(b):
        act_s = act_seq[b_idx]
        t_end = t_start + act_s
        if act_s == 0:
            continue
        for n_idx in range(n):
            output[t_start:t_end, n_idx, :] = tensor[b_idx, n_idx, :act_s, :]
        t_start += act_s
    return output


def trans_tnd_actseq(seq):
    list_len = len(seq)
    output = []
    output.append(seq[0])
    for i in range(list_len - 1):
        new_item = seq[i + 1] - seq[i]
        if new_item >= 0:
            output.append(new_item)
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{seq}, in loop {i}, item {new_item} < 0")
    return output


def cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, sparse_block_size,
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope=None, key_rope=None,
    layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None):
    query = query.cpu().to(torch.float32).numpy()
    if layout_kv == 'PA_BSND':
        key = pa_to_bsnd(key, block_table, actual_seq_lengths_kv)
        key_rope = pa_to_bsnd(key_rope, block_table, actual_seq_lengths_kv)
    key = key.cpu().to(torch.float32).numpy()
    value = key.copy()
    sparse_indices = sparse_indices.cpu().numpy()
    actual_seq_lengths_query = actual_seq_lengths_query.cpu().numpy()
    actual_seq_lengths_kv = actual_seq_lengths_kv.cpu().numpy()
    query_rope = query_rope.cpu().to(torch.float32).numpy()
    key_rope = key_rope.cpu().to(torch.float32).numpy()
    batch_size = actual_seq_lengths_query.shape[0]
    num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    sparse_count = sparse_indices.shape[-1]
    g = num_heads // num_kv_heads

    if layout_query == 'TND':
        actual_seq_lengths_query = trans_tnd_actseq(actual_seq_lengths_query)
        query = trans_tnd_to_bsnd(query, query.shape, actual_seq_lengths_query)
        query_rope = trans_tnd_to_bsnd(query_rope, query_rope.shape, actual_seq_lengths_query)
        sparse_indices = trans_tnd_to_bsnd(sparse_indices, sparse_indices.shape, actual_seq_lengths_query)

    q_bnsd_tensor = np.transpose(np.concatenate([query, query_rope], axis=-1), axes=(0, 2, 1, 3))
    k_bnsd_tensor = np.transpose(np.concatenate([key, key_rope], axis=-1), axes=(0, 2, 1, 3))
    v_bnsd_tensor = np.transpose(value, axes=(0, 2, 1, 3))
    sparse_indices = np.transpose(sparse_indices, axes=(0, 2, 1, 3))
    matmul_dtype = np.float32
    out_shape_bnsd = list(q_bnsd_tensor.shape)
    out_shape_bnsd[-1] = out_shape_bnsd[-1] - query_rope.shape[-1]
    y = np.zeros(out_shape_bnsd, dtype=np.float32)

    for batch in range(batch_size):
        cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
        cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
        for n2_idx in range(num_kv_heads):
            for s1_idx in range(cur_acutal_seq_lengths_q):
                q_curr = q_bnsd_tensor[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :]
                cur_sparse_indices = sparse_indices[batch, n2_idx, s1_idx, :]
                k_sparse, v_sparse = gather_kv(k_bnsd_tensor, v_bnsd_tensor, cur_sparse_indices, sparse_block_size,
                                              sparse_count, batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv)
                mm1_res = np.matmul(q_curr.astype(np.float32), k_sparse.astype(np.float32).T, dtype=matmul_dtype)
                scale_res = mm1_res * scale_value
                if sparse_mode == 3:
                    mask_res = mask(scale_res, cur_acutal_seq_lengths_q, cur_actual_seq_lengths_kv,
                                    cur_sparse_indices, s1_idx, sparse_block_size)
                else:
                    mask_res = scale_res
                softmax_res = softmax(mask_res)
                mm2_res = np.matmul(softmax_res, v_sparse, dtype=matmul_dtype)
                y[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :] = mm2_res

    if layout_query == 'TND':
        cpu_out = torch.tensor(y)
        return trans_bnsd_to_tnd(cpu_out, cpu_out.shape, actual_seq_lengths_query)
    return np.transpose(y, axes=(0, 2, 1, 3))


class TestCustomSFA(TestCase):
    def test_sfa_eager(self):
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        torch_npu.npu.set_device(int(DEVICE_ID))
        query_type = torch.float16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        t = 10
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        s2_act = 4096

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
        act_seq_q = [s1] * 4
        act_seq_kv = [s2_act] * 4

        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        value = value.to("npu:%s" % DEVICE_ID)
        sparse_indices = sparse_indices.to("npu:%s" % DEVICE_ID)
        query_rope = query_rope.to("npu:%s" % DEVICE_ID)
        key_rope = key_rope.to("npu:%s" % DEVICE_ID)
        act_seq_q = torch.tensor(act_seq_q).to(torch.int32).to("npu:%s" % DEVICE_ID)
        act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager BEGIN ========================')
        # start run custom ops
        npu_out = torch_npu.npu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None)

        # compare result
        cpu_out = cpu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None)
        npu_out = npu_out.cpu().to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")
        print(f'======================== PTA eager FINISH ========================')

    def test_sfa_graph(self):
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        torch_npu.npu.set_device(int(DEVICE_ID))
        query_type = torch.float16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        t = 10
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        s2_act = 4096

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
        act_seq_q = [s1] * 4
        act_seq_kv = [s2_act] * 4

        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        value = value.to("npu:%s" % DEVICE_ID)
        sparse_indices = sparse_indices.to("npu:%s" % DEVICE_ID)
        query_rope = query_rope.to("npu:%s" % DEVICE_ID)
        key_rope = key_rope.to("npu:%s" % DEVICE_ID)
        act_seq_q = torch.tensor(act_seq_q).to(torch.int32).to("npu:%s" % DEVICE_ID)
        act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).to("npu:%s" % DEVICE_ID)

        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, query, key, value, sparse_indices, scale_value, sparse_block_size,
                actual_seq_lengths_query, actual_seq_lengths_kv,
                query_rope, key_rope, layout_query, layout_kv,
                sparse_mode, block_table):
                
                out1 = torch_npu.npu_sparse_flash_attention(query, key, value,
                    sparse_indices, scale_value, sparse_block_size, 
                    actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_kv=actual_seq_lengths_kv,
                    query_rope=query_rope, key_rope=key_rope,
                    layout_query=layout_query, layout_kv=layout_kv, sparse_mode=sparse_mode, block_table=block_table)
                return out1

        mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)
        print(f'======================== PTA graph BEGIN ========================')
        # calculate on npu compile
        npu_out = mod(query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None)

        cpu_out = cpu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None)
        npu_out = npu_out.cpu().to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")
        print(f'======================== PTA graph FINISH ========================')

    # Q:TND, KV:PA_BSND
    def test_tnd_pabsnd_sfa_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        query_type = torch.float16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        t = 10
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        s2_act = 4096
        s1_max = 5

        query = torch.tensor(np.random.uniform(-10, 10, (t, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b * (s2 // block_size), block_size, n2, dn))).to(query_type)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1_max + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(t * n2)]).reshape(t, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (t, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type)
        act_seq_q = torch.tensor([1, 3, 8, 10]).to(torch.int32)
        act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32)
        block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1)

        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        value = value.to("npu:%s" % DEVICE_ID)
        sparse_indices = sparse_indices.to("npu:%s" % DEVICE_ID)
        query_rope = query_rope.to("npu:%s" % DEVICE_ID)
        key_rope = key_rope.to("npu:%s" % DEVICE_ID)
        act_seq_q = act_seq_q.to("npu:%s" % DEVICE_ID)
        act_seq_kv = act_seq_kv.to("npu:%s" % DEVICE_ID)
        block_table = block_table.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager BEGIN ========================')
        # start run custom ops
        npu_out = torch_npu.npu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table)

        # compare result
        cpu_out = cpu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table)
        npu_out = npu_out.cpu().to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")
        print(f'======================== PTA eager FINISH ========================')

    # Q:TND, KV:PA_BSND
    def test_tnd_pabsnd_sfa_graph(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        query_type = torch.float16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        t = 10
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        s2_act = 4096
        s1_max = 5

        query = torch.tensor(np.random.uniform(-10, 10, (t, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b * (s2 // block_size), block_size, n2, dn))).to(query_type)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1_max + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(t * n2)]).reshape(t, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (t, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type)
        act_seq_q = torch.tensor([1, 3, 8, 10]).to(torch.int32)
        act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32)
        block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1)

        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        value = value.to("npu:%s" % DEVICE_ID)
        sparse_indices = sparse_indices.to("npu:%s" % DEVICE_ID)
        query_rope = query_rope.to("npu:%s" % DEVICE_ID)
        key_rope = key_rope.to("npu:%s" % DEVICE_ID)
        act_seq_q = act_seq_q.to("npu:%s" % DEVICE_ID)
        act_seq_kv = act_seq_kv.to("npu:%s" % DEVICE_ID)
        block_table = block_table.to("npu:%s" % DEVICE_ID)

        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, query, key, value, sparse_indices, scale_value, sparse_block_size,
                actual_seq_lengths_query, actual_seq_lengths_kv,
                query_rope, key_rope, layout_query, layout_kv,
                sparse_mode, block_table):
                
                out1 = torch_npu.npu_sparse_flash_attention(query, key, value,
                    sparse_indices, scale_value, sparse_block_size, 
                    actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_kv=actual_seq_lengths_kv,
                    query_rope=query_rope, key_rope=key_rope,
                    layout_query=layout_query, layout_kv=layout_kv, sparse_mode=sparse_mode, block_table=block_table)
                return out1

        mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)
        print(f'======================== PTA graph BEGIN ========================')
        # calculate on npu compile
        npu_out = mod(query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table)

        cpu_out = cpu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            query_rope=query_rope, key_rope=key_rope,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table)
        npu_out = npu_out.cpu().to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")
        print(f'======================== PTA graph FINISH ========================')


if __name__ == "__main__":
    run_tests()

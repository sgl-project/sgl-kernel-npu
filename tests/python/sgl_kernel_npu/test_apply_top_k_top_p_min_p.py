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
DTYPE_ATOL = {torch.float32: 0.0001, torch.float16: 0.001, torch.bfloat16: 0.004}


def _apply_top_k_top_p_min_p(
    probs_sort,
    k,
    p,
    min_p=None,
):
    probs_sort_out = probs_sort.clone().to(torch.float32)
    top_k_mask = torch.arange(0, probs_sort.shape[-1], device=probs_sort.device).view(1, -1) >= k.view(-1, 1)
    probs_sort_out.masked_fill_(top_k_mask, 0.0)

    probs_sum = torch.cumsum(probs_sort_out, dim=-1)
    top_p_mask = probs_sum - probs_sort_out > p.view(-1, 1)
    probs_sort_out.masked_fill_(top_p_mask, 0.0)

    if min_p is not None:
        min_p_thresholds = probs_sort_out[:, 0] * min_p
        min_p_mask = probs_sort_out < min_p_thresholds.view(-1, 1)
        probs_sort_out.masked_fill_(min_p_mask, 0.0)
    return probs_sort_out.to(probs_sort.dtype)


class TestCustomApplyTopKTopPMinP(unittest.TestCase):
    def test_apply_top_k_top_p_min_p_eager(self):
        batch_size = 4
        vocab_size = 131072

        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            np.random.seed(3)
            logits = torch.tensor(np.random.uniform(-10, 10, (batch_size, vocab_size))).to(dtype)
            k = torch.tensor(np.random.randint(1, vocab_size, (batch_size))).to(torch.int32)
            p = torch.tensor(np.random.uniform(0, 1, (batch_size))).to(dtype)
            min_p = torch.tensor(np.random.uniform(0, 1, (batch_size))).to(dtype)
            probs = torch.softmax(logits, dim=-1)
            probs_sort, probs_idx = probs.sort(dim=-1, descending=True, stable=True)
            cpu_out = _apply_top_k_top_p_min_p(
                probs_sort,
                k,
                p,
                min_p
            )

            torch_npu.npu.set_device(int(DEVICE_ID))
            probs_sort = probs_sort.to("npu:%s" % DEVICE_ID)
            k = k.to("npu:%s" % DEVICE_ID)
            p = p.to("npu:%s" % DEVICE_ID)
            min_p = min_p.to("npu:%s" % DEVICE_ID)

            npu_out = torch.ops.npu.apply_top_k_top_p_min_p(
                probs_sort,
                k,
                p,
                min_p=min_p
            )

            # compare result
            npu_out = npu_out.cpu()
            cpu_out = cpu_out.cpu()
            tol = DTYPE_ATOL[dtype]
            assert torch.allclose(
                    cpu_out.to(torch.float32),
                    npu_out.to(torch.float32),
                    atol=tol,
                    rtol=tol,
                )


    def test_apply_top_k_top_p_min_p_eager_without_min_p(self):
        batch_size = 4
        vocab_size = 131072

        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            np.random.seed(4)
            logits = torch.tensor(np.random.uniform(-10, 10, (batch_size, vocab_size))).to(dtype)
            k = torch.tensor(np.random.randint(1, vocab_size, (batch_size))).to(torch.int32)
            p = torch.tensor(np.random.uniform(0, 1, (batch_size))).to(dtype)
            probs = torch.softmax(logits, dim=-1)
            probs_sort, probs_idx = probs.sort(dim=-1, descending=True, stable=True)
            cpu_out = _apply_top_k_top_p_min_p(
                probs_sort,
                k,
                p,
                min_p=None
            )

            torch_npu.npu.set_device(int(DEVICE_ID))
            probs_sort = probs_sort.to("npu:%s" % DEVICE_ID)
            k = k.to("npu:%s" % DEVICE_ID)
            p = p.to("npu:%s" % DEVICE_ID)

            npu_out = torch.ops.npu.apply_top_k_top_p_min_p(
                probs_sort,
                k,
                p,
                min_p=None
            )

            # compare result
            npu_out = npu_out.cpu()
            cpu_out = cpu_out.cpu()
            tol = DTYPE_ATOL[dtype]
            assert torch.allclose(
                    cpu_out.to(torch.float32),
                    npu_out.to(torch.float32),
                    atol=tol,
                    rtol=tol,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

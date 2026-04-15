/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef BLOCK_SPARSE_ATTENTION_EMPTY_TENSOR_H
#define BLOCK_SPARSE_ATTENTION_EMPTY_TENSOR_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"

template<typename T>
class BlockSparseAttentionEmptyTensor {
public:
    __aicore__ inline BlockSparseAttentionEmptyTensor() {};
    __aicore__ inline void Init(__gm__ uint8_t *attentionOut, const BlockSparseAttentionTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    TPipe *pipe;
    const BlockSparseAttentionTilingData* __restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template<typename T>
__aicore__ inline void BlockSparseAttentionEmptyTensor<T>::Init(__gm__ uint8_t*  attentionOut,
                                                                const BlockSparseAttentionTilingData* __restrict tiling,
                                                                TPipe *tPipe) {
    pipe = tPipe;
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void BlockSparseAttentionEmptyTensor<T>::Process() {
    uint32_t tmp_block_idx = GetBlockIdx();
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    int32_t tailSize = (int32_t)initParams.totalOutputSize - tmp_block_idx * (int32_t)initParams.singleCoreSize;
    if (tailSize > 0) {
        uint32_t singleInitOutputSize =
            tailSize < initParams.singleCoreSize ? static_cast<uint32_t>(tailSize) : initParams.singleCoreSize;
        InitOutput<T>(attentionOutGm[tmp_block_idx * (int64_t)initParams.singleCoreSize], singleInitOutputSize, 0);
    }
}
#endif  // BLOCK_SPARSE_ATTENTION_EMPTY_TENSOR_H
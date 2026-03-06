/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_expand.cpp
 */

#ifndef SGL_KERNEL_NPU_KERNEL_LORA_COMMON_H
#define SGL_KERNEL_NPU_KERNEL_LORA_COMMON_H

#include "kernel_operator.h"

namespace lora_common {

template <typename scalar_t>
class BlockIterator
{
    AscendC::GlobalTensor<scalar_t> blocks;
    int64_t previous_block;
    int64_t previous_offset;

public:
    __aicore__ explicit BlockIterator(AscendC::GlobalTensor<scalar_t> &blocks_)
        : blocks(blocks_), previous_block(0), previous_offset(0)
    {}
    __aicore__ inline int64_t GetBlockIdx(int64_t index)
    {
        int64_t current_offset = previous_offset;
        uint64_t blockIdx = previous_block;

        for (; blockIdx < blocks.GetSize(); ++blockIdx) {
            int64_t blockOffset = blocks.GetValue(blockIdx);
            if (index >= current_offset + blockOffset) {
                current_offset += blockOffset;
            } else {
                previous_offset = current_offset;
                previous_block = blockIdx;
                return blockIdx;
            }
        }

        return -1;
    }
};

}  // namespace lora_common

#endif  // SGL_KERNEL_NPU_KERNEL_LORA_COMMON_H

// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#ifndef SGL_KERNEL_NPU_LORA_COMMON_KERNEL_H
#define SGL_KERNEL_NPU_LORA_COMMON_KERNEL_H

#include "kernel_operator.h"

namespace lora_common {

// Maps a flat token index to the sequence block that owns it. The iteration
// state is kept so callers that scan tokens in ascending order pay one pass
// over seq_len. Copied from the legacy lora kernel so the old sources stay
// untouched.
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

#endif  // SGL_KERNEL_NPU_LORA_COMMON_KERNEL_H

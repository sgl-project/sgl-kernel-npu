/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
 */

#ifndef SGL_KERNEL_NPU_LASER_ATTENTION_TILING_H
#define SGL_KERNEL_NPU_LASER_ATTENTION_TILING_H

#include <cstdint>

namespace sglang::npu_kernel {

#pragma pack(push, 1)
struct LaserAttentionTilingData {
    int32_t batchSize = 0;
    int32_t headNum = 0;
    int32_t seqSize = 0;
    int32_t headDim = 0;
    int32_t coreNumPerGroup = 0;
    int32_t coreGroupNum = 0;

    int32_t qSeqLength = 0;
    int32_t kSeqLength = 0;
    int32_t vSeqLength = 0;
    int32_t maskSeqLength = 0;
    float scale = 1.0F;
    float keepProb = 1.0F;
    int32_t preTokens = 0;
    int32_t nextTokens = 1;

    int32_t isTriangle = 0;
    int32_t attenType = 0;
    int32_t sparseMode = 0;
    int32_t headGroupSize = 1;
    int32_t windowLen = 0;
    int32_t isHighPrecision = 1;
};
#pragma pack(pop)

}  // namespace sglang::npu_kernel

#endif  // SGL_KERNEL_NPU_LASER_ATTENTION_TILING_H

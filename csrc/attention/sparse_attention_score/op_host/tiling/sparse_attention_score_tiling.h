/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
 * full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_TILING_H
#define SPARSE_ATTENTION_SCORE_TILING_H

#include <cstdint>
#include "graph/types.h"
#include "sparse_attention_score_tiling_data.h"

namespace sglang {
namespace SAHost {

// tiling key constants (mirror op_kernel/sparse_attention_score_tilingkey.h, host side)
constexpr uint64_t SASA_BASE_TILING = 10000;
constexpr uint64_t SASA_FP16_D128_TILING = SASA_BASE_TILING + 1;
constexpr uint64_t SASA_BF16_D128_TILING = SASA_BASE_TILING + 2;
constexpr uint64_t SASA_FP8_D128_TILING = SASA_BASE_TILING + 3;

constexpr uint64_t SASA_BASE_ARCH22_TILING = 20000;
constexpr uint64_t SASA_FP16_D128_ARCH22_TILING = SASA_BASE_ARCH22_TILING + 1;
constexpr uint64_t SASA_BF16_D128_ARCH22_TILING = SASA_BASE_ARCH22_TILING + 2;

// ascend910_93/ascend950 map to SocVersion code 4; 910B (arch22) is any other code.
// PlatformAscendCManager::GetInstance()->GetSocVersion() returns the SocVersion enum value.
constexpr uint32_t SOC_VER_950_CODE = 4;

// All inputs the tiling computation needs. Filled by the HOST_API from at::Tensor
// shapes + function attrs + PlatformAscendCManager (no gert::TilingContext dependency).
struct SAInfo {
    // from input tensor shapes
    uint32_t batch = 0;
    uint32_t numHeads = 0;
    uint32_t kvHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t blockSize = 128;
    uint32_t topK = 16;
    uint32_t maxBlocksPerBatch = 0;
    uint32_t totalQTokens = 0;
    uint32_t maxQSeqlen = 0;
    // attrs
    float scaleValue = 0.0f;
    uint32_t innerPrecise = 0;
    // platform (from PlatformAscendCManager)
    uint32_t aicNum = 0;
    uint64_t libapiSize = 0;
    uint32_t socVer = 0;
    ge::DataType dataType = ge::DT_FLOAT16;
};

class SATiling
{
public:
    // Compute tiling data + workspace size + block dim from SAInfo. No device->host
    // reads, so cuda-graph safe. workspaceSize is the FULL workspace (libapi + user)
    // that the kernel receives; HOST_API allocates at::empty({workspaceSize}).
    void DoTiling(const SAInfo &info, SATilingData &tilingData, size_t &workspaceSize, uint32_t &blockDim);

private:
    uint64_t GenerateTilingKey(const SAInfo &info) const;
};

}  // namespace SAHost
}  // namespace sglang

#endif  // SPARSE_ATTENTION_SCORE_TILING_H

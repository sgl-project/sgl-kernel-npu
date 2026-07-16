/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe tilingData definition file
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe tilingData definition file
 */

#ifndef FUSED_DEEP_MOE_TILING_H
#define FUSED_DEEP_MOE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// #define DEBUG_SPACE
#ifdef DEBUG_SPACE
;
#else
    #define ENABLE_REUSE_MEMORY
#endif
namespace Cam {
struct WorkSpaceOffset
{
    // MM1/GMM1-Swiglu input
    int64_t shareX1TokenOffset;
    int64_t x1TokenOffset;
    int64_t shareX1ScaleOffset;
    int64_t x1ScaleOffset;
    // MM1/GMM1-Swiglu output
    int64_t shareSwigluOffset;
    int64_t swigluOffset;
    // MM2/GMM2 input
    int64_t shareX2TokenOffset;
    int64_t x2TokenOffset;
    int64_t shareX2ScaleOffset;
    int64_t x2ScaleOffset;

    int64_t shareMm1SwapSpaceOffset; // 交换空间，用于C->V数据交换
    int64_t shareMm2SwapSpaceOffset; // 交换空间，用于C->V数据交换
    int64_t gmm1SwapSpaceOffset; // 交换空间，用于C->V数据交换
    int64_t gmm2SwapSpaceOffset; // 交换空间，用于C->V数据交换
    int64_t y2TokenOffset; // 浅融合使用，已反量化无scale
    int64_t groupListOffset; // 各专家token数前缀和形式
    int64_t expandIdxOffset; // dispatch时token在远端索引
    int64_t epSendCountOffset; // 各专家从各个rank收到的token数
    int64_t reservedOffset;    // 预留空间
};

struct FusedDeepMoeInfo {
    uint32_t epRankSize;           // epRankSize
    uint32_t epRankId;             // epRankId
    uint32_t moeExpertNum;         // moe expert number
    uint32_t moeExpertNumPerRank;  // moe expert number per rank
    uint32_t quantMode;            // quant mode
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aicNum;               // aicNum
    uint32_t aivNum;               // aivNum
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    uint64_t gmm1HLen;
    uint64_t shareGmm1HLen;  // shared expert gmm1 hidden length
    bool isTensorList;
};

struct FusedDeepMoeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    FusedDeepMoeInfo fusedDeepMoeInfo;
    WorkSpaceOffset workSpaceOffset;
};

// constexpr uint32_t GM_ALIGN_BYTE = 512;
// constexpr uint32_t CUSTOM_PRELOAD_STAGES = 1;
// constexpr uint32_t CUSTOM_L1_STAGES = 2;
// constexpr uint32_t CUSTOM_L0A_STAGES = 2;
// constexpr uint32_t CUSTOM_L0B_STAGES = 2;
// constexpr uint32_t CUSTOM_L0C_STAGES = 1;
// constexpr bool CUSTOM_ENABLE_UNIT_FLAG = true;
// constexpr bool CUSTOM_ENABLE_SHUFFLE_K = true;

constexpr uint32_t GMM1_L1M = 256;
constexpr uint32_t GMM1_L1N = 256;
constexpr uint32_t GMM1_L1K = 256;
constexpr uint32_t GMM1_L0K = 128;
constexpr uint32_t GMM1_EPIM = 64;
constexpr uint32_t GMM1_SWIZZLE_OFFSET = 3;
constexpr uint32_t GMM1_SWIZZLE_DIRECTION = 0;

// constexpr uint32_t GMM2_L1A_STAGES = 4;
// constexpr uint32_t GMM2_L1B_STAGES = 2;
// constexpr uint32_t GMM2_L0A_STAGES = 4;
// constexpr uint32_t GMM2_L0B_STAGES = 2;
constexpr uint32_t GMM2_L1M = 256;
constexpr uint32_t GMM2_L1N = 256;
constexpr uint32_t GMM2_L1K = 256;
constexpr uint32_t GMM2_L0K = 128;
constexpr uint32_t GMM2_EPIM = 64;
constexpr uint32_t GMM2_SWIZZLE_OFFSET = 3;
constexpr uint32_t GMM2_SWIZZLE_DIRECTION = 0;

// constexpr uint32_t WORKSPACE_STAGES = 4;

constexpr uint32_t EXEC_FLAG_DEEP_FUSE = (1U << 0);
constexpr uint32_t EXEC_FLAG_TENSOR_LIST = (1U << 1);
constexpr uint32_t EXEC_FLAG_X_ACTIVE_MASK = (1U << 2);
constexpr uint32_t EXEC_FLAG_SHARED_EXPERT = (1U << 3);
constexpr uint32_t EXEC_FLAG_SMOOTH_QUANT = (1U << 4);

} // namespace Cam
#endif  // FUSED_DEEP_MOE_TILING_H

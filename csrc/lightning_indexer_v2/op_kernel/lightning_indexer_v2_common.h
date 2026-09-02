/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_common.h
 * \brief
 */
#ifndef SGL_LI_V2_LIGHTNING_INDEXER_COMMON_H
#define SGL_LI_V2_LIGHTNING_INDEXER_COMMON_H

#include "../op_host/tiling/lightning_indexer_v2_tiling_data.h"

// AscendC 的 kernel-direct-launch 构建会把 kernel 源码再编译一遍给 host 侧生成
// aclrtlaunch_* stub（bisheng --cce-host-only）。那一遍没有 __DAV_*/__CCE_AICORE__
// 目标宏，编译器内置的 EVENT_ID4..EVENT_ID7 也就不存在，而 arch22/arch35 的
// service 层用到了它们。stub 只需要签名，kernel body 不会在 host 上执行，因此
// 这里给出占位值；device 侧仍使用编译器内置定义。
#if !defined(__CCE_AICORE__)
#ifndef EVENT_ID4
#define EVENT_ID4 EVENT_ID0
#endif
#ifndef EVENT_ID5
#define EVENT_ID5 EVENT_ID1
#endif
#ifndef EVENT_ID6
#define EVENT_ID6 EVENT_ID2
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 EVENT_ID3
#endif
#endif

namespace sglang::npu_kernel::liv2 {
// 上游由 CANN 的 BEGIN_TILING_DATA_DEF 生成，这里改为 host/device 共用的 POD。
using LITilingData = ::sglang::LIV2Host::LITilingData;

using namespace AscendC;
namespace LICommon {

// 与tiling的layout保持一致
enum class LI_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND,
          bool DT_W_FLAG = false, typename... Args>
struct LIType {
    static constexpr bool weightsTypeFlag = DT_W_FLAG;   // weight的dtype是否为FP32
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t QLI_SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;
    uint16_t INVALID_VAL = 0;
    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t mBaseSizeAlign = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;             // topK选取大小
    uint64_t kSeqSize = 0ULL;         // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    LI_LAYOUT outputLayout;           // 输出的格式
    bool attenMaskFlag = false;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    bool returnValue = false;

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    bool isAccumSeqS1 = false;    // 是否累加模式
    bool isAccumSeqS2 = false;    // 是否累加模式
    bool isSparseCountOver2K = false; //sparseCount小于等于2048为false
    bool isLDOpen = false;
    bool returnValueFlag = false;
    bool splitMFlag = false;
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;     // 当前核是否需要进行Decode归约任务
    bool isCoreEnable = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

// 与上游的差异：上游为单模板参数 template <typename T>，依赖其构建环境里另一个
// CeilDiv 重载来吃掉 CeilDiv(uint64_t, int32_t) 这类混合类型调用；CANN 9.1.0 的
// 内核头里没有该重载，会报 "deduced conflicting types"。这里放宽为两个模板参数
// （与上面的 Max 一致），返回类型仍为第一个实参的类型，语义等价于在调用点把第二
// 个实参显式转成 T1。
template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 num, T2 rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace LICommon

// bank冲突优化
// david 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#define UB_BLOCK              32   // 32B
#define UB_BANK_GROUPS        8
#define UB_BANKS              2
#define UB_BANK_DEPTH         512

#define UB_BANK_GROUP_STRIDE  UB_BLOCK                                   // 32B
#define UB_BANK_STRIDE        (UB_BANK_GROUPS * UB_BLOCK)               // 256B
#define UB_BANK_DEPTH_STRIDE  (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)    // 512B

}  // namespace sglang::npu_kernel::liv2
#endif // SGL_LI_V2_LIGHTNING_INDEXER_COMMON_H

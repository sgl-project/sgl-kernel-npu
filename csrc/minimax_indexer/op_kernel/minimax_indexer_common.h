/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file minimax_indexer_common.h
 * \brief
 */
#ifndef MINIMAX_INDEXER_COMMON_H
#define MINIMAX_INDEXER_COMMON_H

namespace sglang::npu_kernel::MICommon {

// consistent with tiling layout
enum class LI_LAYOUT { BSND = 0, TND = 1, PA_BSND = 2 };

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND, typename... Args>
struct MIType {
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
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
};

struct ConstInfo {
    // CUBE-VEC cross-core sync mode
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // buffer sizes in bytes
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
    // invalid index
    static constexpr int INVALID_IDX = -1;

    // CUBE-VEC cross-core sync EventIDs
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C1 = 0U;

    // base-block sizes
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;              // topK selection count
    uint64_t kSeqSize = 0ULL;          // max KV sequence length
    uint64_t qSeqSize = 1ULL;          // max query sequence length
    uint32_t kCacheBlockSize = 0;      // PA block size
    uint32_t maxBlockNumPerBatch = 0;  // max blocks per batch in PA
    LI_LAYOUT outputLayout;            // output format
    bool attenMaskFlag = false;
    // MiniMax indexer extras
    uint32_t initBlocks = 0;    // sink/init sentinel block count
    uint32_t localBlocks = 0;   // local (recent) sentinel block count
    float smScaleLog2e = 0.0f;  // sm_scale * log2(e), applied after max-reduce
    // Fused-interface extras (host-injected, see minimax_indexer.cpp):
    // 1 = gather the logical->physical block table from req_to_token inside the
    // kernel (direct page lookup), skipping the host-side arange+gather+div.
    uint32_t directMode = 0;
    uint32_t maxTokenSlots = 0;  // req_to_token.shape[1]: token-slot width for the direct gather
    // 1 = append the causal local block at slot topk of each (b, head) output row
    // (output width topk+1, GQA-kernel layout) and emit the final indices in the
    // [QH, B, topk(+1)] memory layout the sparse-attention kernels consume directly.
    uint32_t appendLocal = 0;
    // 1 = packed EAGLE3 verify: actual_seq_lengths_key carries the full [B*gqa]
    // per-row causal lengths. The kernel scores the shared block space (bounded by
    // the per-batch row max) and masks the boundary block per row.
    uint32_t packedMode = 0;

    uint32_t actualLenQDims = 0U;  // actualSeqLength query dimensions
    uint32_t actualLenDims = 0U;   // KV actualSeqLength dimensions
    bool isAccumSeqS1 = false;     // prefix-sum (accumulated) mode
    bool isAccumSeqS2 = false;     // prefix-sum (accumulated) mode
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;  // S2 start position
    uint32_t s2End = 0U;    // S2 loop index upper bound
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;  // whether this core handles the LD merge
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

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
}  // namespace sglang::npu_kernel::MICommon

#endif  // MINIMAX_INDEXER_COMMON_H

/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef MINIMAX_TILING_DATA_H
#define MINIMAX_TILING_DATA_H
#include <cstdint>

namespace sglang {
namespace MIHost {

// ----------- TopK-API raw tiling mirror ---------------
// Plain-POD byte mirror of the kernel-side TopkTiling (adv_api/kernel_tiling.h),
// which is itself a 112-byte pack(8) struct. Field order/names/types match the
// host optiling::TopkTiling (BEGIN_TILING_DATA_DEF in adv_api/sort/topk_tilingdata.h)
// exactly, so a flat SaveToBuffer copy on host == a reinterpret on kernel.
// MITilingData lives under pack(1); TopkTiling's fields are all 4-byte/2-byte with
// no implicit padding, so pack(1) and pack(8) yield identical layout here.
// Verified at kernel build by static_assert(sizeof == sizeof(kernel TopkTiling)).
#pragma pack(push, 1)
struct MITopkTilingRaw {
    int32_t tmpLocalSize = 0;
    int32_t allDataSize = 0;
    int32_t innerDataSize = 0;
    uint32_t sortRepeat = 0;
    int32_t mrgSortRepeat = 0;
    int32_t kAlignFourBytes = 0;
    int32_t kAlignTwoBytes = 0;
    int32_t maskOffset = 0;
    int32_t maskVreducev2FourBytes = 0;
    int32_t maskVreducev2TwoBytes = 0;
    int32_t mrgSortSrc1offset = 0;
    int32_t mrgSortSrc2offset = 0;
    int32_t mrgSortSrc3offset = 0;
    int32_t mrgSortTwoQueueSrc1Offset = 0;
    int32_t mrgFourQueueTailPara1 = 0;
    int32_t mrgFourQueueTailPara2 = 0;
    int32_t srcIndexOffset = 0;
    uint32_t copyUbToUbBlockCount = 0;
    int32_t topkMrgSrc1MaskSizeOffset = 0;
    int32_t topkNSmallSrcIndexOffset = 0;
    uint32_t vreduceValMask0 = 0;
    uint32_t vreduceValMask1 = 0;
    uint32_t vreduceIdxMask0 = 0;
    uint32_t vreduceIdxMask1 = 0;
    uint16_t vreducehalfValMask0 = 0;
    uint16_t vreducehalfValMask1 = 0;
    uint16_t vreducehalfValMask2 = 0;
    uint16_t vreducehalfValMask3 = 0;
    uint16_t vreducehalfValMask4 = 0;
    uint16_t vreducehalfValMask5 = 0;
    uint16_t vreducehalfValMask6 = 0;
    uint16_t vreducehalfValMask7 = 0;
};

// ----------- TilingData Definition ---------------
struct MITilingData {
    uint32_t bSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t gSize = 0U;
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    uint32_t sparseCount = 0U;  // MiniMax: topk (e.g. 16)
    uint32_t usedCoreNum = 0U;
    uint32_t blockSize = 0U;  // MiniMax: KV block_size; also == s2BaseSize (one Cube tile = one block)
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t sparseMode = 0U;
    uint32_t tilingKey = 0U;
    // MiniMax indexer extras (injected by HOST_API, not parsed by LIInfoParser)
    uint32_t initBlocks = 0U;   // sink/init sentinel block count
    uint32_t localBlocks = 0U;  // local (recent) sentinel block count
    float smScaleLog2e = 0.0f;  // sm_scale * log2(e), applied to max-reduced block score
    // Fused-interface extras (injected by HOST_API):
    uint32_t directMode = 0U;     // 1: gather block table from req_to_token in-kernel
    uint32_t maxTokenSlots = 0U;  // req_to_token.shape[1] (direct gather token-slot width)
    uint32_t appendLocal = 0U;    // 1: output topk+1 with causal local block at slot topk
    uint32_t packedMode = 0U;     // 1: actual_seq_lengths_key is [B*gqa] per-row causal
    MITopkTilingRaw topkTiling{};
    uint32_t topkTmpSize = 0U;
};
#pragma pack(pop)
}  // namespace MIHost
}  // namespace sglang
#endif

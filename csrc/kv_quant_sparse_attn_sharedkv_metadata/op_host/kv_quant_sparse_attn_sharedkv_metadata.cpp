/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 *
 * Reimplemented on the host CPU with reference to the vllm-ascend AICPU op
 * `KvQuantSparseAttnSharedkvMetadata` (csrc/attention/kv_quant_sparse_attn_sharedkv_metadata).
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_metadata.cpp
 * \brief Host-side core-distribution scheduler adapted from the existing
 * pure-host shared-KV metadata implementation for the A5 quantized metadata
 * layout and block sizes.
 */

#include "kv_quant_sparse_attn_sharedkv_metadata.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <limits>
#include <string>
#include <vector>

#ifndef KV_QUANT_METADATA_CORE_ONLY
#include <ATen/ATen.h>
#include "acl/acl.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sgl_kernel_npu::kv_quant {

constexpr int64_t ROW_PARALLEL_THRESHOLD = 4096;
constexpr int ROW_PARALLEL_MAX_THREADS = 16;

bool KvQuantSparseAttnSharedkvMetadataHost::Run(const int32_t *cuSeqLenQ, const int32_t *seqUsedKv, int32_t batchSize,
                                                int32_t maxSeqLenQ, int32_t maxSeqLenKv, int32_t queryHeadNum,
                                                int32_t kvHeadNum, int32_t headDim, uint32_t aicCoreNum,
                                                uint32_t aivCoreNum, const std::string &socVersion, int32_t cmpTopK,
                                                int32_t cmpRatio, int32_t oriMaskMode, int32_t cmpMaskMode,
                                                int64_t winLeft, int64_t winRight, const std::string &layoutQuery,
                                                const std::string &layoutKv, bool hasOriKv, bool hasCmpKv,
                                                int32_t *metaData)
{
    cacheReady_ = false;
    actSeqLenQ_ = cuSeqLenQ;
    seqUsedKv_ = seqUsedKv;
    batchSize_ = batchSize;
    maxSeqLenQ_ = maxSeqLenQ;
    maxSeqLenKv_ = maxSeqLenKv;
    queryHeadNum_ = queryHeadNum;
    kvHeadNum_ = kvHeadNum;
    headDim_ = headDim;
    aicCoreNum_ = aicCoreNum;
    aivCoreNum_ = aivCoreNum;
    socVersion_ = socVersion;
    cmpTopK_ = cmpTopK;
    cmpRatio_ = cmpRatio;
    oriMaskMode_ = oriMaskMode;
    cmpMaskMode_ = cmpMaskMode;
    winLeft_ = winLeft;
    winRight_ = winRight;
    layoutQuery_ = layoutQuery;
    layoutKv_ = layoutKv;
    hasOriKv_ = hasOriKv;
    hasCmpKv_ = hasCmpKv;
    metaData_ = metaData;

    if (batchSize_ <= 0 || maxSeqLenQ_ < 0 || maxSeqLenKv_ < 0 || aicCoreNum_ == 0U || aivCoreNum_ == 0U ||
        aicCoreNum_ > kv_quant_optiling::AIC_CORE_NUM || aivCoreNum_ > kv_quant_optiling::AIV_CORE_NUM ||
        metaData_ == nullptr || ProcessSocVersion() != ValidSocVersion::ASCEND950) {
        return false;
    }
    if ((queryHeadNum_ != 64 && queryHeadNum_ != 128) || kvHeadNum_ != 1 || headDim_ != 512 ||
        (layoutQuery_ != "BSND" && layoutQuery_ != "TND") || layoutKv_ != "PA_ND" ||
        (queryHeadNum_ == 128 && ((aicCoreNum_ & 1U) || (aivCoreNum_ & 1U)))) {
        return false;
    }
    if (hasCmpKv) {
        if (cmpTopK != 0 && cmpTopK != 512 && cmpTopK != 1024) {
            return false;
        }
        if (cmpRatio != 4 && cmpRatio != 128) {
            return false;
        }
    }
    isCFA_ = false;
    isSCFA_ = false;
    isN128_ = false;
    if (!ParamsInit()) {
        return false;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    return BalanceSchedule(splitRes) && GenMetaData(splitRes);
}

ValidSocVersion KvQuantSparseAttnSharedkvMetadataHost::ProcessSocVersion()
{
    // Case-insensitive: aclrtGetSocName() casing is not guaranteed stable across SoCs.
    std::string lower;
    lower.reserve(socVersion_.size());
    for (char c : socVersion_) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower.find("ascend950") != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    }
    return ValidSocVersion::ASCEND910;
}

bool KvQuantSparseAttnSharedkvMetadataHost::ParamsInit()
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::DEFAULT_MASK) {
        preToken_ = INT64_MAX;
        nextToken_ = INT64_MAX;
        attentionMode_ = 0;
    } else if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        preToken_ = INT64_MAX;
        nextToken_ = 0;
        attentionMode_ = 1;
    } else {  // SparseMode = 4 (BAND)
        preToken_ = (winLeft_ > -1) ? winLeft_ : INT64_MAX;
        nextToken_ = winRight_;
        attentionMode_ = 1;
    }
    isS1G_ = (layoutQuery_ == "BSND" || layoutQuery_ == "BSH" || layoutQuery_ == "TND");
    groupSize_ = queryHeadNum_ / kvHeadNum_;
    if (hasCmpKv_) {
        if (cmpTopK_ > 0) {
            isSCFA_ = true;
        } else {
            isCFA_ = true;
        }
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    isN128_ = (queryHeadNum_ == 128);
    if (validSocVersion != ValidSocVersion::ASCEND950) {
        return false;
    }
    if (isN128_) {
        mBaseSize_ = groupSize_;
        aicCoreNum_ /= 2;
        aivCoreNum_ /= 2;
    } else {
        mBaseSize_ = isSCFA_ ? groupSize_ : (64U / groupSize_) * groupSize_;
    }
    s2BaseSize_ = 128U;
    return true;
}

uint32_t KvQuantSparseAttnSharedkvMetadataHost::GetS1SeqSize(uint32_t bIdx)
{
    if (layoutQuery_ == "TND" && actSeqLenQ_ != nullptr) {
        return static_cast<uint32_t>(actSeqLenQ_[bIdx + 1U] - actSeqLenQ_[bIdx]);
    }
    return static_cast<uint32_t>(maxSeqLenQ_);
}

uint32_t KvQuantSparseAttnSharedkvMetadataHost::GetS2SeqSize(uint32_t bIdx)
{
    if (seqUsedKv_ != nullptr) {
        return static_cast<uint32_t>(seqUsedKv_[bIdx]);
    }
    return static_cast<uint32_t>(maxSeqLenKv_);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcSplitInfo(SplitContext &splitContext)
{
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (s1Size * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (s1Size * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::WinCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t winAlignCoefM = 16U;
    uint32_t winAlignCoefS2 = 64U;
    uint32_t winAlignBasicM = (basicM + winAlignCoefM - 1U) >> 4U;
    uint32_t winAlignBasicS2 = (basicS2 + winAlignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * winAlignBasicM + 10U * winAlignBasicS2);
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CmpCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) >> 4U;
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * cmpAlignBasicM + 10U * cmpAlignBasicS2);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                          uint32_t s1GTailSize, uint32_t winS2TailSize,
                                                          uint32_t cmpS2TailSize, BlockCost<int64_t> &typeCost)
{
    typeCost[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] = WinCalcCost(s1NormalSize, s2NormalSize);
    typeCost[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, s2NormalSize);
    typeCost[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] = (winS2TailSize == 0U) ? 0U : WinCalcCost(s1NormalSize, winS2TailSize);
    typeCost[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] =
        (s1GTailSize == 0U || winS2TailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, winS2TailSize);
    if (hasCmpKv_) {
        typeCost[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = CmpCalcCost(s1NormalSize, s2NormalSize);
        typeCost[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, s2NormalSize);
        typeCost[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] =
            (cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1NormalSize, cmpS2TailSize);
        typeCost[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] =
            (s1GTailSize == 0U || cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, cmpS2TailSize);
    }
}

Range<int64_t> KvQuantSparseAttnSharedkvMetadataHost::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache)
{
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(0, 0);
    }
    if (!attentionMode_) {
        return std::make_pair(0, static_cast<int64_t>(batchCache.s2Size) - 1);
    }
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
                           1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }
    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;
    return std::make_pair(s2FirstToken, s2LastToken);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                           BatchCache &batchCache)
{
    (void)splitContext;
    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcWinS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo,
                                                            const BlockCost<int64_t> &typeCost)
{
    if (s1GCache.winS2Start >= s1GCache.winS2End) {
        s1GCache.winS1GBlock = 0;
        s1GCache.winS1GCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
    } else {
        s1GCache.winS1GBlock = s1GCache.winS2End - s1GCache.winS2Start;
        uint32_t curWinTailS2Num = (s1GCache.winS2TailSize != 0U) ? 1U : 0U;
        uint32_t curWinNormalS2Num = s1GCache.winS1GBlock - curWinTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.winS1GCost = typeCost[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                                  typeCost[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK]
                                                                : typeCost[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
        } else {
            s1GCache.winS1GCost = typeCost[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                                  typeCost[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK]
                                                                : typeCost[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
        }
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo,
                                                            const BlockCost<int64_t> &typeCost)
{
    if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
        s1GCache.cmpS1GBlock = 0;
        s1GCache.cmpS1GCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
    } else {
        s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
        uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U) ? 1U : 0U;
        uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.cmpS1GCost = typeCost[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                                  typeCost[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK]
                                                                : typeCost[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
        } else {
            s1GCache.cmpS1GCost = typeCost[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                                  typeCost[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK]
                                                                : typeCost[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
        }
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange,
                                                                      const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.s2Size) || oriS2LastToken < 0 ||
        oriS2LastToken < oriS2FirstToken) {
        oriS2FirstToken = 0;
        oriS2LastToken = 0;
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = 0;
        s1GCache.winS2TailSize = 0;
    } else {
        oriS2FirstToken = Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = (oriS2LastToken - oriS2FirstToken) / s2BaseSize_ + 1U;
        s1GCache.winS2TailSize = (oriS2LastToken - oriS2FirstToken + 1) % s2BaseSize_;
    }
    s1GCache.cmpS2Start = s1GCache.winS2End;
    uint32_t cmpS2LastTokenSize = hasCmpKv_ ? (oriS2LastToken + 1) / cmpRatio_ : 0;
    uint32_t actCmpS2LastTokenSize = 0;
    if (isCFA_) {
        actCmpS2LastTokenSize = cmpS2LastTokenSize;
    } else if (isSCFA_) {
        actCmpS2LastTokenSize = std::min(cmpS2LastTokenSize, static_cast<uint32_t>(cmpTopK_));
    }
    s1GCache.cmpS2End = (actCmpS2LastTokenSize == 0)
                            ? s1GCache.cmpS2Start
                            : s1GCache.cmpS2Start + (actCmpS2LastTokenSize - 1) / s2BaseSize_ + 1U;
    s1GCache.cmpS2TailSize = actCmpS2LastTokenSize % s2BaseSize_;
}

void KvQuantSparseAttnSharedkvMetadataHost::GatherWinAndCmpCache(S1GCache &s1GCache)
{
    s1GCache.s2Start = (s1GCache.winS1GBlock > 0) ? s1GCache.winS2Start : s1GCache.cmpS2Start;
    if (s1GCache.cmpS1GBlock > 0) {
        s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
        s1GCache.s2End = s1GCache.cmpS2End;
    } else {
        s1GCache.s1GLastBlockCost = s1GCache.winS1GLastBlockCost;
        s1GCache.s2End = s1GCache.winS2End;
    }
    s1GCache.s1GBlock = s1GCache.winS1GBlock + s1GCache.cmpS1GBlock;
    s1GCache.s1GCost = s1GCache.winS1GCost + s1GCache.cmpS1GCost;
}

// Compute one row's cost state with a caller-provided typeCost scratch.
void KvQuantSparseAttnSharedkvMetadataHost::ComputeS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                            const BatchCache &batchCache, BlockCost<int64_t> &typeCost,
                                                            S1GCache &out)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        out.s1GCost = 0;
        out.s1GLastBlockCost = 0;
        out.winS1GNormalBlockCost = 0;
        out.winS1GLastBlockCost = 0;
        out.cmpS1GNormalBlockCost = 0;
        out.cmpS1GLastBlockCost = 0;
        out.s1GBlock = 0;
        out.s2Start = 0;
        out.cmpS2Start = 0;
        out.s2End = 0;
        return;
    }
    out.bIdx = batchCache.bIdx;
    out.s1GIdx = s1GIdx;
    auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    CalcBlockRangeAndTailSize(oriS2TokenRange, batchCache, out);
    CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[out.bIdx], out.winS2TailSize, out.cmpS2TailSize,
                  typeCost);
    CalcWinS1GCache(out, splitInfo, typeCost);
    CalcCmpS1GCache(out, splitInfo, typeCost);
    GatherWinAndCmpCache(out);
}

void KvQuantSparseAttnSharedkvMetadataHost::BuildRowCache(const SplitContext &splitContext)
{
    const SplitInfo &si = splitContext.splitInfo;
    rowOffset_.assign(static_cast<size_t>(batchSize_) + 1, 0U);
    for (int b = 0; b < batchSize_; b++) {
        rowOffset_[b + 1] = rowOffset_[b] + si.s1GBaseNum[b] + 1U;  // +1 phantom slot per batch
    }
    uint32_t total = rowOffset_[batchSize_];
    rowCache_.assign(total, S1GCache{});
    batchCacheArr_.assign(batchSize_, BatchCache{});
    for (int b = 0; b < batchSize_; b++) {
        CalcBatchCache(static_cast<uint32_t>(b), splitContext, batchCacheArr_[b]);
    }
    // slot -> batch lookup
    std::vector<uint32_t> slotB(total);
    for (int b = 0; b < batchSize_; b++) {
        for (uint32_t g = rowOffset_[b]; g < rowOffset_[b + 1]; g++)
            slotB[g] = static_cast<uint32_t>(b);
    }
    cacheReady_ = true;
    auto worker = [&](int g) {
        uint32_t b = slotB[g];
        uint32_t j = static_cast<uint32_t>(g) - rowOffset_[b];
        BlockCost<int64_t> tc;  // per-iteration scratch -> thread-safe
        ComputeS1GCache(j, splitContext, batchCacheArr_[b], tc, rowCache_[g]);
    };
    if (total >= ROW_PARALLEL_THRESHOLD) {
        int nt = static_cast<int>(total) / 512;
        if (nt < 1) {
            nt = 1;
        }
        if (nt > ROW_PARALLEL_MAX_THREADS) {
            nt = ROW_PARALLEL_MAX_THREADS;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
        for (int g = 0; g < static_cast<int>(total); g++)
            worker(g);
    } else {
        for (uint32_t g = 0; g < total; g++)
            worker(static_cast<int>(g));
    }
}

const S1GCache &KvQuantSparseAttnSharedkvMetadataHost::GetS1GCache(uint32_t s1GIdx, const BatchCache &batchCache)
{
    uint32_t base = rowCache_.empty() ? 0U : (rowOffset_[batchCache.bIdx + 1] - rowOffset_[batchCache.bIdx] - 1U);
    uint32_t idx = rowOffset_[batchCache.bIdx] + (s1GIdx < base ? s1GIdx : base);
    return rowCache_[idx];
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext,
                                                          CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;
    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        const S1GCache &r = rowCache_[rowOffset_[bIdx] + s1GIdx];
        costInfo.bN2CostOfEachBatch[bIdx] += r.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += r.s1GBlock;
        if (r.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = r.s1GCost;
        }
        if (r.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = r.s1GLastBlockCost;
        }
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcCostInfo(SplitContext &splitContext)
{
    CostInfo &costInfo = splitContext.costInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * kvHeadNum_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * kvHeadNum_;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignByBatch(const SplitContext &splitContext,
                                                          AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;
        if (assignContext.curBN2Idx == static_cast<uint32_t>(batchSize_) * kvHeadNum_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }
        if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        assignContext.s1GPtr = &GetS1GCache(assignContext.curS1GIdx, assignContext.batchCache);
        assignContext.curS2Idx = assignContext.s1GPtr->s2Start;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GPtr->s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GPtr->s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GPtr->s1GCost;
        assignContext.coreCache.block += assignContext.s1GPtr->s1GBlock;
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GPtr->s1GCost
                                    ? assignContext.bN2Cost - assignContext.s1GPtr->s1GCost
                                    : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GPtr->s1GBlock
                                     ? assignContext.bN2Block - assignContext.s1GPtr->s1GBlock
                                     : 0U;
        do {
            assignContext.curS1GIdx++;
            assignContext.s1GPtr = &GetS1GCache(assignContext.curS1GIdx, assignContext.batchCache);
        } while (assignContext.s1GPtr->s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GPtr->s2Start;
    }
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CalcCurBlockCost(AssignContext &assignContext)
{
    int64_t curCost = 0;
    if (assignContext.curS2Idx < assignContext.s1GPtr->cmpS2Start) {
        curCost = assignContext.s1GPtr->winS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GPtr->cmpS2Start - 1U)) {
            curCost = assignContext.s1GPtr->winS1GLastBlockCost;
        }
    } else {
        curCost = assignContext.s1GPtr->cmpS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GPtr->s2End - 1U)) {
            curCost = assignContext.s1GPtr->cmpS1GLastBlockCost;
        }
    }
    return curCost;
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignByBlock(const SplitContext &splitContext,
                                                          AssignContext &assignContext)
{
    if (assignContext.isFinished || !supportFd_) {
        return;
    }
    // supportFd path (inert: supportFd_ is always false). Operate on a local mutable copy.
    S1GCache sc = (assignContext.s1GPtr != nullptr) ? *assignContext.s1GPtr : S1GCache{};
    int64_t curCost = CalcCurBlockCost(assignContext);
    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        sc.s1GCost = sc.s1GCost - curCost;
        assignContext.bN2Block--;
        sc.s1GBlock--;
        curCost = CalcCurBlockCost(assignContext);
    }
    (void)sc;
}

bool KvQuantSparseAttnSharedkvMetadataHost::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                               const SplitResult &splitRes)
{
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void KvQuantSparseAttnSharedkvMetadataHost::RecordFDInfo(const SplitContext &splitContext,
                                                         const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / kvHeadNum_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U)
                                ? (s1Size * groupSize_ - splitS1GIdx * mBaseSize_)
                                : mBaseSize_;
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.curFdDataNum - assignContext.curKvSplitPart;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void KvQuantSparseAttnSharedkvMetadataHost::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;
    bool UpdateS1G = false;
    bool UpdateBatch = false;
    if (assignContext.curS2Idx >= assignContext.s1GPtr->s2End) {
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }
    if (assignContext.curBN2Idx == static_cast<uint32_t>(batchSize_) * kvHeadNum_) {
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }
    if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        assignContext.s1GPtr = &GetS1GCache(assignContext.curS1GIdx, assignContext.batchCache);
        assignContext.curS2Idx = (supportFd_) ? assignContext.s1GPtr->winS2Start : 0;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    int64_t curCost = CalcCurBlockCost(assignContext);
    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    if (assignContext.s1GPtr != nullptr) {
        S1GCache sc = *assignContext.s1GPtr;
        sc.s1GCost = sc.s1GCost - curCost;
        sc.s1GBlock--;
        (void)sc;
    }
    UpdateCursor(splitContext, assignContext);
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignBlocksToCore(const SplitContext &splitContext,
                                                               AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    int64_t avgCost = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    assignContext.coreCache = {};
    if (!supportFd_) {
        assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
    } else {
        assignContext.coreCache.costLimit = avgCost;
    }
    AssignByBatch(splitContext, assignContext);
    AssignByRow(splitContext, assignContext);
    AssignByBlock(splitContext, assignContext);
    if (assignContext.coreCache.block == 0 && supportFd_) {
        ForceAssign(splitContext, assignContext);
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    result.maxS2GBaseNum = std::max(result.maxS2GBaseNum, assignContext.coreCache.block);
    if (IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.curKvSplitPart = 1U;
    }
    if (assignContext.curS2Idx > assignContext.s1GPtr->s2Start &&
        assignContext.curS2Idx <= assignContext.s1GPtr->s2End) {
        assignContext.curKvSplitPart++;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext,
                                                          SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    assignContext.s1GPtr = &GetS1GCache(assignContext.curS1GIdx, assignContext.batchCache);
    assignContext.curS2Idx = assignContext.s1GPtr->s2Start;
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlocksToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void KvQuantSparseAttnSharedkvMetadataHost::SplitFD(SplitResult &splitRes)
{
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    uint64_t averageLoad = totalFDLoad / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        uint32_t curAveMSize = splitRes.fdRes.fdMSize[i] / curFDVectorNum;
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool KvQuantSparseAttnSharedkvMetadataHost::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(static_cast<uint32_t>(batchSize_));
    CalcSplitInfo(splitContext);
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = static_cast<uint32_t>(batchSize_) * kvHeadNum_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    BuildRowCache(splitContext);
    CalcCostInfo(splitContext);
    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataHost::GenMetaData(SplitResult &splitRes)
{
    kv_quant_optiling::detail::SasMetaData *metaDataPtr =
        reinterpret_cast<kv_quant_optiling::detail::SasMetaData *>(metaData_);
    auto writeCore = [&](size_t physicalCore, size_t logicalCore) {
        auto &entry = metaDataPtr->faMetadata[physicalCore];
        entry[kv_quant_optiling::FA_CORE_ENABLE_INDEX] = logicalCore < splitRes.usedCoreNum ? 1U : 0U;
        if (isN128_) {
            entry[kv_quant_optiling::FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;
        }
        if (logicalCore >= splitRes.usedCoreNum) {
            return;
        }
        entry[kv_quant_optiling::FA_BN2_START_INDEX] = logicalCore == 0 ? 0U : splitRes.bN2End[logicalCore - 1];
        entry[kv_quant_optiling::FA_M_START_INDEX] = logicalCore == 0 ? 0U : splitRes.gS1End[logicalCore - 1];
        entry[kv_quant_optiling::FA_S2_START_INDEX] = logicalCore == 0 ? 0U : splitRes.s2End[logicalCore - 1];
        entry[kv_quant_optiling::FA_BN2_END_INDEX] = splitRes.bN2End[logicalCore];
        entry[kv_quant_optiling::FA_M_END_INDEX] = splitRes.gS1End[logicalCore];
        entry[kv_quant_optiling::FA_S2_END_INDEX] = splitRes.s2End[logicalCore];
        entry[kv_quant_optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[logicalCore];
    };
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (isN128_) {
            writeCore(2 * i, i);
            writeCore(2 * i + 1, i);
        } else {
            writeCore(i, i);
        }
    }
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_CORE_ENABLE_INDEX] = 1;
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metaDataPtr->fdMetadata[i][kv_quant_optiling::FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}

}  // namespace sgl_kernel_npu::kv_quant

#ifndef KV_QUANT_METADATA_CORE_ONLY
namespace sglang::npu_kernel {
namespace {
struct HostTopology {
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    std::string socVersion;
};

HostTopology ResolveHostTopology()
{
    int32_t device = 0;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS,
                "kv_quant_sparse_attn_sharedkv_metadata_host: no current device");
    int64_t aic = 0;
    int64_t aiv = 0;
    TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_AI_CORE_NUM, &aic) == ACL_SUCCESS,
                "kv_quant_sparse_attn_sharedkv_metadata_host: AI core count query failed");
    TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv) == ACL_SUCCESS,
                "kv_quant_sparse_attn_sharedkv_metadata_host: vector core count query failed");
    TORCH_CHECK(aic > 0 && aic <= kv_quant_optiling::AIC_CORE_NUM && aiv > 0 && aiv <= kv_quant_optiling::AIV_CORE_NUM,
                "kv_quant_sparse_attn_sharedkv_metadata_host: unsupported core topology: AI cores=", aic,
                ", vector cores=", aiv);
    const char *soc = aclrtGetSocName();
    TORCH_CHECK(soc != nullptr, "kv_quant_sparse_attn_sharedkv_metadata_host: SoC name query failed");
    return {static_cast<uint32_t>(aic), static_cast<uint32_t>(aiv), std::string(soc)};
}

const int32_t *CheckHostInt32(const c10::optional<at::Tensor> &input, int64_t expectedSize, const char *name)
{
    if (!input.has_value()) {
        return nullptr;
    }
    const at::Tensor &tensor = *input;
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == at::kInt && tensor.is_contiguous() &&
                    tensor.dim() == 1 && tensor.numel() == expectedSize,
                "kv_quant_sparse_attn_sharedkv_metadata_host: ", name, " must be a contiguous CPU int32 vector with ",
                expectedSize, " elements");
    return static_cast<const int32_t *>(tensor.const_data_ptr());
}
}  // namespace

at::Tensor kv_quant_sparse_attn_sharedkv_metadata_host(int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
                                                       const std::string &layout_q, const std::string &layout_kv,
                                                       const c10::optional<at::Tensor> &cu_seqlens_q,
                                                       const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size,
                                                       int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t cmp_topk,
                                                       int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
                                                       int64_t ori_win_left, int64_t ori_win_right, bool has_ori_kv,
                                                       bool has_cmp_kv, int64_t kv_quant_mode, int64_t tile_size,
                                                       int64_t rope_head_dim)
{
    TORCH_CHECK(kv_quant_mode == 1 && tile_size == 64 && rope_head_dim == 64,
                "kv_quant_sparse_attn_sharedkv_metadata_host: expected kv_quant_mode=1, tile_size=64, "
                "rope_head_dim=64");
    constexpr auto MAX_I32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    int64_t resolvedBatchSize = batch_size;
    if (resolvedBatchSize == 0 && cu_seqlens_q.has_value()) {
        resolvedBatchSize = cu_seqlens_q->numel() - 1;
    } else if (resolvedBatchSize == 0 && seqused_kv.has_value()) {
        resolvedBatchSize = seqused_kv->numel();
    }
    TORCH_CHECK(resolvedBatchSize > 0 && resolvedBatchSize <= MAX_I32,
                "kv_quant_sparse_attn_sharedkv_metadata_host: batch_size must be in [1, INT32_MAX], got ",
                resolvedBatchSize);
    TORCH_CHECK(max_seqlen_q >= 0 && max_seqlen_q <= MAX_I32,
                "kv_quant_sparse_attn_sharedkv_metadata_host: max_seqlen_q must be in [0, INT32_MAX], got ",
                max_seqlen_q);
    TORCH_CHECK(max_seqlen_kv >= 0 && max_seqlen_kv <= MAX_I32,
                "kv_quant_sparse_attn_sharedkv_metadata_host: max_seqlen_kv must be in [0, INT32_MAX], got ",
                max_seqlen_kv);
    TORCH_CHECK(ori_win_left >= 0 && ori_win_left <= MAX_I32,
                "kv_quant_sparse_attn_sharedkv_metadata_host: ori_win_left must be in [0, INT32_MAX], got ",
                ori_win_left);
    TORCH_CHECK(ori_mask_mode == 0 || ori_mask_mode == 3 || ori_mask_mode == 4,
                "kv_quant_sparse_attn_sharedkv_metadata_host: ori_mask_mode must be 0, 3 or 4, got ", ori_mask_mode);
    TORCH_CHECK(num_heads_q > 0 && num_heads_q <= MAX_I32 && num_heads_kv > 0 && num_heads_kv <= MAX_I32 &&
                    head_dim > 0 && head_dim <= MAX_I32 && cmp_topk >= 0 && cmp_topk <= MAX_I32 && cmp_ratio >= -1 &&
                    cmp_ratio <= MAX_I32 && cmp_mask_mode >= 0 && cmp_mask_mode <= MAX_I32 && ori_win_right >= 0 &&
                    ori_win_right <= MAX_I32,
                "kv_quant_sparse_attn_sharedkv_metadata_host: invalid scalar argument");
    TORCH_CHECK(layout_q == "BSND" || layout_q == "TND",
                "kv_quant_sparse_attn_sharedkv_metadata_host: layout_q must be BSND or TND");
    TORCH_CHECK(layout_kv == "PA_ND", "kv_quant_sparse_attn_sharedkv_metadata_host: layout_kv must be PA_ND");
    TORCH_CHECK(num_heads_q == 64 || num_heads_q == 128,
                "kv_quant_sparse_attn_sharedkv_metadata_host: num_heads_q must be 64 or 128, got ", num_heads_q);
    TORCH_CHECK(num_heads_kv == 1, "kv_quant_sparse_attn_sharedkv_metadata_host: num_heads_kv must be 1, got ",
                num_heads_kv);
    TORCH_CHECK(head_dim == 512, "kv_quant_sparse_attn_sharedkv_metadata_host: head_dim must be 512, got ", head_dim);
    if (has_cmp_kv) {
        TORCH_CHECK(
            cmp_topk == 0 || cmp_topk == 512 || cmp_topk == 1024,
            "kv_quant_sparse_attn_sharedkv_metadata_host: A5 scheduling failed: cmp_topk must be 0, 512 or 1024, got ",
            cmp_topk);
        TORCH_CHECK(cmp_ratio == 4 || cmp_ratio == 128,
                    "kv_quant_sparse_attn_sharedkv_metadata_host: A5 scheduling failed: cmp_ratio must be 4 or 128 "
                    "with cmp_kv, got ",
                    cmp_ratio);
    }
    TORCH_CHECK(layout_q != "TND" || cu_seqlens_q.has_value(),
                "kv_quant_sparse_attn_sharedkv_metadata_host: cu_seqlens_q is required for TND");
    TORCH_CHECK(layout_q != "BSND" || max_seqlen_q > 0,
                "kv_quant_sparse_attn_sharedkv_metadata_host: max_seqlen_q is required for BSND");
    TORCH_CHECK(seqused_kv.has_value() || max_seqlen_kv > 0,
                "kv_quant_sparse_attn_sharedkv_metadata_host: provide seqused_kv or max_seqlen_kv");

    const int32_t *cuQ = CheckHostInt32(cu_seqlens_q, resolvedBatchSize + 1, "cu_seqlens_q");
    const int32_t *seqKv = CheckHostInt32(seqused_kv, resolvedBatchSize, "seqused_kv");
    if (cuQ != nullptr) {
        TORCH_CHECK(cuQ[0] == 0, "kv_quant_sparse_attn_sharedkv_metadata_host: cu_seqlens_q must start at zero");
        for (int64_t i = 0; i < resolvedBatchSize; ++i) {
            TORCH_CHECK(cuQ[i + 1] >= cuQ[i],
                        "kv_quant_sparse_attn_sharedkv_metadata_host: cu_seqlens_q must be non-decreasing");
        }
    }
    if (seqKv != nullptr) {
        for (int64_t i = 0; i < resolvedBatchSize; ++i) {
            TORCH_CHECK(seqKv[i] >= 0, "kv_quant_sparse_attn_sharedkv_metadata_host: seqused_kv must be non-negative");
        }
    }

    const HostTopology topo = ResolveHostTopology();
    std::string socLower = topo.socVersion;
    std::transform(socLower.begin(), socLower.end(), socLower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    TORCH_CHECK(socLower.find("ascend950") != std::string::npos,
                "kv_quant_sparse_attn_sharedkv_metadata_host: Ascend 950 is required, got ", topo.socVersion);
    TORCH_CHECK(num_heads_q != 128 || ((topo.aicCoreNum & 1U) == 0U && (topo.aivCoreNum & 1U) == 0U),
                "kv_quant_sparse_attn_sharedkv_metadata_host: 128 heads require even AI/vector core counts, got ",
                topo.aicCoreNum, " and ", topo.aivCoreNum);
    auto options = at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true);
    at::Tensor host = at::zeros({static_cast<int64_t>(kv_quant_optiling::SAS_META_SIZE)}, options);
    sgl_kernel_npu::kv_quant::KvQuantSparseAttnSharedkvMetadataHost scheduler;
    const bool ok =
        scheduler.Run(cuQ, seqKv, static_cast<int32_t>(resolvedBatchSize), static_cast<int32_t>(max_seqlen_q),
                      static_cast<int32_t>(max_seqlen_kv), static_cast<int32_t>(num_heads_q),
                      static_cast<int32_t>(num_heads_kv), static_cast<int32_t>(head_dim), topo.aicCoreNum,
                      topo.aivCoreNum, topo.socVersion, static_cast<int32_t>(cmp_topk), static_cast<int32_t>(cmp_ratio),
                      static_cast<int32_t>(ori_mask_mode), static_cast<int32_t>(cmp_mask_mode), ori_win_left,
                      ori_win_right, layout_q, layout_kv, has_ori_kv, has_cmp_kv, host.data_ptr<int32_t>());
    TORCH_CHECK(ok, "kv_quant_sparse_attn_sharedkv_metadata_host: A5 scheduling failed");
    return host.to(at::Device("npu"), /*non_blocking=*/true);
}
}  // namespace sglang::npu_kernel
#endif  // KV_QUANT_METADATA_CORE_ONLY

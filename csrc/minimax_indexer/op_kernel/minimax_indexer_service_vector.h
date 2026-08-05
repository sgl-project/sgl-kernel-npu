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

/*!
 * \file minimax_indexer_service_vector.h
 * \brief MiniMax-M3 indexer Vector core (multi-core: block-split + per-head LD merge).
 *
 * Per (batch, query head): block_score = max_t(q.k_t)*sm_scale_log2e with init/local
 * sentinels, streaming top-16 over KV blocks -> candidate_indices[QH, B, topk].
 *
 * Multi-core pipeline (MIX_AIC_1_2, 24 AI cores):
 *   - SplitCore partitions the (batch, block-range) base-block space across cores.
 *   - Each AIV0 core Cube<-handshake->Vector: CopyIn mm1ResGm->[G,block_size] UB,
 *     per-head max-reduce (WholeReduceMax in 64-element chunks; mask>64 is unreliable
 *     on 910B), init/local sentinel, streaming top-16 replace-min over ITS block range.
 *   - At its batch's last block each core writes its per-head top-16 partial to WS.
 *   - ProcessLD (after SyncAll): for every (batch, head) gathers all cores' partials
 *     from WS and scalar-merges to the global top-16; num_blocks<=topk batches emit
 *     the trivial [0..num_blocks)+(-1).
 */
#ifndef MINIMAX_INDEXER_SERVICE_VECTOR_H
#define MINIMAX_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "adv_api/sort/topk.h"
#include "minimax_indexer_common.h"
#include "minimax_indexer_vector.h"
#include "../op_host/tiling/minimax_tiling_data.h"

namespace sglang::npu_kernel::MIKernel {
using namespace MICommon;
using namespace AscendC;

constexpr float kNegInf = -1e38f;

// MITopkTilingRaw (raw mirror in MITilingData) and the kernel-side AscendC::TopkTiling
// (adv_api/kernel_tiling.h) must be layout-identical 112-byte PODs so that the
// host-flattened tiling bytes are a valid TopkTiling on the kernel side.
static_assert(sizeof(MIHost::MITopkTilingRaw) == sizeof(TopkTiling),
              "MITopkTilingRaw must mirror kernel TopkTiling layout exactly");

template <typename MIT>
class MIVector
{
public:
    using K_T = typename MIT::keyType;
    using MM1_OUT_T = float;

    __aicore__ inline MIVector() {}
    __aicore__ inline void ProcessVec(const MICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct MICommon::ConstInfo &constInfo,
                                      const __gm__ MIHost::MITilingData *tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int32_t> vec1ParamGm, GlobalTensor<K_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<uint32_t> seqLensGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidOffset);
    __aicore__ inline void AllocEventID() {}
    __aicore__ inline void FreeEventID() {}
    __aicore__ inline void InitLDBuffers(TPipe * /*pipe*/) {}
    __aicore__ inline void InitPartials();
    __aicore__ inline void ProcessLD();

private:
    GlobalTensor<MM1_OUT_T> mm1ResGm_;
    GlobalTensor<float> vec1ResGm_;
    GlobalTensor<int32_t> vec1ParamGm_;
    GlobalTensor<int32_t> indiceOutGm_;
    GlobalTensor<uint32_t> seqLensGm_;

    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TBuf<TPosition::VECCALC> minBuf_;
    TBuf<TPosition::VECCALC> topVBuf_;
    TBuf<TPosition::VECCALC> topIBuf_;
    TBuf<TPosition::VECCALC> initBuf_;
    TBuf<TPosition::VECCALC> fillCntBuf_;
    TBuf<TPosition::VECCALC> outIdxBuf_;
    // TopK-API LD-merge scratch: finishBuf_ is the (outter,1) finish tensor the
    // TopK API requires even with isHasfinish=false (never read functionally).
    TBuf<TPosition::VECCALC> finishBuf_;
    // TopK tiling (host-computed, byte-copied from the GM tiling mirror at init)
    // and the explicit-tmp UB scratch size (bytes). Both consumed by ProcessLD.
    TopkTiling topkTiling_{};
    uint32_t topkTmpSize_ = 0U;

    MICommon::ConstInfo constInfo_{};
    uint32_t aiCoreIdx_ = 0;
    uint32_t aicNum_ = 1;
    uint32_t bSize_ = 0;
    uint32_t gSize_ = 0;
    uint32_t topk_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t aivId_ = 0;
    uint32_t gStart_ = 0;
    uint32_t gNum_ = 0;
    // 1: output rows are [.., topk+1] with the causal local block appended at
    // slot topk (deduped to -1 when already present), in the [QH, B, topk+1]
    // memory layout the GQA sparse-attention kernels consume directly.
    uint32_t appendLocal_ = 0;
    // Streaming-append count. Uniform across heads (every head sees the same
    // blocks), so a single scalar counter replaces the per-head fillCntBuf_ lookups
    // inside the hot append loop. Reset to 0 at the first tile of each batch range.
    uint32_t fillCnt_ = 0;

    __aicore__ inline uint64_t PartOff(uint32_t core, uint32_t b, uint32_t g) const
    {
        return ((static_cast<uint64_t>(b) * gSize_ + g) * aicNum_) * topk_ +
               static_cast<uint64_t>(core) * topk_;
    }
};

template <typename MIT>
__aicore__ inline void MIVector<MIT>::InitParams(const struct MICommon::ConstInfo &constInfo,
                                                  const __gm__ MIHost::MITilingData *tilingData)
{
    this->constInfo_ = constInfo;
    aiCoreIdx_ = static_cast<uint32_t>(GetBlockIdx()) / 2;
    aicNum_ = static_cast<uint32_t>(GetBlockNum());
    bSize_ = constInfo.batchSize;
    gSize_ = constInfo.gSize;
    topk_ = constInfo.sparseCount;
    blockSize_ = constInfo.kCacheBlockSize;
    aivId_ = static_cast<uint32_t>(GetBlockIdx()) % 2;
    gNum_ = gSize_ / 2;
    gStart_ = aivId_ * gNum_;
    appendLocal_ = constInfo.appendLocal;
    // Layout-identical 112-byte copy: MITopkTilingRaw (GM tiling mirror) -> kernel
    // TopkTiling (validated by the static_assert above). Done once at init; the
    // GM->UR byte sequence is a handful of scalar loads, never on the hot path.
    const __gm__ uint8_t *tkSrc = reinterpret_cast<const __gm__ uint8_t *>(&tilingData->topkTiling);
    uint8_t *tkDst = reinterpret_cast<uint8_t *>(&topkTiling_);
    for (uint32_t i = 0; i < sizeof(TopkTiling); ++i) {
        tkDst[i] = tkSrc[i];
    }
    topkTmpSize_ = tilingData->topkTmpSize;
}

template <typename MIT>
__aicore__ inline void MIVector<MIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(scoreBuf_, gSize_ * constInfo_.s2BaseSize * sizeof(float));
    pipe->InitBuffer(maxBuf_, gSize_ * 8 * sizeof(float));
    pipe->InitBuffer(minBuf_, (gSize_ / 2) * 8 * sizeof(float));
    pipe->InitBuffer(topVBuf_, gSize_ * topk_ * sizeof(float));
    pipe->InitBuffer(topIBuf_, gSize_ * topk_ * sizeof(int32_t));
    pipe->InitBuffer(initBuf_, gSize_ * topk_ * sizeof(float));
    pipe->InitBuffer(fillCntBuf_, gSize_ * sizeof(uint32_t));
    // outIdxBuf_ holds one fused output row (topk + append slot), assembled in UB
    // and flushed to GM with a single MTE3 DataCopyPad (the per-element scalar
    // SetValue path races other cores' writes on 910B).
    pipe->InitBuffer(outIdxBuf_, MICommon::Align(topk_ + 1, 8U) * sizeof(int32_t));
    // finishBuf_ for the TopK API (isHasfinish=false -> contents never read, only
    // the address must be valid UB). 32B is the minimum UB allocation granularity.
    pipe->InitBuffer(finishBuf_, MICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);
}

template <typename MIT>
__aicore__ inline void MIVector<MIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                            GlobalTensor<float> vec1ResGm,
                                                            GlobalTensor<int32_t> vec1ParamGm,
                                                            GlobalTensor<K_T> /*weightsGm*/,
                                                            GlobalTensor<int32_t> indiceOutGm,
                                                            GlobalTensor<uint32_t> seqLensGm)
{
    this->mm1ResGm_ = mm1ResGm;
    this->vec1ResGm_ = vec1ResGm;
    this->vec1ParamGm_ = vec1ParamGm;
    this->indiceOutGm_ = indiceOutGm;
    this->seqLensGm_ = seqLensGm;
}

template <typename MIT>
__aicore__ inline void MIVector<MIT>::CleanInvalidOutput(int64_t invalidOffset)
{
    for (int64_t i = 0; i < static_cast<int64_t>(topk_); i++) {
        indiceOutGm_.SetValue(invalidOffset + i, static_cast<int32_t>(constInfo_.INVALID_IDX));
    }
}

template <typename MIT>
__aicore__ inline void MIVector<MIT>::InitPartials()
{
    LocalTensor<float> initV = initBuf_.Get<float>();
    AscendC::DataCopyParams dp;
    dp.blockCount = 1;
    dp.blockLen = topk_ * sizeof(float);
    dp.srcStride = 0;
    dp.dstStride = 0;
    Duplicate(initV, kNegInf, topk_);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> initI = initBuf_.Get<int32_t>();
    Duplicate(initI, static_cast<int32_t>(constInfo_.INVALID_IDX), topk_);
    PipeBarrier<PIPE_V>();
    for (uint32_t b = 0; b < bSize_; b++) {
        for (uint32_t g = gStart_; g < gStart_ + gNum_; g++) {
            const uint64_t off = PartOff(aiCoreIdx_, b, g);
            event_t eVM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eVM);
            WaitFlag<HardEvent::V_MTE3>(eVM);
            DataCopyPad(vec1ResGm_[off], initV, dp);
            event_t eVM2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eVM2);
            WaitFlag<HardEvent::V_MTE3>(eVM2);
            DataCopyPad(vec1ParamGm_[off], initI, dp);
        }
    }
}

template <typename MIT>
__aicore__ inline void MIVector<MIT>::ProcessVec(const MICommon::RunInfo &info)
{
    const uint32_t gSize = gSize_;
    const uint32_t gStart = gStart_;
    const uint32_t gNum = gNum_;
    const uint32_t topk = topk_;
    const uint32_t blockSize = blockSize_;
    const uint32_t stride = info.actualSingleProcessSInnerSizeAlign;
    const uint32_t valid = info.actualSingleProcessSInnerSize;
    const uint64_t mmBase =
        (static_cast<uint64_t>(info.loop) % 2) * constInfo_.mBaseSize * constInfo_.s2BaseSize;
    const float scale = constInfo_.smScaleLog2e;
    const uint32_t numBlocks = MICommon::CeilDiv(info.actS2Size, blockSize);
    const uint32_t localStart =
        (numBlocks > constInfo_.localBlocks) ? (numBlocks - constInfo_.localBlocks) : 0;
    const uint32_t gNum8 = MICommon::Align(gNum, 8U);

    LocalTensor<float> topV = topVBuf_.Get<float>();
    LocalTensor<int32_t> topI = topIBuf_.Get<int32_t>();
    LocalTensor<uint32_t> fillCnt = fillCntBuf_.Get<uint32_t>();
    if (info.isFirstS2InnerLoop) {
        fillCnt_ = 0;
        for (uint32_t g = gStart; g < gStart + gNum; g++) {
            for (uint32_t s = 0; s < topk; s++) {
                topV.SetValue(g * topk + s, kNegInf);
                topI.SetValue(g * topk + s, static_cast<int32_t>(constInfo_.INVALID_IDX));
            }
        }
    }

    LocalTensor<float> scoreUb = scoreBuf_.Get<float>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = gNum;
    copyParams.blockLen = stride * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(scoreUb[gStart * stride], mm1ResGm_[mmBase + gStart * stride], copyParams, padParams);
    event_t eMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eMte2V);
    WaitFlag<HardEvent::MTE2_V>(eMte2V);

    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    for (uint32_t blockB = 0; blockB < 2; blockB++) {
        const uint32_t validB =
            (blockB == 0) ? MICommon::Min(valid, blockSize)
                          : (valid > blockSize ? valid - blockSize : 0);
        if (validB == 0) {
            continue;
        }
        const uint32_t tokenOff = blockB * blockSize;
        const uint32_t blocksPerTile = constInfo_.s2BaseSize / blockSize;
        const uint32_t logicalBlock = info.s2Idx * blocksPerTile + blockB;
        const bool isInit = (constInfo_.initBlocks > 0) && (logicalBlock < constInfo_.initBlocks) &&
                            (logicalBlock < numBlocks);
        const bool isLocal = (constInfo_.localBlocks > 0) && (logicalBlock >= localStart) &&
                             (logicalBlock < numBlocks);

        const uint32_t nChunks = MICommon::CeilDiv(validB, 64U);
        const uint32_t mbOff = blockB * 2;
        for (uint32_t c = 0; c < nChunks; c++) {
            uint32_t chunkMask = (validB - c * 64 < 64) ? (validB - c * 64) : 64;
            WholeReduceMax(maxUb[(mbOff + c) * gNum8],
                           scoreUb[gStart * stride + tokenOff + c * 64], chunkMask, gNum, 1, 1,
                           stride / 8, ReduceOrder::ORDER_ONLY_VALUE);
        }
        LocalTensor<float> mxUb = maxUb[mbOff * gNum8];
        for (uint32_t c = 1; c < nChunks; c++) {
            Max(mxUb, mxUb, maxUb[(mbOff + c) * gNum8], gNum);
        }
        if (isInit) {
            Duplicate(mxUb, 1e30f, gNum);
        } else if (isLocal) {
            Duplicate(mxUb, 1e29f, gNum);
        } else {
            Muls(mxUb, mxUb, scale, gNum);
        }

        const bool replacePhase = (fillCnt_ >= topk);
        if (replacePhase) {
            event_t eSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eSV);
            WaitFlag<HardEvent::S_V>(eSV);
            LocalTensor<float> minUb = minBuf_.Get<float>();
            WholeReduceMin(minUb, topV[gStart * topk], topk, gNum, 1, 1, topk / 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            event_t eVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eVS2);
            WaitFlag<HardEvent::V_S>(eVS2);
            LocalTensor<int32_t> minIdx = minUb.template ReinterpretCast<int32_t>();
            // WholeReduceMin ORDER_VALUE_INDEX lays out [value, index] pairs at
            // stride 2 per repeat: head k -> value @ 2k, index @ 2k+1. (Earlier code
            // read at stride 8, which silently corrupted every head except head 0
            // once the replace phase activated at blocks/core > topk.)
            for (uint32_t g = gStart; g < gStart + gNum; g++) {
                float mx = mxUb.GetValue(g - gStart);
                uint32_t li = (g - gStart) * 2;
                float minV = minUb.GetValue(li);
                if (mx > minV) {
                    uint32_t minSlot = static_cast<uint32_t>(minIdx.GetValue(li + 1));
                    if (minSlot >= topk) {
                        minSlot = 0;
                    }
                    topV.SetValue(g * topk + minSlot, mx);
                    topI.SetValue(g * topk + minSlot, static_cast<int32_t>(logicalBlock));
                }
            }
        } else {
            event_t eVSapp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eVSapp);
            WaitFlag<HardEvent::V_S>(eVSapp);
            const uint32_t fc = fillCnt_;  // uniform across heads; hoisted out of per-head loop
            for (uint32_t g = gStart; g < gStart + gNum; g++) {
                float mx = mxUb.GetValue(g - gStart);
                const uint32_t base = g * topk;
                topV.SetValue(base + fc, mx);
                topI.SetValue(base + fc, static_cast<int32_t>(logicalBlock));
            }
            fillCnt_ = fc + 1;
        }
    }

    if (info.isLastS2InnerLoop) {
        event_t eSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eSV);
        WaitFlag<HardEvent::S_MTE3>(eSV);
        AscendC::DataCopyParams dp;
        dp.blockCount = 1;
        dp.blockLen = topk * sizeof(float);
        dp.srcStride = 0;
        dp.dstStride = 0;
        for (uint32_t g = gStart; g < gStart + gNum; g++) {
            const uint64_t off = PartOff(aiCoreIdx_, info.bIdx, g);
            DataCopyPad(vec1ResGm_[off], topV[g * topk], dp);
            DataCopyPad(vec1ParamGm_[off], topI[g * topk], dp);
        }
    }
}

template <typename MIT>
__aicore__ inline void MIVector<MIT>::ProcessLD()
{
    const uint32_t gSize = gSize_;
    const uint32_t gStart = gStart_;
    const uint32_t gNum = gNum_;
    const uint32_t topk = topk_;
    const uint32_t blockSize = blockSize_;
    using MIServiceVec::SetWaitFlag;  // lightning-derived template combos
    const uint32_t partPerHead = aicNum_ * topk;
    // TopK on 910B tiles the inner axis in 64-element groups; a non-÷64 tail is
    // mishandled. Pad partPerHead up to a multiple of 64 with -inf/-1 candidates
    // (host computes the TopkTiling for this same paddedLen). The padding region
    // [partPerHead..paddedLen) is never touched by the per-(b,g) DataCopyPad
    // (which writes exactly partPerHead elements), so a one-time fill here
    // persists across all merge iterations; -inf sorts to the bottom and is never
    // selected (k=16 << paddedLen).
    const uint32_t paddedLen = MICommon::Align(partPerHead, 64U);
    LocalTensor<float> mergeVals = scoreBuf_.Get<float>();
    LocalTensor<int32_t> mergeIdx = initBuf_.Get<int32_t>();
    if (paddedLen > partPerHead) {
        Duplicate(mergeVals[partPerHead], kNegInf, paddedLen - partPerHead);
        Duplicate(mergeIdx[partPerHead], static_cast<int32_t>(constInfo_.INVALID_IDX), paddedLen - partPerHead);
        PipeBarrier<PIPE_V>();
    }
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = partPerHead * sizeof(float);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    extParams.rsv = 0;
    AscendC::DataCopyPadExtParams<float> padF{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
    const uint32_t outW = topk + appendLocal_;  // fused append: output row width topk+1
    for (uint32_t b = 0; b < bSize_; b++) {
        uint32_t seqLen = seqLensGm_.GetValue(b);
        uint32_t numBlocks = MICommon::CeilDiv(seqLen, blockSize);
        // Causal local block to append at slot topk (triton append_local semantics):
        // the last block the query position touches, deduped to -1 when it is
        // already among the candidates. numBlocks==0 batches append -1.
        const uint32_t queryPos = (seqLen > 0) ? (seqLen - 1) : 0;
        const int32_t localBlk =
            (numBlocks > 0) ? static_cast<int32_t>(MICommon::Min(queryPos / blockSize, numBlocks - 1)) : -1;
        if (numBlocks <= topk) {
            for (uint32_t g = gStart; g < gStart + gNum; g++) {
                if ((b * gSize + g) % aicNum_ != aiCoreIdx_) {
                    continue;
                }
                // Fused output layout [QH, B, outW]: row (g, b) at (g*bSize + b)*outW.
                const uint64_t outOff = (static_cast<uint64_t>(g) * bSize_ + b) * outW;
                LocalTensor<int32_t> outIdx = outIdxBuf_.Get<int32_t>();
                for (uint32_t slot = 0; slot < topk; slot++) {
                    outIdx.SetValue(slot,
                                    (slot < numBlocks) ? static_cast<int32_t>(slot) : constInfo_.INVALID_IDX);
                }
                if (appendLocal_) {
                    bool present = false;
                    for (uint32_t s = 0; s < topk; s++) {
                        if (s < numBlocks && static_cast<int32_t>(s) == localBlk) {
                            present = true;
                            break;
                        }
                    }
                    outIdx.SetValue(topk, present ? constInfo_.INVALID_IDX : localBlk);
                }
                // single MTE3 batch write of the whole row (scalar SetValue GM races)
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                AscendC::DataCopyParams dp;
                dp.blockCount = 1;
                dp.blockLen = outW * sizeof(int32_t);
                dp.srcStride = 0;
                dp.dstStride = 0;
                DataCopyPad(indiceOutGm_[outOff], outIdx, dp);
            }
            continue;
        }
        for (uint32_t g = gStart; g < gStart + gNum; g++) {
            if ((b * gSize + g) % aicNum_ != aiCoreIdx_) {
                continue;
            }
            // Fused output layout [QH, B, outW]: row (g, b) at (g*bSize + b)*outW.
            const uint64_t outOff = (static_cast<uint64_t>(g) * bSize_ + b) * outW;
            const uint64_t poff = PartOff(0, b, g);
            DataCopyPad(mergeVals, vec1ResGm_[poff], extParams, padF);
            DataCopyPad(mergeIdx, vec1ParamGm_[poff], extParams, padI);
            // TopK (vectorized) replaces the scalar running-min topk-16 over the
            // partPerHead = aicNum_*topk per-head partials gathered from WS. UB is
            // carved from scoreBuf_/initBuf_ (both free in the LD phase: ProcessVec
            // already ran): mergeVals/tmp/dstValue share scoreBuf_; mergeIdx/dstIndex
            // share initBuf_. isInitIndex=true carries mergeIdx (block indices) into
            // dstIndex; isLargest=true keeps the 16 largest scores. DataCopyPad writes
            // exactly partPerHead candidates; the [partPerHead..paddedLen) tail is the
            // -inf padding filled once above. TopK runs over paddedLen (÷64).
            event_t eMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eMte2V);
            WaitFlag<HardEvent::MTE2_V>(eMte2V);

            const uint32_t tmpBytes = topkTmpSize_;
            // dstValue sits after mergeVals[0..paddedLen) + tmp in scoreBuf_,
            // float-aligned, and must not overlap src (isReuseSrc=false).
            const uint32_t dstValOff =
                (paddedLen * sizeof(float) + tmpBytes + sizeof(float) - 1U) / sizeof(float);
            LocalTensor<uint8_t> tmpLocal = mergeVals.ReinterpretCast<uint8_t>()[paddedLen * sizeof(float)];
            LocalTensor<float> dstValueLocal = mergeVals[dstValOff];
            LocalTensor<int32_t> dstIndexLocal = mergeIdx[paddedLen];
            LocalTensor<bool> finishLocal = finishBuf_.Get<bool>();
            // TopK runs over the ÷64-padded inner (host tiled TopkTiling for the
            // same paddedLen). n==inner==paddedLen: the [partPerHead..paddedLen)
            // tail is the -inf padding filled once above, never selected.
            TopKInfo topKInfo;
            topKInfo.outter = 1;
            topKInfo.inner = static_cast<int32_t>(paddedLen);
            topKInfo.n = static_cast<int32_t>(paddedLen);
            TopK<float, /*isInitIndex=*/true, /*isHasfinish=*/false, /*isReuseSrc=*/false, TopKMode::TOPK_NORMAL>(
                dstValueLocal, dstIndexLocal, mergeVals, mergeIdx, finishLocal, tmpLocal,
                static_cast<int32_t>(topk), topkTiling_, topKInfo, /*isLargest=*/true);
            // V -> S sync before the scalar reads of dstIndexLocal below.
            event_t eVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eVS);
            WaitFlag<HardEvent::V_S>(eVS);

            LocalTensor<int32_t> outIdx = outIdxBuf_.Get<int32_t>();
            for (uint32_t s = 0; s < topk; s++) {
                outIdx.SetValue(s, dstIndexLocal.GetValue(s));
            }
            if (appendLocal_) {
                bool present = false;
                for (uint32_t s = 0; s < topk; s++) {
                    if (dstIndexLocal.GetValue(s) == localBlk) {
                        present = true;
                        break;
                    }
                }
                outIdx.SetValue(topk, present ? constInfo_.INVALID_IDX : localBlk);
            }
                // single MTE3 batch write of the whole row (scalar SetValue GM races)
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                AscendC::DataCopyParams dp;
                dp.blockCount = 1;
                dp.blockLen = outW * sizeof(int32_t);
                dp.srcStride = 0;
                dp.dstStride = 0;
                DataCopyPad(indiceOutGm_[outOff], outIdx, dp);
        }
    }
}
}  // namespace sglang::npu_kernel::MIKernel
#endif  // MINIMAX_INDEXER_SERVICE_VECTOR_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "tensorutils.h"

using namespace matmul;
using namespace AscendC;

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64;  // 256Byte for float
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr float EPSILON_FOR_STABILITY = 1e-6f;

constexpr int BLK_SIZE = 64;
constexpr int VEC_FLOAT = 64;
constexpr int NUM_DBLK_FLOAT = 8;

struct RGDRInitParams {
    GM_ADDR mixqkv;
    GM_ADDR gama;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;

    GM_ADDR recurrentState;
    GM_ADDR cacheIndices;
};

template <typename inType, typename outType>
class RGDR
{
public:
    __aicore__ inline RGDR(uint32_t B, uint32_t S, uint32_t nk, uint32_t dk, uint32_t nv, uint32_t dv,
                           bool hasIntermediateState, bool hasAcceptedTokens, bool hasGama, uint32_t vStep,
                           uint32_t ubRestBytes, float scale)
    {
        B_ = B;
        S_ = S;
        T_ = B * S;
        NK_ = nk;
        realK_ = dk;
        NV_ = nv;
        realV_ = dv;

        hasIntermediateState_ = hasIntermediateState;
        needRecurrentInit_ = false;

        hasAcceptedTokens_ = hasAcceptedTokens;
        hasGama_ = hasGama;
        vStep_ = vStep;
        restUbSize_ = ubRestBytes;
        alignK_ = Ceil(dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        scale_ = scale;
        load = 0;
        usedblk = 0;
    }

    __aicore__ inline void Init(const RGDRInitParams &initParams, TPipe *pipe)
    {
        uint64_t blockDim = GetBlockNum();
        blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        pipe_ = pipe;

        tokenStrideSize_ = (NK_ * realK_ + NK_ * realK_ + NV_ * realV_) * sizeof(inType);

        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RGDRInitParams &initParams)
    {
        mixqkvGm_.SetGlobalBuffer((__gm__ inType *)initParams.mixqkv);
        gamaGm_.SetGlobalBuffer((__gm__ float *)initParams.gama);
        betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
        initStateGm_.SetGlobalBuffer((__gm__ inType *)initParams.initState);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        numAcceptedTokensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ outType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);

        if (hasIntermediateState_) {
            recurrentStateGm_.SetGlobalBuffer((__gm__ inType *)initParams.recurrentState);
            cacheIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cacheIndices);
        }
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
        uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
        uint32_t betaUbNum = Ceil(MAX_MTP * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;  // 8: 8 * 4 = 32B;
        uint32_t sumLocalNum = Ceil(MAX_MTP, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;

        uint32_t inQSize = BUFFER_NUM * MAX_MTP * alignK_ * sizeof(inType);
        uint32_t inVSize = BUFFER_NUM * MAX_MTP * alignV_ * sizeof(inType);
        uint32_t inStateSize = BUFFER_NUM * alignK_ * vStep_ * sizeof(inType);
        uint32_t inGamaSize = BUFFER_NUM * MAX_MTP * NV_ * sizeof(float);
        uint32_t inBetaSize = BUFFER_NUM * MAX_MTP * NV_ * sizeof(inType);
        uint32_t outStateSize = BUFFER_NUM * alignK_ * vStep_ * sizeof(outType);
        uint32_t outAttnSize = BUFFER_NUM * vStep_ * sizeof(outType);
        uint32_t totalBufferSize = inQSize * 2 +   // Q and K queues (each double buffered)
                                   inVSize +       // V queue (double buffered)
                                   inStateSize +   // State input queue (double buffered)
                                   inGamaSize +    // Gamma queue
                                   inBetaSize +    // Beta queue
                                   outStateSize +  // State output queue (double buffered)
                                   outAttnSize;

        pipe_->InitBuffer(stageBuff, totalBufferSize);
        uint32_t totalBufferOffset = 0;
        qLocal =
            stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * MAX_MTP * alignK_), totalBufferOffset);
        totalBufferOffset += inQSize;
        kLocal =
            stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * MAX_MTP * alignK_), totalBufferOffset);
        totalBufferOffset += inQSize;
        vLocal =
            stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * MAX_MTP * alignV_), totalBufferOffset);
        totalBufferOffset += inVSize;
        stateLocal =
            stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * alignK_ * vStep_), totalBufferOffset);
        totalBufferOffset += inStateSize;
        gamaLocal =
            stageBuff.GetWithOffset<float>(static_cast<uint32_t>(BUFFER_NUM * MAX_MTP * NV_), totalBufferOffset);
        totalBufferOffset += inGamaSize;
        betaLocal =
            stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * MAX_MTP * NV_), totalBufferOffset);
        totalBufferOffset += inBetaSize;
        stateOutLocal =
            stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * alignK_ * vStep_), totalBufferOffset);
        totalBufferOffset += outStateSize;
        attnOutLocal = stageBuff.GetWithOffset<inType>(static_cast<uint32_t>(BUFFER_NUM * vStep_), totalBufferOffset);
        pipe_->InitBuffer(tmpBuff, restUbSize_);
        uint32_t buffOffset = 0;
        deltaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;

        qTempInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kTempInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;

        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        broadTmpInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;

        qSumLocal = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(sumLocalNum), buffOffset);
        buffOffset += sumLocalNum * sizeof(float);
        kSumLocal = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(sumLocalNum), buffOffset);
        buffOffset += sumLocalNum * sizeof(float);

        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbNum), buffOffset);

        in_ready_Beta.Init();
        in_empty_Beta.Init();
        out_ready_Attn.Init();
        out_empty_Attn.Init();
    }

    __aicore__ inline void ComputeAvgload()
    {
        avgload = Ceil(B_ * NV_, GetBlockNum());
    }

    __aicore__ inline void Process()
    {
        ComputeAvgload();
        const uint64_t totalWorkload = B_ * NV_;
        const uint64_t startIdx = blockIdx * avgload;
        uint64_t calculatedIdx = (blockIdx + 1) * avgload;
        const uint64_t endIdx = (calculatedIdx < totalWorkload) ? calculatedIdx : totalWorkload;
        if (startIdx >= totalWorkload) {
            return;
        }
        in_empty_Beta.setall();
        out_empty_Attn.setall();
        uint64_t lastProcessedBatch = UINT64_MAX;
        for (uint64_t workIdx = startIdx; workIdx < endIdx; workIdx++) {
            uint64_t batchIdx = workIdx / NV_;
            uint64_t headIdx = workIdx % NV_;
            // Each batch has only one token. seq0 = batchIdx, seq1 = batchIdx + 1
            uint64_t seq0 = batchIdx * S_;
            uint64_t seq1 = seq0 + cuSeqlensGm_.GetValue(batchIdx);

            uint64_t stateOffset = hasAcceptedTokens_
                                       ? ssmStateIndicesGm_.GetValue(seq0 + numAcceptedTokensGm_.GetValue(batchIdx) - 1)
                                       : ssmStateIndicesGm_.GetValue(seq0);
            uint64_t recurrentStateOffset = 0;
            if (hasIntermediateState_) {
                recurrentStateOffset = cacheIndicesGm_.GetValue(batchIdx);
            }

            if (batchIdx != lastProcessedBatch) {
                needRecurrentInit_ = false;

                if (hasIntermediateState_ && stateOffset % S_ == 0) {
                    needRecurrentInit_ = true;
                }
                CopyInGamaBeta(seq0, seq1);
                lastProcessedBatch = batchIdx;
            }
            in_empty_Beta.wait();
            ProcessHead(seq0, seq1, headIdx, stateOffset, recurrentStateOffset);
            in_empty_Beta.set();
        }
        in_empty_Beta.release();  // wait
        out_empty_Attn.release();
    }

private:
    __aicore__ inline void CopyInState(uint64_t stateOffset, uint64_t recurrentOffset, uint32_t curSingleV)
    {
        DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                        static_cast<uint16_t>(realK_ * sizeof(inType)), 0, 0, 0};
        DataCopyPadExtParams<inType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};

        if (needRecurrentInit_) {
            DataCopyPad(stateLocal, recurrentStateGm_[recurrentOffset], stateInParams, padParams);
        } else {
            DataCopyPad(stateLocal, initStateGm_[stateOffset], stateInParams, padParams);
        }

        in_ready_Beta.set();
        in_ready_Beta.wait();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(stateInUb, stateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * Note: This method is specifically designed and invoked only when
     * curSingleV == 64 and alignK_ == 128.
     */
    __aicore__ inline void ReduceSum_AR_64x128_ReuseSource(const LocalTensor<float> &dstTensor,
                                                           const LocalTensor<float> &srcTensor,
                                                           const uint32_t srcShape[2])
    {
        const uint32_t B32_DATA_NUM_PER_REPEAT = 64;
        const uint32_t B32_DATA_NUM_PER_BLOCK = 8;
        uint32_t first = srcShape[0];  // 64
        uint32_t last = srcShape[1];   // 128
        uint32_t padLast = 128;        // AlignUp(128, 8) = 128（已对齐）
        LocalTensor<float> tmpBuf = srcTensor;

        uint32_t splitK = 128;           // 1 << FindClosestPowerOfTwo(128) = 128
        uint32_t tail = 0;               // last - splitK = 0
        uint32_t perRowReduceSize = 64;  // splitK >> 1 = 64
        SetMaskCount();

        uint32_t perRowSize = padLast;
        uint8_t dstRepStride = static_cast<uint8_t>(perRowSize / B32_DATA_NUM_PER_BLOCK);
        uint8_t src0RepStride = dstRepStride;  // 16
        uint8_t src1RepStride = static_cast<uint8_t>(padLast / B32_DATA_NUM_PER_BLOCK);

        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(B32_DATA_NUM_PER_REPEAT);  // 64
        uint32_t srcOffset = perRowReduceSize;                            // 64
        uint32_t tmpOffset = 0;                                           // 0

        Add<float, false>(tmpBuf[tmpOffset], tmpBuf[tmpOffset], srcTensor[srcOffset], MASK_PLACEHOLDER, first,
                          {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
        PipeBarrier<PIPE_V>();
        uint16_t blockCount = first;                                                 // 64
        uint16_t blockLen = perRowReduceSize / B32_DATA_NUM_PER_BLOCK;               // 64 / 8 = 8
        uint16_t srcStride = (padLast - perRowReduceSize) / B32_DATA_NUM_PER_BLOCK;  // (128 - 64) / 8 = 8
        uint16_t dstStride = 0;

        DataCopy(tmpBuf, tmpBuf, {blockCount, blockLen, srcStride, dstStride});
        PipeBarrier<PIPE_V>();

        ResetMask();
        SetMaskCount();

        SetVectorMask<float, MaskMode::COUNTER>(first * B32_DATA_NUM_PER_REPEAT);  // 64 * 64 = 4096
        BlockReduceSum<float, false>(tmpBuf, tmpBuf, 1, MASK_PLACEHOLDER, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        SetVectorMask<float, MaskMode::COUNTER>(first * B32_DATA_NUM_PER_BLOCK);  // 64 * 8 = 512
        BlockReduceSum<float, false>(dstTensor, tmpBuf, 1, MASK_PLACEHOLDER, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        SetMaskNorm();
        PipeBarrier<PIPE_V>();
        ResetMask();
    }

    __aicore__ inline void kL2Norm(uint32_t seqLen)
    {
        Mul<float>(kTempInUb, kInUb, kInUb, seqLen * alignK_);
        PipeBarrier<PIPE_V>();

        Sum<float>(kSumLocal, kTempInUb, {seqLen, alignK_, realK_});
        PipeBarrier<PIPE_V>();

        Adds<float>(kSumLocal, kSumLocal, EPSILON_FOR_STABILITY, seqLen);
        PipeBarrier<PIPE_V>();
        Rsqrt<float>(kSumLocal, kSumLocal, seqLen);

        AscendC::TEventID eventIDS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIDS);
        WaitFlag<HardEvent::V_S>(eventIDS);

        float normFactors[MAX_MTP];
        for (uint32_t i = 0; i < seqLen; ++i) {
            normFactors[i] = kSumLocal.GetValue(i);

            eventIDS = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(eventIDS);
            WaitFlag<HardEvent::S_V>(eventIDS);

            uint32_t offset = i * alignK_;
            float inv_norm = normFactors[i];

            Muls<float>(kInUb[offset], kInUb[offset], inv_norm, realK_);
        }
    }

    __aicore__ inline void qL2Norm(uint32_t seqLen)
    {
        Mul<float>(qTempInUb, qInUb, qInUb, seqLen * alignK_);
        PipeBarrier<PIPE_V>();

        Sum<float>(qSumLocal, qTempInUb, {seqLen, alignK_, realK_});
        PipeBarrier<PIPE_V>();

        Adds<float>(qSumLocal, qSumLocal, EPSILON_FOR_STABILITY, seqLen);
        PipeBarrier<PIPE_V>();
        Rsqrt<float>(qSumLocal, qSumLocal, seqLen);

        AscendC::TEventID eventIDS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIDS);
        WaitFlag<HardEvent::V_S>(eventIDS);

        float normFactors[MAX_MTP];
        for (uint32_t i = 0; i < seqLen; ++i) {
            normFactors[i] = qSumLocal.GetValue(i);

            eventIDS = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(eventIDS);
            WaitFlag<HardEvent::S_V>(eventIDS);

            uint32_t offset = i * alignK_;
            float inv_norm = normFactors[i];

            Muls<float>(qInUb[offset], qInUb[offset], inv_norm, realK_);
        }
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset, uint64_t curVOffset, uint64_t seq_i,
                                   uint64_t attnOffset, int32_t seq0, int32_t seq1, uint64_t head_i,
                                   uint64_t stateOffset, uint64_t v_i)
    {
        uint32_t qk_head_idx = head_i / (NV_ / NK_);
        int32_t seqLen = seq1 - seq0;

        uint64_t qTokenOffset = qk_head_idx * realK_;

        uint64_t kTokenOffset = (NK_ * realK_) + (qk_head_idx * realK_);

        uint64_t vTokenOffset = (2 * NK_ * realK_) + (head_i * realV_) + v_i;

        uint64_t mixQGlobalOffset = seq0 * (tokenStrideSize_ / sizeof(inType)) + qTokenOffset;
        uint64_t mixKGlobalOffset = seq0 * (tokenStrideSize_ / sizeof(inType)) + kTokenOffset;
        uint64_t mixVGlobalOffset = seq0 * (tokenStrideSize_ / sizeof(inType)) + vTokenOffset;

        uint32_t qkSrcStride = tokenStrideSize_ - realK_ * sizeof(inType);
        uint32_t vSrcStride = tokenStrideSize_ - realV_ * sizeof(inType);

        DataCopyExtParams qkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                     qkSrcStride, 0, 0};
        DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};

        DataCopyExtParams vInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                    vSrcStride, 0, 0};
        DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};

        if (!qkvcopyFlag_) {
            //===================================KV START===============================
            DataCopyPad(kLocal, mixqkvGm_[mixKGlobalOffset], qkInParams, qkPadParams);
            DataCopyPad(vLocal, mixqkvGm_[mixVGlobalOffset], vInParams, vPadParams);

            in_ready_Beta.set();
            in_ready_Beta.wait();

            Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
            AscendC::PipeBarrier<PIPE_V>();
            kL2Norm(seqLen);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
            AscendC::PipeBarrier<PIPE_V>();
            //===================================KV END===============================
            DataCopyPad(qLocal, mixqkvGm_[mixQGlobalOffset], qkInParams, qkPadParams);
            in_ready_Beta.set();
        }

        uint32_t stateShape[2] = {curSingleV, alignK_};
        uint32_t ktShape[2] = {1, alignK_};
        uint32_t deltaShape[2] = {curSingleV, 1};

        {
            uint64_t gbOffset = head_i + (seq_i - seq0) * NV_;
            gama_ = hasGama_ ? gamaLocal.GetValue(gbOffset) : 1;
            beta_ = betaInUb.GetValue(gbOffset);
        }

        if (hasGama_) {
            AscendC::PipeBarrier<PIPE_V>();
            Muls(stateInUb, stateInUb, gama_, alignK_ * curSingleV);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < alignK_ / VEC_FLOAT; i++) {
            // {1, 8} * {1, alignK_} => {1, alignK_}
            uint32_t index = i * VEC_FLOAT;  // 0, 64

            Mul<float, false>(broadTmpInUb[index], stateInUb[index], kInUb[curQKOffset + index], MASK_PLACEHOLDER,
                              curSingleV, {1, 1, 1, 16, 16, 0});
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::PipeBarrier<PIPE_V>();
        if (curSingleV == 64 && alignK_ == 128) {
            ReduceSum_AR_64x128_ReuseSource(deltaInUb, broadTmpInUb, stateShape);
        } else {
            ReduceSum<float, Pattern::Reduce::AR, true>(deltaInUb, broadTmpInUb, stateShape, true);
        }
        AscendC::PipeBarrier<PIPE_V>();
        Sub<float>(deltaInUb, vInUb[curVOffset], deltaInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(deltaInUb, deltaInUb, beta_, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();

        if (!qkvcopyFlag_) {
            //===================================QQQQQQQ START===============================
            in_ready_Beta.wait();
            Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
            AscendC::PipeBarrier<PIPE_V>();
            qL2Norm(seqLen);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(qInUb, qInUb, scale_, seqLen * alignK_);
            AscendC::PipeBarrier<PIPE_V>();
            //===================================QQQQQQQ END===============================
            qkvcopyFlag_ = true;
        }

        Brcb(broadTmpInUb, deltaInUb, curSingleV / 8, {1, NUM_DBLK_FLOAT});
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < alignK_ / VEC_FLOAT; i++) {
            // {1, 8} * {1, alignK_} => {1, alignK_}
            uint32_t index = i * VEC_FLOAT;  // 0, 64
            MulAddDst<float, float, false>(stateInUb[index], broadTmpInUb, kInUb[curQKOffset + index], MASK_PLACEHOLDER,
                                           curSingleV, {1, 0, 1, 16, 1, 0});
            AscendC::PipeBarrier<PIPE_V>();
        }

        out_empty_Attn.wait();

        Cast(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);

        out_ready_Attn.set();
        out_ready_Attn.wait();

        uint64_t curStateOutOffset = ((ssmStateIndicesGm_.GetValue(seq_i) * NV_ + head_i) * realV_ + v_i) * realK_;

        CopyOutState(curStateOutOffset, curSingleV);

        for (uint32_t i = 0; i < alignK_ / VEC_FLOAT; i++) {
            // {1, 8} * {1, alignK_} => {1, alignK_}
            uint32_t index = i * VEC_FLOAT;  // 0, 64

            Mul<float, false>(broadTmpInUb[index], stateInUb[index], qInUb[curQKOffset + index], MASK_PLACEHOLDER,
                              curSingleV, {1, 1, 1, 16, 16, 0});
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::PipeBarrier<PIPE_V>();
        if (curSingleV == 64 && alignK_ == 128) {
            ReduceSum_AR_64x128_ReuseSource(attnInUb, broadTmpInUb, stateShape);
        } else {
            ReduceSum<float, Pattern::Reduce::AR, true>(attnInUb, broadTmpInUb, stateShape, true);
        }
        AscendC::PipeBarrier<PIPE_V>();
        Cast(attnOutLocal, attnInUb, AscendC::RoundMode::CAST_RINT, curSingleV);
        out_ready_Attn.set();
        out_ready_Attn.wait();

        CopyOutAttn(attnOffset, curSingleV);

        out_empty_Attn.set();
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV)
    {
        DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};

        DataCopyPad(attnOutGm_[attnOffset], attnOutLocal, attnOutParams);
    }

    __aicore__ inline void CopyOutState(uint64_t stateOffset, uint32_t curSingleV)
    {
        DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                      static_cast<uint16_t>(realK_ * sizeof(outType)), 0, 0};

        DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
    }

    __aicore__ inline void CopyInGamaBeta(int32_t seq0, int32_t seq1)
    {
        int32_t seqLen = seq1 - seq0;
        uint64_t bBatchSize = Ceil(seqLen * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;

        DataCopyParams betaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(inType)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(betaLocal, betaGm_[seq0 * NV_], betaInParams, padParams);

        AscendC::PipeBarrier<PIPE_MTE2>();

        in_ready_Beta.set();
        in_ready_Beta.wait();

        Cast(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, bBatchSize);
        AscendC::PipeBarrier<PIPE_V>();
        Sigmoid<float, false>(betaInUb, betaInUb, bBatchSize);
        AscendC::PipeBarrier<PIPE_V>();

        if (hasGama_) {
            DataCopyParams gamaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
            DataCopyPad(gamaLocal, gamaGm_[seq0 * NV_], gamaInParams, padParams);
            AscendC::PipeBarrier<PIPE_MTE2>();

            in_ready_Beta.set();
            in_ready_Beta.wait();
            Exp(gamaLocal, gamaLocal, seqLen * NV_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::TEventID eventIDS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIDS);
        WaitFlag<HardEvent::V_S>(eventIDS);
    }

    __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t stateOffset,
                                       uint64_t recurrentStateOffset)
    {
        qkvcopyFlag_ = false;
        for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
            uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
            uint64_t curStateOffset = ((stateOffset * NV_ + head_i) * realV_ + v_i) * realK_;
            uint64_t curRecurrentOffset = ((recurrentStateOffset * NV_ + head_i) * realV_ + v_i) * realK_;

            CopyInState(curStateOffset, curRecurrentOffset, curSingleV);
            for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                uint64_t curQKOffset = (seq_i - seq0) * alignK_;
                uint64_t curVOffset = (seq_i - seq0) * alignV_ + v_i;
                uint64_t attnOffset = (seq_i * NV_ + head_i) * realV_ + v_i;

                Compute(curSingleV, curQKOffset, curVOffset, seq_i, attnOffset, seq0, seq1, head_i, stateOffset, v_i);
            }
        }
    }

private:
    GlobalTensor<inType> mixqkvGm_;
    GlobalTensor<inType> betaGm_;
    GlobalTensor<float> gamaGm_;
    GlobalTensor<inType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<int32_t> numAcceptedTokensGm_;
    GlobalTensor<outType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    GlobalTensor<inType> recurrentStateGm_;
    GlobalTensor<int32_t> cacheIndicesGm_;

    TPipe *pipe_;

    LocalTensor<inType> qLocal;
    LocalTensor<inType> kLocal;
    LocalTensor<inType> vLocal;
    LocalTensor<float> gamaLocal;
    LocalTensor<inType> betaLocal;
    LocalTensor<inType> stateLocal;
    LocalTensor<outType> stateOutLocal;
    LocalTensor<outType> attnOutLocal;

    TBuf<TPosition::VECCALC> tmpBuff;
    TBuf<TPosition::VECCALC> stageBuff;

    LocalTensor<float> qInUb;
    LocalTensor<float> kInUb;
    LocalTensor<float> vInUb;
    LocalTensor<float> qTempInUb;
    LocalTensor<float> kTempInUb;

    LocalTensor<float> betaInUb;
    LocalTensor<float> deltaInUb;
    LocalTensor<float> broadTmpInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;

    LocalTensor<float> qSumLocal;
    LocalTensor<float> kSumLocal;

    uint32_t B_;
    uint32_t S_;
    uint32_t T_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t vStep_;
    uint32_t restUbSize_;
    uint32_t load;
    uint32_t usedblk;
    uint32_t avgload;

    uint32_t tokenStrideSize_;

    SEvent<HardEvent::MTE2_V> in_ready_Beta;
    SEvent<HardEvent::V_MTE2> in_empty_Beta;
    SEvent<HardEvent::V_MTE3> out_ready_Attn;
    SEvent<HardEvent::MTE3_V> out_empty_Attn;

    bool hasIntermediateState_;
    bool needRecurrentInit_;
    bool hasAcceptedTokens_;
    bool hasGama_;
    float gama_;
    float beta_;
    float scale_;
    uint64_t blockIdx;

    bool qkvcopyFlag_;
};

extern "C" __global__ __aicore__ void recurrent_gated_delta_rule(
    GM_ADDR mixqkv, GM_ADDR beta, GM_ADDR initState, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices,
    GM_ADDR mtpRecurrentState, GM_ADDR cacheIndices, GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out,
    GM_ADDR stateOut, uint32_t b, uint32_t s, uint32_t nk, uint32_t dk, uint32_t nv, uint32_t dv,
    bool hasIntermediateState, bool hasAcceptedTokens, bool hasGama, uint32_t vStep, uint32_t ubRestBytes, float scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    RGDR<bfloat16_t, bfloat16_t> op(b, s, nk, dk, nv, dv, hasIntermediateState, hasAcceptedTokens, hasGama, vStep,
                                    ubRestBytes, scale);

    RGDRInitParams initParams{
        mixqkv,
        g,
        beta,
        initState,
        cuSeqlens,
        ssmStateIndices,
        numAcceptedTokens,
        out,
        stateOut,
        mtpRecurrentState,
        cacheIndices,
    };

    op.Init(initParams, &pipe);
    op.Process();
}

#endif

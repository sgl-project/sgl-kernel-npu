/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_common.h"
#include "lightning_indexer_vector.h"
#include "../op_host/tiling/mlp_lightning_tiling_data.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
using sglang::MlpLIHost::LITilingData;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t MAX_LOGITS_TMP = 0x7F7FFFFF;
constexpr uint32_t LOGITS_SCALE = 4;
template <typename LIT>
class LIVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using K_T = typename LIT::keyType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline LIVector(){};
    __aicore__ inline void ProcessVec(const LICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo &constInfo,
                                      __gm__ const LITilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<float> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);
    
    GlobalTensor<int32_t> initNumGm;
    GlobalTensor<int32_t> localNumGm;
    bool useInitNumTensor = false;
    bool useLocalNumTensor = false;
protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    // max score
    const float MAX_LOGITS = *((float *)&MAX_LOGITS_TMP);
    const float SECOND_MAX_LOGITS = MAX_LOGITS / 2;
    const float CLIP_MAX_LOGITS = MAX_LOGITS / LOGITS_SCALE;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct LICommon::ConstInfo constInfo_;
    event_t eventIdVToMte2A;
    event_t eventIdVToMte2B;
    event_t eventIdMTE2ToV;
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;

    virTopK = constInfo_.sparseCountFlag ? constInfo_.sparseCount : BASE_TOPK;

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);                                            // 32.25KB
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));  // 68KB 
    // pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));      // 64KB
    pipe->InitBuffer(sortOutBuf_, 4 * BASE_TOPK * 2 * sizeof(float));      // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));                          // 4KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float)); //16 * 32 = 512
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);

    // step3. 初始化vec1ParamGm，是否进行LD的标志位设为-1(needFd=-1)
    // vec1ResIn32Gm = [aic, 2, s1BaseSize_, 16] int32
    // ws清零 [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, ......]
    LocalTensor<float> tmpfBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpfBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpfBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(tmpfBuff);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitParams(const struct LICommon::ConstInfo &constInfo,
                                                 __gm__ const LITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    // group ub 切分因子当前按照UB空间强制为16
    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void
LIVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                    GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<float> weightsGm,
                                    GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::AllocEventID()
{
    eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
    
    if (constInfo_.returnValue) {
        uint16_t negInf = 0;
        if constexpr(std::is_same<K_T, float16_t>::value) {
            negInf = 0xFC00;
        } else {
            negInf = 0xFF80;
        }
        LocalTensor<uint16_t> valueULocal = outQueue_.AllocTensor<uint16_t>();
        Duplicate(valueULocal, negInf, constInfo_.sparseCount);
        outQueue_.EnQue<uint16_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint16_t>();
        GlobalTensor<uint16_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
        LIServiceVec::CopyOut(valueOutGmTmp[invalidS1offset], valueULocal, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVec(const LICommon::RunInfo &info)
{
    /// ---- 预处理 ---- ///
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_) * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum = cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;

    /// ---- main loop ---- ///
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();

            /// ---- 处理gemm结果 ---- ///
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                LIServiceVec::CopyIn(dbTmpUb, weightsInUb, mm1ResGm, weightsGm,
                                     mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                         outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign,
                                     weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                                     info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use LIServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            /// kv block indexer处理 ///
            int32_t blockS2LenVecAlign = cuS2LenVecAlign / constInfo_.blockLen;
            int32_t blockS2Len = (cuS2Len + constInfo_.blockLen - 1) / constInfo_.blockLen;
            int32_t blockS2BaseSize = s2BaseSize_ / constInfo_.blockLen;
            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[blockS2LenVecAlign];
            if (constInfo_.blockLen == 2 || constInfo_.blockLen == 4) {
                // 512 -> 512 * 8
                AscendC::Brcb(tmpSortBuf, reduceOutInner, cuS2LenVecAlign / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                // block reduce: blockLen->1, here blockLen*8->1
                // 用高阶api Sum对s2维度reduce(不支持源操作数与目的操作数地址重叠)
                AscendC::SumParams params;
                params.outter = cuS2LenVecAlign * 8 / (constInfo_.blockLen * 8);
                params.n = (constInfo_.blockLen * 8);
                params.inner = (constInfo_.blockLen * 8);
                // 得到 cuS2LenVecAlign / BLOCK_LEN个结果
                AscendC::Sum(reduceOutInner, tmpSortBuf, tmpUb_.template ReinterpretCast<uint8_t>(), params);
                PipeBarrier<PIPE_V>();
                AscendC::Muls(reduceOutInner, reduceOutInner, static_cast<float>(1.0f / (8 * constInfo_.blockLen)), blockS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            } else if (constInfo_.blockLen > 4) { // blockLen is 8 or 16
                // 用高阶api Sum对s2维度reduce(不支持源操作数与目的操作数地址重叠)
                AscendC::SumParams params;
                params.outter = blockS2LenVecAlign;
                params.n = constInfo_.blockLen;
                params.inner = constInfo_.blockLen;
                // 得到 cuS2LenVecAlign / BLOCK_LEN个结果
                AscendC::Sum(tmpSortBuf, reduceOutInner, tmpUb_.template ReinterpretCast<uint8_t>(), params);
                PipeBarrier<PIPE_V>();
                AscendC::Muls(reduceOutInner, tmpSortBuf, static_cast<float>(1.0f / constInfo_.blockLen), blockS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            // 对logits做clip，避免冲撞SECOND_MAX_LOGITS
            AscendC::Mins(reduceOutInner, reduceOutInner, CLIP_MAX_LOGITS, blockS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            
            /// ---- sink & swa处理 ---- ///
            // 1. 指定head区域全选，赋予第二高分数
            // if (info.s2Idx == 0 && constInfo_.initNum > 0) {
            //     uint32_t blockInitNum = constInfo_.initNum / constInfo_.blockLen;
            //     Duplicate(reduceOutInner, SECOND_MAX_LOGITS, blockInitNum);
            //     PipeBarrier<PIPE_V>();
            // }
            uint32_t curInitNum = useInitNumTensor 
                ? static_cast<uint32_t>(initNumGm.GetValue(info.bIdx)) 
                : static_cast<uint32_t>(constInfo_.initNum);
            if (info.s2Idx == 0 && curInitNum > 0) {
                uint32_t blockInitNum = curInitNum / constInfo_.blockLen;
                Duplicate(reduceOutInner, SECOND_MAX_LOGITS, blockInitNum);
                PipeBarrier<PIPE_V>();
            }
            // 2. 指定tail区域全选，赋予第二高分数
            // 当前处理块踩到了local 部分
            // int32_t localStart = cuRealAcSeq - constInfo_.localNum;
            // bool hasLocal = cuBaseS2Idx > localStart || cuBaseS2Idx + cuS2Len > localStart;
            // if (constInfo_.localNum > 0 && hasLocal) {
            uint32_t curLocalNum = useLocalNumTensor
                ? static_cast<uint32_t>(localNumGm.GetValue(info.bIdx))
                : static_cast<uint32_t>(constInfo_.localNum);
            int32_t localStart = cuRealAcSeq - curLocalNum;
            bool hasLocal = cuBaseS2Idx > localStart || cuBaseS2Idx + cuS2Len > localStart;
            if (curLocalNum > 0 && hasLocal) {
                // 部分落在local 区间
                if (cuBaseS2Idx < localStart) {
                    int32_t nonLocalBlockNum = (localStart - cuBaseS2Idx + constInfo_.blockLen - 1) / constInfo_.blockLen;
                    int32_t localBlockNum = blockS2Len - nonLocalBlockNum;
                    if (nonLocalBlockNum % 8 == 0) {
                        Duplicate(reduceOutInner[nonLocalBlockNum], SECOND_MAX_LOGITS, localBlockNum);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Duplicate(tmpSortBuf, SECOND_MAX_LOGITS, blockS2Len);
                        PipeBarrier<PIPE_V>();
                        Adds(tmpSortBuf, reduceOutInner, 0.0f, nonLocalBlockNum);
                        PipeBarrier<PIPE_V>();
                        Adds(reduceOutInner, tmpSortBuf, 0.0f, blockS2Len);
                        PipeBarrier<PIPE_V>();
                    }
 
                } else {  // 当前block 全落在local 区间
                    Duplicate(reduceOutInner, SECOND_MAX_LOGITS, blockS2Len);
                    PipeBarrier<PIPE_V>();   
                }
            }
            // 3. 最后一个block必选，赋予最高分数
            // if (isS2End) {
            //     reduceOutInner.SetValue(blockS2Len - 1, MAX_LOGITS);
            //     SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            // }

            /// ---- 整理score & indice ---- ///
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, blockS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, blockS2Len);
            PipeBarrier<PIPE_V>();

            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            if (blockS2LenVecAlign != blockS2Len) {
                Duplicate(sortIndiceUbInt, -1, blockS2LenVecAlign); // 无效数据索引填充为-1
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx / constInfo_.blockLen), blockS2Len);
            PipeBarrier<PIPE_V>();

            /// ---- 全排序 ---- ///
            LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf, blockS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
            PipeBarrier<PIPE_V>();
            LIServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff, blockS2LenVecAlign, tmpUb_);
            
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A); // 反向同步，下一轮循环开始位置会用到 tmpUb_
            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            /// ---- 排序结果搬出 ---- ///
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;
            if (needCopyOutGm) {
                // 对角线block当前在队首位置，我们把它放回到对角线上
                // uint32_t validS2Block = (cuRealAcSeq + constInfo_.blockLen - 1) / constInfo_.blockLen;
                // validS2Block = validS2Block < constInfo_.sparseCount ? validS2Block : constInfo_.sparseCount;
                // SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                // float lhsValue = globalTopkUb_[innerS1Idx * virTopK * 2].GetValue(0);
                // int32_t lhsIdx = (globalTopkUb_[innerS1Idx * virTopK * 2].template ReinterpretCast<int32_t>()).GetValue(1);
                // float rhsValue = globalTopkUb_[innerS1Idx * virTopK * 2].GetValue(2 * validS2Block - 2);
                // int32_t rhsIdx = (globalTopkUb_[innerS1Idx * virTopK * 2].template ReinterpretCast<int32_t>()).GetValue(2 * validS2Block - 1);
                // globalTopkUb_[innerS1Idx * virTopK * 2].SetValue(0, rhsValue);
                // (globalTopkUb_[innerS1Idx * virTopK * 2].template ReinterpretCast<int32_t>()).SetValue(1, rhsIdx);
                // globalTopkUb_[innerS1Idx * virTopK * 2].SetValue(2 * validS2Block - 2, lhsValue);
                // (globalTopkUb_[innerS1Idx * virTopK * 2].template ReinterpretCast<int32_t>()).SetValue(2 * validS2Block - 1, lhsIdx);
                // SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

                // 结果提取 & 搬出
                int64_t offset = (constInfo_.sparseCount <= 4096) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen = (constInfo_.sparseCount <= 4096) ? constInfo_.sparseCount : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= 4096) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset], (offset / 32));
                    
                    LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
                    if (constInfo_.returnValue) {
                        PipeBarrier<PIPE_V>();
                        AscendC::Mins(outValueUb, outValueUb, 65504.0f, copyLen); 
						PipeBarrier<PIPE_V>();
                        Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, copyLen);
                        PipeBarrier<PIPE_V>();
                    }

                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();
                    
                    LIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset], idxULocal1, copyLen);
                    if (constInfo_.returnValue) {
                        LIServiceVec::CopyOut(valueOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset], valueULocal1, copyLen);
                    }
                    outQueue_.FreeTensor(outValueUb);
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +       // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +       // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                // tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                // tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                // tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                // tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                // 写入头尾判断
                // [head, tail]
                // head: 与前面规约，与前后规约
                // tail: 与后面规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则 blockS2StartIdx_ != 0
                // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
                // 跟后面块做规约 写到1偏移  需要 + s1BaseSize_, BASE_TOPK*2
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    // int64_t s2Start;
    // int64_t s2End;
    int64_t isS2End;
    // int64_t bn2Idx;
    // int64_t s1Idx;
    uint32_t acc_list_num = 0;
    // int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    // int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        // s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        // 对角线block当前在队首位置，我们把它放回到对角线上
        // s2ActSeq = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_ + 1);
        // uint32_t validS2Block = (s2ActSeq + constInfo_.blockLen - 1) / constInfo_.blockLen;
        // validS2Block = validS2Block < constInfo_.sparseCount ? validS2Block : constInfo_.sparseCount;
        // SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        // float lhsValue = curValueIdxUb.GetValue(0);
        // int32_t lhsIdx = (curValueIdxUb.template ReinterpretCast<int32_t>()).GetValue(1);
        // float rhsValue = curValueIdxUb.GetValue(2 * validS2Block - 2);
        // int32_t rhsIdx = (curValueIdxUb.template ReinterpretCast<int32_t>()).GetValue(2 * validS2Block - 1);
        // curValueIdxUb.SetValue(0, rhsValue);
        // (curValueIdxUb.template ReinterpretCast<int32_t>()).SetValue(1, rhsIdx);
        // curValueIdxUb.SetValue(2 * validS2Block - 2, lhsValue);
        // (curValueIdxUb.template ReinterpretCast<int32_t>()).SetValue(2 * validS2Block - 1, lhsIdx);
        // SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        if (!constInfo_.returnValue) {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            AscendC::Mins(outValueUb, outValueUb, 65504.0f, constInfo_.sparseCount);  
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
            Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, constInfo_.sparseCount);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            DataCopyPad(valueOutGm[outOffset], valueULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(K_T)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace LIKernel
#endif

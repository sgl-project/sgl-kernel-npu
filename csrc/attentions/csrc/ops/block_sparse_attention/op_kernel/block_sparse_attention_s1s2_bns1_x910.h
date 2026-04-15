/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */


#ifndef BLOCK_SPARSE_ATTENTION_S1S2_BNS1_X910_H
#define BLOCK_SPARSE_ATTENTION_S1S2_BNS1_X910_H

#include "block_sparse_attention_s1s2_bns1_x910_base.h"

using namespace matmul;
template<typename BSAT>
class BlockSparseAttentionS1s2Bns1X910 : public BlockSparseAttentionS1s2Bns1X910Base<BSAT> {
public:
    using FT = float;
    using T = typename BSAT::inputType;
    using KV_T = typename BSAT::kvInputType;
    using U = typename BSAT::maskType;
    using O = typename BSAT::outputType;
    using mmOutputTypeTmp = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::mmOutputType;
    using computeType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::softmaxType;
    using pseShiftType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::pseShiftCastType;
    using MM_IN_T = typename AscendC::Conditional<BSAT::msdMode == MsdMode::MSD_ON, KV_T, T>::type;
    using mmOutputType =
        typename AscendC::Conditional<BSAT::msdMode == MsdMode::MSD_ON, int32_t, mmOutputTypeTmp>::type;
    static constexpr FT BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0f;
    constexpr static bool USE_BLOCK_SPARE = BSAT::MM_TYPE == MatMulType::MM_SP;
    static constexpr int32_t BSA_PARAMS_QUEUE_CAPBABILITY = 4;

    __aicore__ inline BlockSparseAttentionS1s2Bns1X910() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AllocGlobalResources();

    __aicore__ inline void FreeGlobalResources();

    __aicore__ inline void Bmm1ResDoVecBmm2Compute();

    __aicore__ inline void ComputeEachCoreSInnerLoop();

    __aicore__ inline void SInnerLoopFunc(int curBatch, int32_t sparseBlockCount, uint32_t sparseMaskIdRowOffset,
        int32_t lastSparseBlockId);
    __aicore__ inline void ComputeEachCore(int32_t coreIdx);
    __aicore__ inline void  GetTaskCost(uint64_t sparseBlockCountOffsetBase, uint32_t sOuterBlockNum);
    __aicore__ inline int TaskAlloc(const int32_t coreIdx, int &allocatedQtaskAllCore, const int sOuterBlockNum,
        const int coreNum, uint32_t bid, uint32_t nid);
    __aicore__ inline void InitEachCoreWorkspace(uint32_t coreIdx, int32_t blockNum);
    __aicore__ inline void CalcSparsePos(TBuf<>& tbuf, uint32_t srcOffset, uint32_t s1Count, uint32_t s2,
        uint32_t realS2);

    __aicore__ inline void ProcessLastSouterLoopFinalRes(const LocalTensor<computeType>& bmm2tmpUb)
    {
        if (!this->copyOutPrevIter) {
            return;
        }
        auto souterSize = this->preHeadParams->singleProcessSOuterSize;
        auto bmm2ResPreUb = this->preBmm2Buf.template Get<computeType>();
        if (this->preHeadParams->isFirstInnerIter) {
            Duplicate(bmm2ResPreUb, static_cast<computeType>(0), this->bmm2ResUbSize);
            pipe_barrier(PIPE_V);
        }
        this->bmm2.WaitIterateAll();

        DataCopy(bmm2tmpUb, this->bmm2ResGmDb[this->preHeadParams->gmPingpong], souterSize * this->headSize);
        SetFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[this->preHeadParams->gmPingpong]);
        WaitFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[this->preHeadParams->gmPingpong]);
        Add(bmm2ResPreUb, bmm2ResPreUb, bmm2tmpUb, souterSize * this->headSize);
        pipe_barrier(PIPE_V);
        LocalTensor<float> softmaxSumTmp = this->softmaxExpBuf.template Get<float>();
        this->Bmm2UpdateDivNoTail(bmm2ResPreUb, softmaxSumTmp);
        
        if ((BSAT::layout == BSALayout::BSH) ||
            (BSAT::layout == BSALayout::BNSD && this->tilingData->promptAttentionBaseParams.isBSNDOut == 1)) {
            this->DataCopyTransposeOutBSH(bmm2ResPreUb);
        } else {
            this->DataCopyTransposeOutBNSD(bmm2ResPreUb);
        }
        this->copyOutPrevIter = false;
    }

    static __aicore__ inline void Mask2Pos(const LocalTensor<int32_t>& sparseId, const LocalTensor<int32_t>& index,
        const LocalTensor<int8_t>& sparseMask, uint32_t count, LocalTensor<uint8_t>& tmpUb)
    {
        auto alignCount = ((count + 32 - 1) / 32) * 32;
        auto maskHalf = tmpUb.template ReinterpretCast<half>(); // 2
        Cast(maskHalf, sparseMask, RoundMode::CAST_NONE, alignCount); // 11
        pipe_barrier(PIPE_V);
        auto maskFloat = maskHalf[alignCount].template ReinterpretCast<float>(); // 4
        Cast(maskFloat, maskHalf, RoundMode::CAST_NONE, alignCount);
        pipe_barrier(PIPE_V);
        auto sortedLocal = maskFloat[alignCount]; // 8
        auto sortTmpLocal = sortedLocal[alignCount * 2];
        AscendC::Sort<float, true>(sortedLocal, maskFloat, index.template ReinterpretCast<uint32_t>(),
            sortTmpLocal, alignCount / 32);
        pipe_barrier(PIPE_V);
        auto dstIndexLocal = sparseId.template ReinterpretCast<uint32_t>();
        AscendC::Extract<float>(maskFloat, dstIndexLocal, sortedLocal, alignCount / 32);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Bmm1ComputeIterate(BSAComputeParam *params)
    {
        if (this->mm1SingleCoreNPrev != params->mm1SingleCoreN || this->isGlobalFirstCompute) {
            this->mm.SetOrgShape(params->singleProcessSOuterSize, params->mm1SingleCoreN,
                this->tilingData->bmm1TilingDataRect.Ka, this->tilingData->bmm1TilingDataRect.Kb,
                params->mm1SingleCoreN);
            this->mm1SingleCoreNPrev = params->mm1SingleCoreN;
        }
        // * for the begin of each row, sparseMaskPosOffset = sparsePosOffset
        // * mm datacopy  from pos.getvalue(sparseMaskPosOffset)
        // to pos.getvalue(sparseMaskPosOffset + copyOnceBlockCount - 1) kv中的第几个block
        // * tensorBcoreOffset + 第几个block  * blocksize * h or d
        // * sparsePos_ptr 是global的ptr
        if constexpr (USE_BLOCK_SPARE) {
            this->bmm1LocalInfo = this->PABmm1UB.template Get<uint32_t>();
            uint32_t ii = 0;
            if (this->isGlobalFirstCompute) {
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->sparseBlockSize));
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->headNumSize / this->headNumRatio));
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->headSize));
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm1TilingDataRect.baseN));
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm1TilingDataRect.baseK));
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm1TilingDataRect.Kb));
                this->bmm1LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm2TilingDataRect.N));
            }
            for (int j = 0; j < 8; j++) { // 循环8次
                this->bmm1LocalInfo.SetValue(8 + j, params->pos[j]); // pos[j]设置为8+j
            }

            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            DataCopy(this->bmm1CBDataGm[params->gmPingpong], this->bmm1LocalInfo, 16); // 一次搬运16个

            event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            this->mm.SetSelfDefineData(reinterpret_cast<int64_t>(this->bmm1CBDataPtr[params->gmPingpong]));
        }

        this->mm.SetTail(params->singleProcessSOuterSize, params->singleProcessSInnerBmmTail);
        this->mm.SetTensorA(this->queryGm[params->tensorAOffset]);
        this->mm.SetTensorB(this->keyGm[params->tensorBCoreOffset], true);

        this->mm.template IterateAll<false>(this->bmm1ResGmDb[params->gmPingpong], false, false, true,
            params->singleProcessSOuterSize == 0);
    }

    template <bool nonFirst>
    __aicore__ inline  void  SoftmaxCompute(const LocalTensor<computeType>& mmResUb,
                                            const LocalTensor<float>& softmaxMaxUb,
                                            const LocalTensor<float>& softmaxSumUb,
                                            const LocalTensor<computeType>& softmaxExpMaxUb,
                                            const LocalTensor<uint8_t>& tmpUb, uint32_t souterSize, uint32_t colCount,
                                            uint32_t actColCount)
    {
        SoftMaxShapeInfo softmaxShapeInfo {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(colCount),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(actColCount)
        };
        SoftMaxTiling& softmaxFlashTilingData = this->softmaxSouterStepLen == this->softmaxFlashTilingData.srcM ?
            this->softmaxFlashTilingData : this->softmaxFlashTilingDataNew;
        SoftmaxFlashV2<computeType, nonFirst, true, false>(mmResUb, softmaxSumUb, softmaxMaxUb,
            mmResUb, softmaxExpMaxUb, softmaxSumUb, softmaxMaxUb, tmpUb, softmaxFlashTilingData, softmaxShapeInfo);
    }

    __aicore__ inline uint64_t CalcSparseMaskOffset(int bIdx)
    {
        auto s1 = this->sparseMaskS1Size;
        auto s2 = this->sparseMaskS2Size;
        auto N =  this->headNumSize;
        uint64_t batchOffset = bIdx * N * s1 * s2;
        return batchOffset + this->tailParams->batchNOffset * s1 * s2;
    }

    __aicore__ inline uint64_t CalcSparseMaskCountOffset(int bIdx)
    {
        auto s1 = this->sparseMaskS1Size;
        auto N =  this->headNumSize;
        uint64_t batchOffset = bIdx * N * s1;
        return batchOffset + this->tailParams->batchNOffset * s1 ;
    }

    __aicore__ inline void Res1VecCompute(BSAComputeParam *params)
    {
        constexpr uint16_t byteNum = BYTE_BLOCK_BSA / sizeof(computeType);
        uint32_t actColCount = params->singleProcessSInnerBmmTail;
        auto colCount = params->mm1SingleCoreN; // ! for isLastBlockCausalDiag
        auto singleProcessSOuterSize = params->singleProcessSOuterSize;
        auto gmPingpong = params->gmPingpong;
        auto isFirstInnerIter = params->isFirstInnerIter;
        auto isLastBlockCausalDiag = params->isLastBlockCausalDiag;

        //* 避免在softmax内部check tiling == false 反复更新softmaxFlashTilingData
        if (this->softmaxSouterStepLen != this->softmaxFlashTilingData.srcM) {
                SoftMaxShapeInfo softmaxShapeInfo {
                static_cast<uint32_t>(this->softmaxSouterStepLen),
                static_cast<uint32_t>(colCount),
                static_cast<uint32_t>(this->softmaxSouterStepLen),
                static_cast<uint32_t>(actColCount)
            };
            constexpr uint32_t tmpBuffSize = sizeof(computeType) == 2 ? BSA_TEMP_BUFFER_SIZE_BYTE :
                BSA_TEMP_BUFFER_SIZE_BYTE / 2;
            this->softmaxFlashTilingDataNew =
                SoftMaxFlashV2TilingFunc(softmaxShapeInfo, sizeof(computeType), sizeof(float),
                    tmpBuffSize, true, false);
        }
        auto maskValue = static_cast<computeType>(sizeof(computeType) == sizeof(float) ?
            BOOL_ATTEN_MASK_SCALAR_VALUE : -10000.0f);

        int ubPingpong = 0;
        for (int64_t souterOffset = 0, mm1ResGmOffset = 0, nextSouterOffset = 0; souterOffset < singleProcessSOuterSize;
            souterOffset = nextSouterOffset) {     // Pending rectification
            int64_t remainSouterSize = singleProcessSOuterSize - souterOffset;
            int64_t souterSize = (remainSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen :
                remainSouterSize;
            nextSouterOffset = souterOffset + this->softmaxSouterStepLen;
            int64_t nextLeftSouterSize = singleProcessSOuterSize - nextSouterOffset;
            int64_t nextSouterSize = (nextLeftSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen :
                nextLeftSouterSize;
            auto nextMm1ResGmOffset = mm1ResGmOffset + souterSize * colCount;
            uint32_t computeSize = souterSize * colCount;

            auto softmaxMaxUb_ = this->softmaxMaxUb[souterOffset * 8];
            auto softmaxSumUb_ = this->softmaxSumUb[souterOffset * 8];
            auto softmaxExpUb_ = this->softmaxExpUb[souterOffset * byteNum];

            WaitFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);
            Muls(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong],
                static_cast<computeType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
            pipe_barrier(PIPE_V);

            if (isLastBlockCausalDiag) {
                auto &dst = this->mmResUb[ubPingpong];
                auto maskTensor = this->tmpBuff.template Get<computeType>();
                for (auto j = 0; j < souterSize; j++) {
                    auto i = j + souterOffset;
                    auto ubOffset = (j+1) * colCount - this->sparseBlockSize;
                    Duplicate(maskTensor, maskValue, this->sparseBlockSize);
                    pipe_barrier(PIPE_V);
                    Duplicate(maskTensor, static_cast<computeType>(0.0f), i+1);
                    pipe_barrier(PIPE_V);
                    Add(maskTensor, maskTensor, dst[ubOffset], i+1);
                    pipe_barrier(PIPE_V);
                    DataCopy(dst[ubOffset], maskTensor, this->sparseBlockSize);
                    pipe_barrier(PIPE_V);
                }
            }
            if (isFirstInnerIter) {
                SoftmaxCompute<false>(this->mmResUb[ubPingpong], softmaxMaxUb_,  softmaxSumUb_, softmaxExpUb_,
                    this->tmpBuff.template Get<uint8_t>(), souterSize, colCount, actColCount);
            } else {
                SoftmaxCompute<true >(this->mmResUb[ubPingpong], softmaxMaxUb_,  softmaxSumUb_, softmaxExpUb_,
                    this->tmpBuff.template Get<uint8_t>(), souterSize, colCount, actColCount);
            }
            if (nextSouterOffset < singleProcessSOuterSize) { // noLastSoftmaxLoop
                auto &srcGm = this->bmm1ResGmDb[gmPingpong];
                WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong^1]);
                DataCopy(this->mmResUb[ubPingpong^1], srcGm[nextMm1ResGmOffset],  nextSouterSize * colCount);
                SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong^1]);
            }
            auto mmResUbOut = this->mmResUb[ubPingpong].template ReinterpretCast<T>();
            if constexpr (sizeof(computeType) == 4) {
                pipe_barrier(PIPE_V);
                Cast(mmResUbOut, this->mmResUb[ubPingpong], RoundMode::CAST_ROUND, computeSize);
            }
            this->Bmm1Queue.EnQue(this->mmResUb[ubPingpong]);
            this->Bmm1Queue.template DeQue<computeType>();

            // !todo ： bf16 to check
            if constexpr (sizeof(computeType) == 2) {
                DataCopy(this->bmm1ResGmDb[gmPingpong][mm1ResGmOffset],
                    this->mmResUb[ubPingpong], souterSize * colCount);
            } else {
                DataCopy(this->softmaxResGmDb[gmPingpong][mm1ResGmOffset], mmResUbOut, souterSize * colCount);
            }
            SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
            mm1ResGmOffset = nextMm1ResGmOffset;
            ubPingpong ^= 1;
        }
        // ! from dcci // maybe?
        // ? why
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    __aicore__ inline void  Bmm1VecInputCopyIn()
    {
        BSAComputeParam *params = this->headParams;
        auto softmaxColCount = params->mm1SingleCoreN;
        this->softmaxSouterStepLen = this->tilingData->promptAttentionBaseParams.softmaxOuterSize;
        if (this->softmaxFlashTilingData.srcK != softmaxColCount) {
            // 8 alignment
            uint32_t minSoftmaxSouterStepLen = this->softmaxFlashTilingData.srcSize / softmaxColCount / 8 * 8;
            this->softmaxSouterStepLen = ((minSoftmaxSouterStepLen > params->singleProcessSOuterSize) ||
                (minSoftmaxSouterStepLen == 0) ? params->singleProcessSOuterSize : minSoftmaxSouterStepLen);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[0]);
        DataCopy(this->mmResUb[0], this->bmm1ResGmDb[params->gmPingpong],
            this->softmaxSouterStepLen * softmaxColCount);
        SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[0]);
    }

    __aicore__ inline void Bmm2ComputeIterate()
    {
        BSAComputeParam *params = this->headParams;
        uint32_t mm2KaStride = params->mm1SingleCoreN;

        if ((this->mm2MStridePrev != params->singleProcessSOuterSize) || (this->mm2KaStridePrev != mm2KaStride) ||
            this->isGlobalFirstCompute) {
            this->bmm2.SetOrgShape(params->singleProcessSOuterSize, this->headSize,  mm2KaStride,
                this->tilingData->bmm2TilingDataRect.Kb, this->headSize);
            this->mm2MStridePrev = params->singleProcessSOuterSize;
            this->mm2KaStridePrev = mm2KaStride;
        }

        // *mm datacopy  from pos.getvalue(sparseMaskPosOffset)
        // to pos.getvalue(sparseMaskPosOffset + copyOnceBlockCount - 1) kv中的第几个block
        // *tensorBcoreOffset + 第几个block  * blocksize * h or d
        //* sparsePos_ptr 是global的ptr
        if constexpr (USE_BLOCK_SPARE) {
            this->bmm2LocalInfo = this->PABmm2UB.template Get<uint32_t>();
            uint32_t ii = 0;
            if (this->isGlobalFirstCompute) {
                // fixed parameters 9
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->sparseBlockSize));
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->headNumSize / this->headNumRatio));
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->headSize));
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm2TilingDataRect.baseN));
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm2TilingDataRect.baseK));
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm1TilingDataRect.Kb));
                this->bmm2LocalInfo.SetValue(ii++, static_cast<uint32_t>(this->tilingData->bmm2TilingDataRect.N));
            }
            for (int j = 0; j < 8; j++) { // 循环8次
                this->bmm2LocalInfo.SetValue(8+j, params->pos[j]); // params->pos[j]设置为8
            }
            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            DataCopy(this->bmm2CBDataGm[params->gmPingpong], this->bmm2LocalInfo, 16); // 16: 搬运个数
            event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

            this->bmm2.SetSelfDefineData(reinterpret_cast<int64_t>(this->bmm2CBDataPtr[params->gmPingpong]));
        }
        this->bmm2.SetTail(params->singleProcessSOuterSize, this->headSize, params->singleProcessSInnerBmmTail);
        if constexpr (sizeof(mmOutputType) == 2) {
            this->bmm2.SetTensorA(this->bmm1ResGmDb[params->gmPingpong]);
        } else {
            this->bmm2.SetTensorA(this->softmaxResGmDb[params->gmPingpong]);
        }
        this->bmm2.SetTensorB(this->valueGm[params->tensorBCoreOffset]);
        this->bmm2.template IterateAll<false>(this->bmm2ResGmDb[params->gmPingpong], false, false, true,
            params->singleProcessSOuterSize == 0);
    }

    __aicore__ inline void CopyParamsAttrOutOfInnerLoop(BSAComputeParam *dst, BSAComputeParam *src)
    {
        dst->sInnerLoopIdx = src->sInnerLoopIdx; //* debug 专用 可以删掉
        dst->isFirstInnerIter = src->isFirstInnerIter;
        dst->isSecondInnerIter = src->isSecondInnerIter;
        dst->isLastInnerIter = src->isLastInnerIter;
        dst->isLastBlockCausalDiag = src->isLastBlockCausalDiag;

        dst->singleProcessSOuterSize = src->singleProcessSOuterSize;
        dst->singleProcessSInnerSize = src->singleProcessSInnerSize;
        dst->singleProcessSInnerBmmTail = src->singleProcessSInnerBmmTail;
        dst->mm1SingleCoreN = src->mm1SingleCoreN;

        dst->tensorAOffset = src->tensorAOffset;
        dst->tensorBCoreOffset = src->tensorBCoreOffset;
        dst->attentionOutOffset = src->attentionOutOffset;
        dst->sOuterOffset = src->sOuterOffset;
        dst->batchNOffset = src->batchNOffset;
        dst->multiSeqOffset = src->multiSeqOffset;
        dst->multiSeqOffsetBSNDOut = src->multiSeqOffsetBSNDOut;

        dst->actualSeqLengthPerBatch = src->actualSeqLengthPerBatch;
        dst->actualSeqLengthKVPerBatch = src->actualSeqLengthKVPerBatch;
        dst->sparseBlockCount = src->sparseBlockCount;
        for (auto i = 0; i < 8; i++) { // 循环8次
            dst->pos[i] = src->pos[i];
        }
    }
};

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::AllocGlobalResources()
{
    for (int i = 0; i < 2; ++i) {
        this->mmResUb[i] = this->Bmm1Queue.template AllocTensor<computeType>(); // enque deque 仅仅用于SoftmaxResCopyOut
    }
    for (int i = 0; i < 2; ++i) {
        this->bmm1ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->bmm1ResCopyOutEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        this->bmm2ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
    }
    this->attenOutCopyOut = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxSumUb = this->softmaxMaxUb[this->tilingData->promptAttentionTensorSizeRect.softmaxMaxSize];
}

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::FreeGlobalResources()
{
    this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb);

    for (int i = 0; i < 2; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(this->attenOutCopyOut);
    for (int i = 0; i < 2; ++i) {
        this->Bmm1Queue.FreeTensor(this->mmResUb[i]);
    }
}

// * QK0	QK1	PV0	QK2	PV1	QK3	PV2	PV3
// *	SM0	SM1	UO0	SM2	UO1	SM3	UO2	UO3
template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::Process()
{
    this->isGlobalFirstCompute = true;
    AllocGlobalResources();
    ComputeEachCore(static_cast<int32_t>(this->tmp_block_idx));
    
    // * Clear the remaining parameters of the queue.
    while (this->queSize > 0) {
        this->queSize--;
        ComputeEachCoreSInnerLoop();
        this->preHeadParams = this->headParams;
        this->headId = (this->headId + 1) % BSA_PARAMS_QUEUE_CAPBABILITY;
        this->headParams = &this->bsaParamsQueue[this->headId];
    }
    ProcessLastSouterLoopFinalRes(this->mmResUb[0]);
    FreeGlobalResources();
}

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::Bmm1ResDoVecBmm2Compute()
{
    BSAComputeParam *params = this->headParams;
    uint32_t resShapeSize = params->singleProcessSOuterSize * this->headSize;
    auto& bmm2tmpUb = this->mmResUb[0];
    LocalTensor<computeType> bmm2ResPreUb = this->preBmm2Buf.template Get<computeType>();
    this->Res1VecCompute(params);
    if (params->isFirstInnerIter) {
        ProcessLastSouterLoopFinalRes(bmm2tmpUb);
        if (this->queSize >= 0) {
            this->Bmm2ComputeIterate();
        }
    } else {
        // ! dealing last innerloop
        if (params->isSecondInnerIter) {
            // ! so clear bmm2ResPreUb
            Duplicate(bmm2ResPreUb, static_cast<computeType>(0), this->bmm2ResUbSize);
            pipe_barrier(PIPE_V);
        }
        this->bmm2.WaitIterateAll();
        DataCopy(bmm2tmpUb, this->bmm2ResGmDb[params->gmPingpong ^ 1], resShapeSize);
        SetFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[params->gmPingpong ^ 1]);
        WaitFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[params->gmPingpong ^ 1]);
        Add(bmm2ResPreUb, bmm2tmpUb, bmm2ResPreUb, this->bmm2ResUbSize);
        pipe_barrier(PIPE_V);
        if (this->queSize >= 0) {
            this->Bmm2ComputeIterate();
        }
        this->UpdateVmul(this->softmaxExpUb);
    }
    if (params->isLastInnerIter) {
        LocalTensor<float> softmaxSumTmp = this->softmaxExpBuf.template Get<float>(this->softmaxSumSize);
        DataCopy(softmaxSumTmp, this->softmaxSumUb, this->softmaxSumSize);
        pipe_barrier(PIPE_V);
        this->copyOutPrevIter = true;
    }
}


template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::SInnerLoopFunc(int curBatch, int32_t sparseBlockCount,
    uint32_t sparseMaskIdRowOffset, int32_t lastSparseBlockId)
{
    constexpr uint32_t softmaxInnerBasicSize = 64;
    BSAComputeParam *&params = this->tailParams;
    auto basicSInnerSize = params->singleProcessSInnerSize; // 1024
    int32_t exceedKvSize = static_cast<int32_t>((lastSparseBlockId + 1) * this->sparseBlockSize) -
        static_cast<int32_t>(params->actualSeqLengthKVPerBatch);
    exceedKvSize = exceedKvSize <= 0 ? 0: exceedKvSize; // 此block行的尾块block超过kvlen的长度
    int32_t computeSize = static_cast<int32_t>(sparseBlockCount * this->sparseBlockSize) - exceedKvSize; // important
    auto innerLoopTimes = (computeSize + basicSInnerSize - 1) / basicSInnerSize;
    params->tensorAOffset = this->tensorACoreOffset;
    params->tensorBCoreOffset = this->tensorBCoreOffset;
    params->sparseBlockCount = sparseBlockCount;
    for (auto sInnerLoopIdx = 0; sInnerLoopIdx < innerLoopTimes; sInnerLoopIdx++) {
        params->isLastBlockCausalDiag = this->causal &&
            (lastSparseBlockId * this->sparseBlockSize == params->sOuterOffset) &&
            (sInnerLoopIdx == innerLoopTimes - 1);  // 是否被对角线穿过
        params->sInnerLoopIdx = sInnerLoopIdx;
        params->isFirstInnerIter = (sInnerLoopIdx == 0);
        params->isSecondInnerIter = (sInnerLoopIdx == 1);
        params->isLastInnerIter = (sInnerLoopIdx == innerLoopTimes - 1);
        //* actualsingleProcessSInnerSize
        params->singleProcessSInnerBmmTail = params->isLastInnerIter ? computeSize - sInnerLoopIdx * basicSInnerSize :
            basicSInnerSize;
        //* actualsingleProcessSInnerSizeAlign
        params->mm1SingleCoreN = Align(params->singleProcessSInnerBmmTail, this->sparseBlockSize);

        auto sparseMaskPosOffset =  sInnerLoopIdx * basicSInnerSize / this->sparseBlockSize; // + sparseMaskIdRowOffset
        auto copyOnceBlockCount = (params->singleProcessSInnerBmmTail + this->sparseBlockSize - 1) /
            this->sparseBlockSize;
        // * mm datacopy  from pos.getvalue(sparseMaskPosOffset)
        // to pos.getvalue(sparseMaskPosOffset + copyOnceBlockCount - 1)
        auto eid_vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eid_vs);
        WaitFlag<HardEvent::V_S>(eid_vs);
        for (auto j = 0; j < 8; j++) {
            params->pos[j] = this->blockIdUb.GetValue(sparseMaskPosOffset + j);
        }
        if (this->queSize >= this->queSizeLimit) {
            ComputeEachCoreSInnerLoop();
            this->preHeadParams = this->headParams;
            this->headId = (this->headId + 1) % BSA_PARAMS_QUEUE_CAPBABILITY;
            this->headParams = &this->bsaParamsQueue[this->headId];
           
            // tail join the queue
            this->tailId = (this->tailId + 1) % BSA_PARAMS_QUEUE_CAPBABILITY;
            BSAComputeParam *nextTailParams = &this->bsaParamsQueue[this->tailId];
            if (sInnerLoopIdx < BSA_PARAMS_QUEUE_CAPBABILITY - 1) {
                this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
            }
            nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
            this->tailParams = nextTailParams; //!  this->tailParams  isnot fixed
        } else { // tail join the queue
            this->tailId = (this->tailId + 1) % BSA_PARAMS_QUEUE_CAPBABILITY;
            BSAComputeParam *nextTailParams = &this->bsaParamsQueue[this->tailId];
            this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
            nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
            this->tailParams = nextTailParams; //!  this->tailParams  isnot fixed
            this->queSize++;
        }
    }
}

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::ComputeEachCoreSInnerLoop()
{
    BSAComputeParam *params = this->headParams;
    BSAComputeParam *nextParams = &(this->bsaParamsQueue[(this->headId + 1) % BSA_PARAMS_QUEUE_CAPBABILITY]);
    if (this->isGlobalFirstCompute) {
        this->Bmm1ComputeIterate(params);
    }
    this->mm.WaitIterateAll();
    Bmm1VecInputCopyIn();
    if (this->queSize > 0) {
        this->Bmm1ComputeIterate(nextParams);
    }
    Bmm1ResDoVecBmm2Compute();
    this->isGlobalFirstCompute = false;
}

template<typename BSAT>
__aicore__ inline int BlockSparseAttentionS1s2Bns1X910<BSAT>::TaskAlloc(const int32_t coreIdx,
    int &allocatedQtaskAllCore, const int sOuterBlockNum, const int coreNum, uint32_t bid, uint32_t nid)
{
    const int onceAlloc = coreNum; //* Align<int>(coreNum/2, 8);
    int taskId = -1;
    auto dstSort = this->tmpBuff.template Get<float>();
    auto sortTmpLocal = dstSort[64 * 2];
    AscendC::Sort<float, true>(dstSort, this->loads, this->cores.template ReinterpretCast<uint32_t>(), sortTmpLocal, 2);
    pipe_barrier(PIPE_V);
    AscendC::Extract<float>(this->loads, this->cores.template ReinterpretCast<uint32_t>(), dstSort, 2);
    pipe_barrier(PIPE_V);

    auto n = onceAlloc < sOuterBlockNum - allocatedQtaskAllCore ? onceAlloc : sOuterBlockNum - allocatedQtaskAllCore;
    constexpr auto FP16BYTENUM = BYTE_BLOCK_BSA / sizeof(half);
    auto calcStart = allocatedQtaskAllCore / FP16BYTENUM * FP16BYTENUM; //! allocatedQtaskAllCore is 8 aligned
    Cast(dstSort, this->taskCost[calcStart], AscendC::RoundMode::CAST_NONE, allocatedQtaskAllCore + n - calcStart);

    pipe_barrier(PIPE_V);
    Sub(this->loads, this->loads, dstSort[allocatedQtaskAllCore - calcStart], n);
    pipe_barrier(PIPE_V);

    //* good impl
    auto cmpRes = this->cores[64].template ReinterpretCast<uint64_t>();
    // * 64*sizeof(int) == 256
    AscendC::CompareScalar(this->cores[64].template ReinterpretCast<uint8_t>(), this->cores, coreIdx, CMPMODE::EQ, 64);
    auto eid_vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eid_vs);
    WaitFlag<HardEvent::V_S>(eid_vs);
    auto pos = ScalarGetSFFValue<1>(cmpRes.GetValue(0));
    if ((pos >= 0) && (pos < n)) {taskId = this->taskQid.GetValue(allocatedQtaskAllCore + pos);}

    allocatedQtaskAllCore += n;
    return taskId;
}

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::GetTaskCost(uint64_t sparseBlockCountOffsetBase,
    uint32_t sOuterBlockNum)
{
    auto cntUb = this->tmpBuff.template Get<int32_t>();
    auto colCount = Align<uint32_t>(sOuterBlockNum, BYTE_BLOCK_BSA / sizeof(int32_t)); // for corenum
    auto colCount64 = Align<uint32_t>(sOuterBlockNum, 64);
    DataCopyExtParams ext {1, static_cast<uint32_t>(sOuterBlockNum * sizeof(int32_t)), 0, 0, 0};  // * byte num
    DataCopyPadExtParams pad {true, 0, static_cast<uint8_t>(colCount - sOuterBlockNum), 0}; //* element num
    DataCopyPad(cntUb, this->sparseBlockCountGm[sparseBlockCountOffsetBase], ext, pad); //* pad to 64 align
    auto costUb = cntUb.template ReinterpretCast<half>();
    auto tmp = cntUb.template ReinterpretCast<int16_t>();
    auto index = cntUb[colCount64];
    auto sortedLocal = index[colCount64].template ReinterpretCast<half>();
    auto sortTmpLocal = sortedLocal[8 * colCount64 / sizeof(half)];
    ArithProgression<int32_t>(index, static_cast<int32_t>(0), static_cast<int32_t>(1), colCount64);
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
    
    if (colCount < colCount64) {
        Duplicate(cntUb[colCount], 0, colCount64 - colCount);
        pipe_barrier(PIPE_V);
        colCount = colCount64;
    }
    Cast(tmp, cntUb, AscendC::RoundMode::CAST_NONE, colCount); // int32->int16
    pipe_barrier(PIPE_V);
    Cast(costUb, tmp,  AscendC::RoundMode::CAST_NONE, colCount); // int16 -> half
    pipe_barrier(PIPE_V);
    AscendC::Sort<half, true>(sortedLocal, costUb, index.template ReinterpretCast<uint32_t>(),
        sortTmpLocal, colCount / 32);
    pipe_barrier(PIPE_V);
    AscendC::Extract<half>(this->taskCost, index.template ReinterpretCast<uint32_t>(), sortedLocal, colCount / 32);
    pipe_barrier(PIPE_V);
    Cast(this->taskQid, index, AscendC::RoundMode::CAST_NONE, colCount); // int32->int16
    pipe_barrier(PIPE_V);

    eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
}

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::ComputeEachCore(int32_t coreIdx)
{
    int32_t blockNum = static_cast<int32_t>(GetBlockNum() * GetTaskRation());
    BSAComputeParam *&params = this->tailParams;

    ArithProgression<int32_t>(this->cores, static_cast<int32_t>(0), static_cast<int32_t>(1), 64);
    Duplicate(this->loads, 0.0f, blockNum);
    Duplicate(this->loads[blockNum], static_cast<float>(-1e20), 64-blockNum);
    InitEachCoreWorkspace(coreIdx, blockNum);
    pipe_barrier(PIPE_V);
    
    for (uint32_t loopNIdx = 0; loopNIdx < this->headNumSize; loopNIdx++) {
        params->batchNOffset = loopNIdx;
        for (int bIdx = 0; bIdx < this->batchSize; bIdx++) {
            auto sparseBlockCountOffsetBase = CalcSparseMaskCountOffset(bIdx);
            auto sparseMaskOffsetBase = CalcSparseMaskOffset(bIdx);
            this->GetSingleCoreParam(bIdx); // actualSeqLengthKVPerBatch and other tiling data
            uint32_t sOuterSize = this->singleProcessSOuterSizeWhole;
            int sOuterBlockNum = (params->actualSeqLengthPerBatch + sOuterSize - 1) / sOuterSize;
            GetTaskCost(sparseBlockCountOffsetBase, sOuterBlockNum);
            auto realS2 = (params->actualSeqLengthKVPerBatch + this->sparseBlockSize - 1) / this->sparseBlockSize;
            params->multiSeqOffset = this->CalMultiSeqOffset(bIdx);
            int allocatedQtaskAllCore = 0;
            while (allocatedQtaskAllCore < sOuterBlockNum) {
                int sOuterLoopIdx = TaskAlloc(coreIdx, allocatedQtaskAllCore, sOuterBlockNum,
                    static_cast<int>(blockNum), bIdx, loopNIdx);
                auto eid_vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eid_vs); WaitFlag<HardEvent::S_V>(eid_vs);
                if (sOuterLoopIdx < 0) {continue;}

                params->singleProcessSOuterSize = sOuterLoopIdx == sOuterBlockNum - 1 ?
                    this->singleProcessSOuterSizeTail : sOuterSize;
                params->sOuterOffset = sOuterLoopIdx * sOuterSize;
                this->LoopSOuterOffsetInit(params->multiSeqOffset, bIdx);
                auto qBlockOffset = params->sOuterOffset / this->sparseBlockSize; // *  = sOuterLoopIdx
                auto sparseMaskIdRowOffset = sparseMaskOffsetBase + qBlockOffset * this->sparseMaskS2Size;

                auto sparseBlockCount = this->sparseBlockCountGm.GetValue(qBlockOffset + sparseBlockCountOffsetBase);
                if (sparseBlockCount == 0) {
                    InitOutput(this->attentionOutGm[params->attentionOutOffset],
                        params->singleProcessSOuterSize * this->headSize);
                    continue;
                }
                CalcSparsePos(this->tmpBuff, sparseMaskOffsetBase + sOuterLoopIdx * this->sparseMaskS2Size, 1,
                    this->sparseMaskS2Size, realS2);
                eid_vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eid_vs);
                WaitFlag<HardEvent::V_S>(eid_vs);
                auto lastSparseBlockId = this->blockIdUb.GetValue(sparseBlockCount - 1);
                SInnerLoopFunc(bIdx, sparseBlockCount, sparseMaskIdRowOffset, lastSparseBlockId);
            }
        }
    }
}

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::CalcSparsePos(TBuf<>& tbuf, uint32_t srcOffset,
    uint32_t s1Count, uint32_t s2, uint32_t realS2)
{
    auto mask = tbuf.Get<int8_t>();
    auto computeSize = s1Count * Align<uint32_t>(s2, BYTE_BLOCK_BSA);
    auto tailSize = s2 % BYTE_BLOCK_BSA;
    // DataCopy(mask, this->sparseMaskGm[srcOffset], computeSize); // 1
    DataCopyExtParams ext {1, static_cast<uint32_t>(s1Count * s2), 0, 0, 0};  // * byte num
    DataCopyPadExtParams pad {tailSize > 0, 0, static_cast<uint8_t>(tailSize == 0 ? 0: BYTE_BLOCK_BSA - tailSize),
        (int8_t)0}; //* element num
    DataCopyPad(mask, this->sparseMaskGm[srcOffset], ext, pad); //* pad to 32 align
    auto index = mask[computeSize].template ReinterpretCast<int32_t>();
    ArithProgression<int32_t>(index, static_cast<int32_t>(0), static_cast<int32_t>(1), s2); // 5
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
    pipe_barrier(PIPE_V);

    auto sparseId = index[computeSize]; // 9
    auto tmpUb = sparseId[computeSize].template ReinterpretCast<uint8_t>();

    Mask2Pos(sparseId, index, mask, realS2, tmpUb);
    DataCopy(this->blockIdUb, sparseId, computeSize);
    pipe_barrier(PIPE_V);
    eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
}


template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910<BSAT>::InitEachCoreWorkspace(uint32_t coreIdx,
    int32_t blockNum)
{
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;
    constexpr int reuseWorkspaceRatio = 2;
    int64_t msdExpandsize = 1;

    int64_t mm1ResSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * \
        this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    mm1ResSize = mm1ResSize * msdExpandsize;
    int64_t mm2ResSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * \
        this->tilingData->promptAttentionBaseParams.headSize  * msdExpandsize;
    
    this->bmm1ResGmDb[0].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize  * reuseWorkspaceRatio].GetPhyAddr());
    this->bmm1ResGmDb[1].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize ].GetPhyAddr());

    // same addr with bmm1ResGmDb
    this->softmaxResGmDb[0].SetGlobalBuffer((__gm__ MM_IN_T*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize  * reuseWorkspaceRatio].GetPhyAddr());
    this->softmaxResGmDb[1].SetGlobalBuffer((__gm__ MM_IN_T*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize ].GetPhyAddr());

    int64_t buff_offset = blockNum * (this->spmTmpSize + mm1ResSize * reuseWorkspaceRatio);
    this->bmm2ResGmDb[0].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[buff_offset +
        coreIdx * mm2ResSize * reuseWorkspaceRatio].GetPhyAddr());
    this->bmm2ResGmDb[1].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[buff_offset +
        coreIdx * mm2ResSize * reuseWorkspaceRatio + mm2ResSize].GetPhyAddr());

    buff_offset += blockNum * mm2ResSize * reuseWorkspaceRatio;

    // After placing the four structures of the first kernel, place the four structures of the second kernel
    // When IFA, After placing the first structure of all kernel, place the second one.
    if constexpr (USE_BLOCK_SPARE) {  // If compute type is different，the offset size here is different.
        GlobalTensor<uint32_t> workspaceGmPA;  //  storage PA callback structure data
        workspaceGmPA.SetGlobalBuffer((__gm__ uint32_t*)this->workspaceGm[buff_offset].GetPhyAddr());
        int32_t paStructSize = 64 / sizeof(uint32_t);  //  dcci cacheline 64B alignment  16 * 4B = 64B
        int32_t NumOfBmm = 2;
        int64_t baseCBDataOffset = coreIdx * paStructSize * NumOfBmm * reuseWorkspaceRatio;
        this->bmm1CBDataGm[0].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset].GetPhyAddr());
        this->bmm1CBDataPtr[0] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset].GetPhyAddr();
        this->bmm1CBDataGm[1].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset +
            paStructSize].GetPhyAddr());
        this->bmm1CBDataPtr[1] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize].GetPhyAddr();

        this->bmm2CBDataGm[0].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize *
            reuseWorkspaceRatio].GetPhyAddr());
        this->bmm2CBDataPtr[0] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize *
            reuseWorkspaceRatio].GetPhyAddr();
        this->bmm2CBDataGm[1].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize *
            reuseWorkspaceRatio + paStructSize].GetPhyAddr());
        this->bmm2CBDataPtr[1] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize *
            reuseWorkspaceRatio + paStructSize].GetPhyAddr();
        buff_offset += blockNum * baseCBDataOffset + paStructSize * reuseWorkspaceRatio + paStructSize *
            reuseWorkspaceRatio;
    }
}

#endif  // PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_H

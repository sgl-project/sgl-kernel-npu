/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_sort_one_core.h
 * \brief
 */
#ifndef MOE_V3_SORT_ONE_CORE_H_REGBASE
#define MOE_V3_SORT_ONE_CORE_H_REGBASE

#include "moe_v3_sort_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class MoeSortOneCore : public MoeSortBase
{
public:
    __aicore__ inline MoeSortOneCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
};

__aicore__ inline void MoeSortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
    LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeSortOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdx = inLocal[0];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->tileLength);

    uint16_t repeatTimes = Ceil(this->tileLength, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->tileLength);
    __local_mem__ float *inUbAddr = (__local_mem__ float *)expertIdxFp32.GetPhyAddr();
    float cmpScalar = static_cast<float>(expertStart_);

    VfTagOutOfRangeExpert(inUbAddr, repeatTimes, sreg, cmpScalar);

    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, expertIdxFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<uint32_t> expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(expandedRowIdxGm, outLocal[this->sortNum], intriParams);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                            const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);
    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength);
    } else {
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->tileLength, sizeof(int32_t)),
                                         Align(this->tileLength, sizeof(int32_t)));
    }

    // key and value
    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void MoeSortOneCore::Process()
{
    if (GetBlockIdx() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    this->SyncAll();
}
}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_SORT_ONE_CORE_H_REGBASE

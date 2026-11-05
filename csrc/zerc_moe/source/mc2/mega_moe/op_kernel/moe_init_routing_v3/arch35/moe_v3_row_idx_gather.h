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
 * \file moe_v3_row_idx_gather.h
 * \brief
 */
#ifndef MOE_V3_ROW_IDX_GATHER_H_REGBASE
#define MOE_V3_ROW_IDX_GATHER_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class RowIdxGather
{
public:
    __aicore__ inline RowIdxGather(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> sortedExpertIndicesGm_;

    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIndicesInQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutQueue_;

    const MoeV3Arch35ExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t actualExpertNum_ = 0;

    int64_t loops_ = 0;
    int64_t perLoopElements_ = 0;
    int64_t lastLoopElements_ = 0;

    int64_t perCoreElements;
    int64_t lastCoreElements;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeSimt(int64_t elements, int64_t indexBase,
                                                                             __gm__ int32_t *sortedExpertIndicesGmAddr,
                                                                             __gm__ int32_t *expandedRowIdxGmAddr)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(elements);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int64_t outIndices = expandedRowIdxGmAddr[indexBase + index];
        sortedExpertIndicesGmAddr[outIndices] = indexBase + index;
    }
}

__aicore__ inline void RowIdxGather::Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                          const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    actualExpertNum_ = tilingData->actualExpertNum;

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    if (blockIdx_ < needCoreNum_ - 1) {
        curCoreElements_ = perCoreElements_;
    } else if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
    }

    int64_t expertTotalCount_ = tilingData->n * tilingData->k;

    perCoreElements = Ceil(expertTotalCount_, needCoreNum_);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreElements);
    lastCoreElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreElements;
    int64_t perCoreLoops = Ceil(perCoreElements, expertTokensCountTilingData_->perCorePerLoopElements);
    int64_t perCorePerLoopElements = Ceil(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreLoops = Ceil(lastCoreElements, expertTokensCountTilingData_->perCorePerLoopElements);
    int64_t lastCorePerLoopElements = Ceil(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopELements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    loops_ = perCoreLoops;
    if (blockIdx_ == needCoreNum_ - 1) {
        loops_ = lastCoreLoops;
        perLoopElements_ = lastCorePerLoopElements;
        lastLoopElements_ = lastCoreLastLoopELements;
    } else {
        loops_ = perCoreLoops;
        perLoopElements_ = perCorePerLoopElements;
        lastLoopElements_ = perCoreLastLoopElements;
    }

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, actualExpertNum_);
    sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)workspace, actualExpertNum_);

    pipe_->InitBuffer(sortedExpertIndicesInQueue_, 1, AlignBytes(perLoopElements_, sizeof(int32_t)));
    pipe_->InitBuffer(copyOutQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}

__aicore__ inline void RowIdxGather::Process()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t elements = (blockIdx_ == needCoreNum_ - 1 ? lastCoreElements : perCoreElements);
        __gm__ int32_t *sortedExpertIndicesGmAddr = (__gm__ int32_t *)sortedExpertIndicesGm_.GetPhyAddr();
        __gm__ int32_t *expandedRowIdxGmAddr = (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr();
        Simt::VF_CALL<ComputeSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, elements, blockIdx_ * perCoreElements,
                                   sortedExpertIndicesGmAddr, expandedRowIdxGmAddr);
    }
    SyncAll<true>();
    // printf("blockIdx: %d, sortedExpertIndicesGm_: %d, expandedRowIdxGm_: %d, needCoreNum_: %d, curCoreElements_:
    // %d\n", blockIdx_, sortedExpertIndicesGm_(blockIdx_), expandedRowIdxGm_(blockIdx_), needCoreNum_,
    // curCoreElements_);
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_ROW_IDX_GATHER_H_REGBASE

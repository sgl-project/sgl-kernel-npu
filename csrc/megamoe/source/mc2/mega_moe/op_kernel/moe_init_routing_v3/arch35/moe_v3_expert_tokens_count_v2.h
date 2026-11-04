/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_expert_tokens_count_v2.h
 * \brief ExpertTokensCount V2: 直接对 expertIdx 执行原子加，无需排序
 */

#ifndef MOE_V3_EXPERT_TOKENS_COUNT_V2_H_REGBASE
#define MOE_V3_EXPERT_TOKENS_COUNT_V2_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class ExpertTokensCountV2
{
public:
    __aicore__ inline ExpertTokensCountV2(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCount,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    // expertIdx批量搬入UB的块大小(元素数): 8192个int32 = 32KB, 循环作超长兜底。
    static constexpr int64_t EXPERT_IDX_CHUNK_ELEMS = 8192;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expertTokensCountGm_;
    // expertIdx输入队列(批量GM->UB) + 计数缓冲, 均走TPipe管理。
    TQue<QuePosition::VECIN, 1> expertIdxInQueue_;
    TBuf<QuePosition::VECCALC> tmpSumBuf_;
    TPipe *pipe_;

    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t expertTokensNumType_ = 0;
};

__aicore__ inline void ExpertTokensCountV2::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCount,
                                                 const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    // 使用与 V1 相同的 Tiling 参数
    auto *tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    needCoreNum_ = tokensCountTiling->needCoreNum;
    perCoreElements_ = tokensCountTiling->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertTokensNumType_ = tilingData->expertTokensNumType;

    // 计算当前 block 处理的元素数
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = tokensCountTiling->lastCoreElements;
    } else {
        curCoreElements_ = perCoreElements_;
    }

    // 设置 GM 地址
    // expertIdx: 展平后的数组，总长度 = n * k = M * topK
    // 这里假设 expertIdx 已经展平为一维数组
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + blockIdx_ * perCoreElements_);

    // expertTokensCount: 每个 expert 的 token 数量
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int32_t *)expertTokensCount);

    // 输入队列 + 计数缓冲经TPipe分配(与V1同模式)。
    pipe_->InitBuffer(expertIdxInQueue_, 1, AlignBytes(EXPERT_IDX_CHUNK_ELEMS, sizeof(int32_t)));
    pipe_->InitBuffer(tmpSumBuf_, AlignBytes(actualExpertNum_, sizeof(int32_t)));
}

__aicore__ inline void ExpertTokensCountV2::Process()
{
    // 仅需要的 core 执行
    if (blockIdx_ >= needCoreNum_) {
        return;
    }
    LocalTensor<int32_t> tmpSum = tmpSumBuf_.Get<int32_t>();
    AscendC::Duplicate(tmpSum, 0, actualExpertNum_);
    SetWaitFlag<HardEvent::V_S>();

    // expertIdx按块批量GM->UB, 再从UB累加计数(消除逐元素标量GM读)。
    // 计数保留标量: 直方图写是随机位置RMW, 向量化需跨lane求和, 这里不盲用。
    int64_t remainElements = curCoreElements_;
    int64_t gmOffset = 0;
    while (remainElements > 0) {
        int64_t curLoopElements = remainElements > EXPERT_IDX_CHUNK_ELEMS ? EXPERT_IDX_CHUNK_ELEMS : remainElements;
        LocalTensor<int32_t> expertIdxLocal = expertIdxInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopElements * sizeof(int32_t)),
                                     0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(expertIdxLocal, expertIdxGm_[gmOffset], copyParams, padParams);
        expertIdxInQueue_.EnQue(expertIdxLocal);
        LocalTensor<int32_t> expertIdxInLocal = expertIdxInQueue_.DeQue<int32_t>();
        // DeQue只保证MTE2->V; 标量GetValue走S管道, 必须显式等MTE2->S,
        // 否则抢跑DMA读到垃圾expertId, tmpSum(expertId)越界写UB(507015)。
        SetWaitFlag<HardEvent::MTE2_S>();
        for (int64_t i = 0; i < curLoopElements; i++) {
            int32_t expertId = expertIdxInLocal.GetValue(i);
            tmpSum(expertId) = tmpSum(expertId) + 1;
        }
        expertIdxInQueue_.FreeTensor(expertIdxInLocal);
        gmOffset += curLoopElements;
        remainElements -= curLoopElements;
    }
    if (expertTokensNumType_ == CUMSUM_MODE) {
        for (int i = 1; i < actualExpertNum_; ++i) {
            tmpSum(i) = tmpSum(i) + tmpSum(i - 1);
        }
    }
    // tmpSum为标量写, 搬出走MTE3, 需S->MTE3同步(V1同款S_MTE3)。
    SetWaitFlag<HardEvent::S_MTE3>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertTokensCountGm_, tmpSum, copyParams);
    // DataCopy(expertTokensCountGm_, tmpSum, actualExpertNum_);
    SetAtomicNone();
}

}  // namespace MoeInitRoutingV3

#endif  // MOE_V3_EXPERT_TOKENS_COUNT_V2_H_REGBASE

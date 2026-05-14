/**
?* Copyright (c) 2025 Huawei Technologies Co., Ltd.
?* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
?* CANN Open Software License Agreement Version 2.0 (the "License").
?* Please refer to the License for details. You may not use this file except in compliance with the License.
?* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
?* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
?* See LICENSE in the root of the software repository for the full text of the License.
?*/

/*!
 * \file moe_distribute_v2_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_V2_BASE_H
#define MOE_DISTRIBUTE_V2_BASE_H

#include "moe_distribute_base.h"

namespace MoeDistributeV2Base {
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t ZERONE_STATE_POS = 0U;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr uint32_t HCCL_DFX_POS = 8U;
constexpr uint32_t HCCL_DFX_NUM = 2U;
constexpr uint32_t HCCL_EPRANKId_POS = 0U;
constexpr uint32_t HCCL_WORLDSIZE_POS = 1U;
constexpr uint32_t UB_ALIGN = 32U;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

using namespace AscendC;

__aicore__ inline uint32_t InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor, __gm__ HcclOpParam *winContext,
                                        uint32_t epRankIdOriginal, uint32_t moeExpertNum, uint32_t epWorldSizeOriginal,
                                        uint32_t globalBS, TBuf<> dataStateBuf)
{
    LocalTensor<uint64_t> dataStateLocalTensor64 = dataStateBuf.Get<uint64_t>();
    LocalTensor<uint32_t> dataStateLocalTensor = dataStateBuf.Get<uint32_t>();
    DataCopy(dataStateLocalTensor, selfDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t epRankIdHccl = GetRankId(winContext);
    uint32_t epWorldSizeHccl = GetRankDim(winContext);
    uint32_t dataState = dataStateLocalTensor.GetValue(ZERONE_STATE_POS);
    dataStateLocalTensor.SetValue(ZERONE_STATE_POS, dataState == 0 ? 1 : 0);
    dataStateLocalTensor.SetValue(OPOSITION_POS, 1);
    dataStateLocalTensor.SetValue(TILING_EPRANKID_POS, epRankIdOriginal);
    dataStateLocalTensor.SetValue(MOE_NUM_POS, moeExpertNum);
    dataStateLocalTensor.SetValue(TILING_WORLDSIZE_POS, epWorldSizeOriginal);
    dataStateLocalTensor.SetValue(GLOBALBS_POS, globalBS);
    uint32_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
    opCnt = opCnt + 1;
    dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusGMTensor, dataStateLocalTensor, UB_ALIGN / sizeof(uint32_t));

    if ((epRankIdOriginal != epRankIdHccl) || (epWorldSizeOriginal != epWorldSizeHccl)) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        DataCopyParams hcclDatacopyParams{1U, HCCL_DFX_NUM * sizeof(uint32_t), 0U, 0U};
        dataStateLocalTensor.SetValue(HCCL_EPRANKId_POS, epRankIdHccl);
        dataStateLocalTensor.SetValue(HCCL_WORLDSIZE_POS, epWorldSizeHccl);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor[HCCL_DFX_POS], dataStateLocalTensor, hcclDatacopyParams);
    }
    return dataState;
}

/*cycle prof*/
#define USE_CYCLE_PROF

#ifdef USE_CYCLE_PROF
#pragma message("Use cycle prof")
#define CYCLE_PROF_HEADER_LEN (50 * 128)    //
#define CYCLE_PROF_ONE_FRAME_COUNT (16)     // 每次最多支持256次打点
#define CYCLE_PROF_MAX_FRAME (1024)         // 最多纪录1024次
#define CYCLE_PROF_HEADRE_COUNTER_OFFSET 4  // 头4个8字节留给其他作用
#define DATA_FULSH(_gm_tensor, _type)                                                                  \
    AscendC::Barrier();                                                                                \
    DataCacheCleanAndInvalid<_type, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(_gm_tensor); \
    __asm__("NOP");                                                                                    \
    dsb(DSB_ALL);

#define CYCLE_PROF_CLASS_DEFINE()                    \
    bool enableProf_{false};                         \
    GlobalTensor<uint64_t> profHeader_;              \
    GlobalTensor<int64_t> profDataTensor_;           \
    GM_ADDR profData_{nullptr};                      \
    int64_t profTime_[CYCLE_PROF_ONE_FRAME_COUNT]{}; \
    uint64_t profCounter_{0};

#define CYCLE_PROF_INIT(__header)                                                                   \
    if (__header != nullptr) {                                                                      \
        enableProf_ = true;                                                                         \
        __gm__ uint8_t *header = ((__gm__ uint8_t *)__header);                                      \
        profHeader_.SetGlobalBuffer((__gm__ uint64_t *)(header + (2 + GetBlockIdx()) * 128));       \
        DATA_FULSH(profHeader_, uint64_t);                                                          \
        profCounter_ = profHeader_.GetValue(0);                                                     \
        DATA_FULSH(profHeader_, uint64_t);                                                          \
        profData_ = header + CYCLE_PROF_HEADER_LEN +                                                \
                    (profCounter_ % CYCLE_PROF_MAX_FRAME) * (CYCLE_PROF_ONE_FRAME_COUNT * 8 * 48) + \
                    GetBlockIdx() * (CYCLE_PROF_ONE_FRAME_COUNT * 8);                               \
        profDataTensor_.SetGlobalBuffer((__gm__ int64_t *)profData_);                               \
    }

#define CYCLE_PROF_RECORD(_id)                             \
    if (enableProf_ && _id < CYCLE_PROF_ONE_FRAME_COUNT) { \
        pipe_barrier(PIPE_ALL);                            \
        auto cycle = GetSystemCycle();                     \
        profTime_[_id] = cycle;                            \
    }

#define CYCL_PROF_INC(_c)              \
    if (enableProf_) {                 \
        profTime_[15] += (uint64_t)_c; \
    }

#define CYCLE_PROF_FINI()                                                                                     \
    if (enableProf_) {                                                                                        \
        tpipe_->Reset();                                                                                      \
        TBuf<> profBuf_;                                                                                      \
        tpipe_->InitBuffer(profBuf_, CYCLE_PROF_ONE_FRAME_COUNT * 16);                                        \
        /* copy prof */                                                                                       \
        auto dataLocalTensor = profBuf_.GetWithOffset<int64_t>(CYCLE_PROF_ONE_FRAME_COUNT * 8, 0);            \
        for (int i = 1; i < 8; ++i) {                                                                         \
            int64_t cycle = profTime_[i];                                                                     \
            int64_t preCycle = profTime_[i - 1];                                                              \
            dataLocalTensor.SetValue(i - 1, (cycle - preCycle) / 50);                                         \
        }                                                                                                     \
        pipe_barrier(PIPE_ALL);                                                                               \
        DataCopy(profDataTensor_, dataLocalTensor, CYCLE_PROF_ONE_FRAME_COUNT);                               \
        pipe_barrier(PIPE_ALL);                                                                               \
        SyncAll<true>();                                                                                      \
        /* copy counter */                                                                                    \
        auto counterLocalTensor =                                                                             \
            profBuf_.GetWithOffset<uint64_t>(CYCLE_PROF_ONE_FRAME_COUNT * 8, CYCLE_PROF_ONE_FRAME_COUNT * 8); \
        counterLocalTensor.SetValue(0, profCounter_ + 1);                                                     \
        pipe_barrier(PIPE_ALL);                                                                               \
        DataCopy(profHeader_, counterLocalTensor, 16);                                                        \
        pipe_barrier(PIPE_ALL);                                                                               \
    }
#else
#pragma message("original version")
#define CYCLE_PROF_CLASS_DEFINE()
#define CYCLE_PROF_INIT(__head)
#define CYCLE_PROF_RECORD(_facker_id)
#define CYCLE_PROF_FINI()
#define CYCL_PROF_INC(_c)
#endif

}  // namespace MoeDistributeV2Base
#endif  // MOE_DISTRIBUTE_V2_BASE_H

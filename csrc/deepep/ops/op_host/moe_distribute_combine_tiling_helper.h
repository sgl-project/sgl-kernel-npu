/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling_helper.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H
#define MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "mc2_tiling_utils.h"

namespace optiling {
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t EXPAND_IDX_INDEX = 2;
constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
constexpr uint32_t EXPERT_SCALES_INDEX = 4;
constexpr uint32_t TP_SEND_COUNTS_INDEX = 5;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 6;
constexpr uint32_t SHARED_EXPERT_X_INDEX = 11;
constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;

struct CombineV2Config {
    uint32_t contextIndex = 0;               // 0: 根据combineV3算子原型标志位初始化x索引
    uint32_t expandXIndex = 0;               // 0: 根据combineV2算子原型标志位初始化expertX索引
    uint32_t expertIdsIndex = 1;             // 1: 根据combineV2算子原型标志位初始化expertIds索引
    uint32_t assistInfoIndex = 2;            // 2: 根据combineV2算子原型标志位初始化assistInfo索引
    uint32_t epSendCountIndex = 3;           // 3: 根据combineV2算子原型标志位初始化epSendCount索引
    uint32_t expertScalesIndex = 4;          // 4: 根据combineV2算子原型标志位初始化expertScales索引
    uint32_t residualXIndex = 5;             // 根据combineARN算子原型标志位初始化residualX索引
    uint32_t gammaIndex = 6;                 // 根据combineARN算子原型标志位初始化gamma索引
    uint32_t tpSendCountsIndex = 5;          // 根据combineV2算子原型标志位初始化tpSendCounts索引
    uint32_t xActiveMaskIndex = 6;           // 根据combineV2算子原型标志位初始化xActiveMask索引
    uint32_t activationScaleIndex = 7;       // 根据combineV2算子原型标志位初始化activationScale索引
    uint32_t weightScaleIndex = 8;           // 根据combineV2算子原型标志位初始化weightScale索引
    uint32_t groupListIndex = 9;             // 根据combineV2算子原型标志位初始化groupList索引
    uint32_t sharedExpertXIndex = 10;        // 根据combineV2算子原型标志位初始化sharedExpertX索引
    uint32_t elasticInfoIndex = 11;          // 根据combineV2算子原型标志位初始化elasticInfo索引
    uint32_t oriXIndex = 13;                 // 根据combineV2算子原型标志位初始化oriX索引
    uint32_t constExpertAlpha1Index = 14;    // 根据combineV2算子原型标志位初始化constExpertAlpha1索引
    uint32_t constExpertAlpha2Index = 15;    // 根据combineV2算子原型标志位初始化constExpertAlpha2索引
    uint32_t constExpertVIndex = 16;         // 根据combineV2算子原型标志位初始化constExpertV索引
    uint32_t performanceInfoIndex = 17;      // 根据combineV2算子原型标志位初始化 performanceInfo索引
    uint32_t outputYIndex = 0;               // 根据combineARN算子原型标志位初始化outputY索引
    uint32_t outputRstdIndex = 1;            // 根据combineARN算子原型标志位初始化outputRstd索引
    uint32_t outputXIndex = 0;               // 根据combineV2算子原型标志位初始化outputX索引
    uint32_t attrGroupEpIndex = 0;           // 0: 根据combineV2算子原型标志位初始化groupEp索引
    uint32_t attrEpWorldSizeIndex = 1;       // 1: 根据combineV2算子原型标志位初始化epWorldSize索引
    uint32_t attrEpRankIdIndex = 2;          // 2: 根据combineV2算子原型标志位初始化epRankId索引
    uint32_t attrMoeExpertNumIndex = 3;      // 3: 根据combineV2算子原型标志位初始化moeExpertNum索引
    uint32_t attrCclBufferSizeIndex = 3;     // 3: 根据combineV3算子原型标志位初始化cclBufferSize索引
    uint32_t attrGroupTpIndex = 4;           // 4: 根据combineV2算子原型标志位初始化groupTp索引
    uint32_t attrTpWorldSizeIndex = 5;       // 5: 根据combineV2算子原型标志位初始化tpWorldSize索引
    uint32_t attrTpRankIdIndex = 6;          // 6: 根据combineV2算子原型标志位初始化tpRankId索引
    uint32_t attrExpertSharedTypeIndex = 7;  // 7: 根据combineV2算子原型标志位初始化expertSharedType索引
    uint32_t attrSharedExpertNumIndex = 8;   // 8: 根据combineV2算子原型标志位初始化sharedExpertNum索引
    uint32_t attrSharedExpertRankNumIndex = 9;  // 9: 根据combineV2算子原型标志位初始化sharedExpertRankNum索引
    uint32_t attrGlobalBsIndex = 10;            // 10: 根据combineV2算子原型标志位初始化globalBs索引
    uint32_t attrOutDTypeIndex = 11;            // 11: 根据combineV2算子原型标志位初始化outDType索引
    uint32_t attrCommQuantModeIndex = 12;       // 12: 根据combineV2算子原型标志位初始化commQuantMode索引
    uint32_t attrGroupListTypeIndex = 13;       // 13: 根据combineV2算子原型标志位初始化groupListType索引
    uint32_t attrCommAlgIndex = 14;             // 14: 根据combineV2算子原型标志位初始化commAlg索引
    uint32_t attrNormEpsIndex = 15;             // 根据combineARN算子原型标志位初始化attrNormEps索引
    uint32_t attrZeroExpertNumIndex = 15;       // 根据combineV2算子原型标志位初始化attrZeroExpertNum索引
    uint32_t attrCopyExpertNumIndex = 16;       // 根据combineV2算子原型标志位初始化attrCopyExpertNum索引
    uint32_t attrConstExpertNumIndex = 17;      // 根据combineV2算子原型标志位初始化attrConstExpertNum索引
    bool hasAddRmsNorm = false;
    bool isMc2Context = false;
};
}  // namespace optiling
#endif  // MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H

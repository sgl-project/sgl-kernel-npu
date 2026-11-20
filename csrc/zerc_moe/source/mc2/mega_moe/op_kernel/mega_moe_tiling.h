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
 * \file mega_moe_tiling.h
 * \brief
 */

#ifndef ASCENDC_MEGA_MOE_TILING
#define ASCENDC_MEGA_MOE_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_arch35_tiling_def.h"

using namespace AscendC;

constexpr uint32_t EXPERT_SCHEDULE_M_TILE_ROWS = 256U;
constexpr uint32_t EXPERT_SCHEDULE_MIN_TILE_SPREAD = 2U;
constexpr uint32_t EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS = 3U;

struct MegaMoeTilingData {
    MoeInitRoutingV3Arch35TilingData moeInitRoutingTilingData;
    uint32_t expertPerRank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;  // if blockNumPerEP < dispatchRows, then dispatchRows is invalid
    uint32_t maxOutputSize;
    uint32_t rankId;
    uint32_t topK;
    int32_t dispatchRows;
    uint8_t groupListType;
    bool transX;
    bool transW;
    bool transW2;
    // dispatch_quant_mode == 0: BF16 activations and BF16 weights end to end,
    // without MX scales. Workspace and peermem layouts depend on it.
    bool noQuant;
    // Shared experts are computed on every rank from that rank's own tokens and
    // never take part in Dispatch/Combine. sharedExpertNum comes from
    // shared_weight1.dim0 and is 0 when the feature is unused; sharedN is the
    // shared GMM1 output width (2 * shared intermediate), which may differ from
    // the routed n. Both stay 0 without the feature so every existing layout is
    // byte-identical.
    uint32_t sharedExpertNum;
    uint32_t sharedN;
    float activationClamp;  // -1 disables; >= 0 is a BF16-rounded clamp threshold.
};
#endif

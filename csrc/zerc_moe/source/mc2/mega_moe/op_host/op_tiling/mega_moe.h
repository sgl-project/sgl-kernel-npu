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
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_TILING
#define MEGA_MOE_TILING

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

struct MegaMoeConfig {
    uint32_t contextIndex = 0U;
    uint32_t xIndex = 1U;
    uint32_t topkIdsIndex = 2U;
    uint32_t topkWeightsIndex = 3U;
    uint32_t weightOneIndex = 4U;
    uint32_t weightTwoIndex = 5U;
    uint32_t weightScalesOneIndex = 6U;
    uint32_t weightScalesTwoIndex = 7U;
    uint32_t xActiveMaskIndex = 8U;
    uint32_t scalesIndex = 9U;
    // Shared experts are appended after `scales` so the established input ABI
    // and Timer V2's reuse of the scales slot both stay unchanged.
    uint32_t yIndex = 14U;
    uint32_t expertTokenNumsIndex = 15U;
    uint32_t attrMoeExpertNumIndex = 0U;
    uint32_t attrEpWorldSizeIndex = 1U;
    uint32_t attrCclBufferSizeIndex = 2U;
    uint32_t attrMaxRecvTokenNumIndex = 3U;
    uint32_t attrDispatchQuantModeIndex = 4U;
    uint32_t attrDispatchQuantOutTypeIndex = 5U;
    uint32_t attrCombineQuantModeIndex = 6U;
    uint32_t attrCommAlgIndex = 7U;
    uint32_t attrGlobalBsIndex = 8U;
    uint32_t attrSharedExpertNumIndex = 9U;
    uint32_t attrActivationClampIndex = 10U;
    bool isMc2Context = false;
};
using namespace Ops::Transformer::OpTiling;

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config);

}  // namespace optiling

#endif

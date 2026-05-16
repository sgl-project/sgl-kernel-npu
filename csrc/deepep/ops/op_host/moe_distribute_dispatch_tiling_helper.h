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
 * \file moe_distribute_dispatch_tiling_helper.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_HELPER_H
#define MOE_DISTRIBUTE_DISPATCH_TILING_HELPER_H

#include <cstdint>
#include <map>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"

namespace optiling {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_EXPAND_IDX_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t ONE_DIM = 1;

constexpr uint32_t OP_VERSION_1 = 1U;

enum class RealModeA5 : uint32_t {
    NO_SCALES = 0,
    STATIC_SCALES = 1,
    PERTOKEN_SCALES = 2,
    PERGROUP_SCALES = 3,
    MX_SCALES = 4,
    HIF8_SCALES = 5,
    INVALID_MODE
};

// quantMode 2
constexpr uint32_t DYNAMIC_SCALE_ONE_DIM_NUM = 1;
// quantMode 3-4
constexpr uint32_t DYNAMIC_SCALE_TWO_DIM_NUM = 2;

enum class QuantModeA5 : uint32_t {
    NON_QUANT = 0,
    STATIC_QUANT = 1,
    PERTOKEN_DYNAMIC_QUANT = 2,
    PERGROUP_DYNAMIC_QUANT = 3,
    MX_QUANT = 4,
    BUTT
};

const std::map<std::pair<QuantModeA5, ge::DataType>, RealModeA5> QUANT_MODE_MAP = {
    {{QuantModeA5::NON_QUANT, ge::DT_BF16}, RealModeA5::NO_SCALES},
    {{QuantModeA5::NON_QUANT, ge::DT_FLOAT16}, RealModeA5::NO_SCALES},
    {{QuantModeA5::NON_QUANT, ge::DT_FLOAT8_E4M3FN}, RealModeA5::NO_SCALES},
    {{QuantModeA5::NON_QUANT, ge::DT_FLOAT8_E5M2}, RealModeA5::NO_SCALES},
    {{QuantModeA5::NON_QUANT, ge::DT_HIFLOAT8}, RealModeA5::NO_SCALES},
    {{QuantModeA5::NON_QUANT, ge::DT_FLOAT4_E2M1}, RealModeA5::NO_SCALES},
    {{QuantModeA5::NON_QUANT, ge::DT_FLOAT4_E1M2}, RealModeA5::NO_SCALES},
    {{QuantModeA5::STATIC_QUANT, ge::DT_INT8}, RealModeA5::STATIC_SCALES},
    {{QuantModeA5::STATIC_QUANT, ge::DT_HIFLOAT8}, RealModeA5::HIF8_SCALES},
    {{QuantModeA5::PERTOKEN_DYNAMIC_QUANT, ge::DT_INT8}, RealModeA5::PERTOKEN_SCALES},
    {{QuantModeA5::PERTOKEN_DYNAMIC_QUANT, ge::DT_FLOAT8_E5M2}, RealModeA5::PERTOKEN_SCALES},
    {{QuantModeA5::PERTOKEN_DYNAMIC_QUANT, ge::DT_FLOAT8_E4M3FN}, RealModeA5::PERTOKEN_SCALES},
    {{QuantModeA5::PERGROUP_DYNAMIC_QUANT, ge::DT_FLOAT8_E5M2}, RealModeA5::PERGROUP_SCALES},
    {{QuantModeA5::PERGROUP_DYNAMIC_QUANT, ge::DT_FLOAT8_E4M3FN}, RealModeA5::PERGROUP_SCALES},
    {{QuantModeA5::MX_QUANT, ge::DT_FLOAT8_E5M2}, RealModeA5::MX_SCALES},
    {{QuantModeA5::MX_QUANT, ge::DT_FLOAT8_E4M3FN}, RealModeA5::MX_SCALES},
    {{QuantModeA5::MX_QUANT, ge::DT_FLOAT4_E2M1}, RealModeA5::MX_SCALES},
    {{QuantModeA5::MX_QUANT, ge::DT_FLOAT4_E1M2}, RealModeA5::MX_SCALES}};

// Supported x datatype in nonquant mode, the same as expandX
const std::set<ge::DataType> NON_QUANT_DTYPE = {ge::DT_FLOAT16,       ge::DT_BF16,        ge::DT_HIFLOAT8,
                                                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1,
                                                ge::DT_FLOAT4_E1M2};

}  // namespace optiling
#endif  // MOE_DISTRIBUTE_DISPATCH_TILING_HELPER_H

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
 * \file moe_distribute_combine_v2_ccu_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_CCU_TILING_H
#define MOE_DISTRIBUTE_COMBINE_V2_CCU_TILING_H

#include "mc2_tiling_utils.h"

namespace optiling {

ge::graphStatus MoeDistributeCombineTilingImpl(gert::TilingContext *context);

}  // namespace optiling

#endif  // MOE_DISTRIBUTE_COMBINE_TILING_A5_H

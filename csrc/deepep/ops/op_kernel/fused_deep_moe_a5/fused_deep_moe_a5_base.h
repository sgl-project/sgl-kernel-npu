/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: Definition of communication group related structures
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 Create a definition file for a distribution group related structure
 */
#ifndef FUSED_DEEP_MOE_BASE_H
#define FUSED_DEEP_MOE_BASE_H

#include "op_kernel/moe_distribute_base.h"

#define TemplateMC2TypeClass                                                                      \
        typename ExpandXType, typename WeightType, bool WEIGHT_NZ, typename ExpandIdxType, \
        bool IsNeedReduceScatter, uint32_t EXEC_FLAG
#define TemplateMC2TypeFunc ExpandXType, WeightType, WEIGHT_NZ, ExpandIdxType, IsNeedReduceScatter, EXEC_FLAG

#define TemplateDispatchTypeClass                                                                                  \
            typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
            bool IsNeedAllgater, uint32_t EXEC_FLAG
#define TemplateDispatchTypeFunc                                                  \
            XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, \
            IsNeedAllgater, EXEC_FLAG

constexpr int64_t SLEEP_CYCLE = 50;

__aicore__ inline void SPIN_WAIT_CYCLES()
{
    AscendC::Nop<SLEEP_CYCLE>();
}

#endif  // FUSED_DEEP_MOE_BASE_H

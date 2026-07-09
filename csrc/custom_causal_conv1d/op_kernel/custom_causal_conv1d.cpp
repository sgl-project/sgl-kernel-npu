/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d.cpp
 * \brief causal_conv1d kernel entry with runtime dispatch
 */

#include "custom_causal_conv1d_fn.h"
#include "custom_causal_conv1d_update.h"

using namespace AscendC;
using namespace NsCausalConv1d;

namespace {

template <typename T>
__aicore__ inline void DispatchFn(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                  GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y,
                                  GM_ADDR workspace, const __gm__ CausalConv1dTilingData *tilingData, uint32_t widthKey,
                                  uint32_t fnPlanKey)
{
    if (fnPlanKey == CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS) {
        if (widthKey == CAUSAL_CONV1D_TPL_WIDTH_2) {
            RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_2, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else if (widthKey == CAUSAL_CONV1D_TPL_WIDTH_3) {
            RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_3, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else {
            RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        }
    } else {
        if (widthKey == CAUSAL_CONV1D_TPL_WIDTH_2) {
            RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_2, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else if (widthKey == CAUSAL_CONV1D_TPL_WIDTH_3) {
            RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_3, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else {
            RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void custom_causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
                                                           GM_ADDR bias, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
                                                           GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens,
                                                           GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

    auto tilingData = reinterpret_cast<__gm__ CausalConv1dTilingData *>(tiling);
    auto runModeKey = static_cast<uint32_t>(tilingData->runModeKey);
    auto widthKey = static_cast<uint32_t>(tilingData->widthKey);
    auto fnPlanKey = static_cast<uint32_t>(tilingData->fnPlanKey);
    auto dtypeKey = static_cast<uint32_t>(tilingData->dtypeKey);

    if (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE) {
        if (dtypeKey == 0) {
            RunCausalConv1dUpdate<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices,
                                              initialStateMode, numAcceptedTokens, y, userWorkspace, tilingData);
        } else {
            RunCausalConv1dUpdate<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode,
                                        numAcceptedTokens, y, userWorkspace, tilingData);
        }
        return;
    }

    if (dtypeKey == 0) {
        uint32_t effectiveFnPlan =
            (fnPlanKey == CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD) ? fnPlanKey : CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        DispatchFn<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode,
                               numAcceptedTokens, y, userWorkspace, tilingData, widthKey, effectiveFnPlan);
    } else {
        uint32_t effectiveFnPlan =
            (fnPlanKey == CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD) ? fnPlanKey : CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        DispatchFn<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
                         y, userWorkspace, tilingData, widthKey, effectiveFnPlan);
    }
}

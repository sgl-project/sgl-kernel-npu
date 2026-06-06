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
 * \file causal_conv1d_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

class CausalConv1d : public OpDef {
public:
    explicit CausalConv1d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("convStates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("queryStartLoc")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("cacheIndices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("initialStateMode")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("numAcceptedTokens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("activationMode").AttrType(OPTIONAL).Int(0);
        this->Attr("padSlotId").AttrType(OPTIONAL).Int(-1);
        this->Attr("runMode").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(CausalConv1d);

} // namespace ops


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
 * \file causal_conv1d_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeCausalConv1d(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCausalConv1d");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCausalConv1d");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CausalConv1d).InferShape(InferShapeCausalConv1d);
} // namespace ops


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
 * \file causal_conv1d_tiling_planner.h
 * \brief
 */
#ifndef CAUSAL_CONV1D_TILING_PLANNER_H
#define CAUSAL_CONV1D_TILING_PLANNER_H

#include "causal_conv1d_tiling_utils.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"

namespace optiling::causal_conv1d_host {

using namespace Ops::Transformer::OpTiling;

inline DimTileChoice ChooseCanonicalUpdateBaseDimChoice(gert::TilingContext *context, int64_t batch, int64_t dim,
                                                        uint32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto chooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t baseDim : candidates) {
            if (baseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % baseDim != 0)) {
                continue;
            }

            const int64_t baseDimCnt = requireExactDiv ? (dim / baseDim) : CeilDivInt64(dim, baseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }

            OP_LOGD(context,
                    "DimTile(update) candidate[%s]: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld], coreNum[%u].",
                    requireExactDiv ? "exact" : "tail", baseDim, baseDimCnt, gridSize, coreNum);
            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    // bestOver = {baseDim, baseDimCnt, gridSize};
                    bestOver.baseDim = baseDim;
                    bestOver.baseDimCnt = baseDimCnt;
                    bestOver.gridSize = gridSize;
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && baseDim < bestUnder.baseDim)) {
                // bestUnder = {baseDim, baseDimCnt, gridSize};
                bestUnder.baseDim = baseDim;
                bestUnder.baseDimCnt = baseDimCnt;
                bestUnder.gridSize = gridSize;
            }
        }

        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    DimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    OP_LOGD(context, "DimTile(update) chosen: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld].", result.baseDim,
            result.baseDimCnt, result.gridSize);
    return result;
}

inline int64_t ResolveFnTokenCoreBudget(int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    if (baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return 0;
    }

    int64_t tokenCoreBudget = static_cast<int64_t>(coreNum);
    if (fnExecutionPlan == FN_EXECUTION_PLAN_CUTBSD) {
        tokenCoreBudget = std::max<int64_t>(1, tokenCoreBudget / baseDimCnt);
    }
    return tokenCoreBudget;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum);

inline int64_t ComputeFnUbLimitedBaseDim(uint64_t ubSize)
{
    if (ubSize <= static_cast<uint64_t>(FN_UB_RESERVED_BYTES)) {
        return 0;
    }

    const int64_t bytesPerElem = (RING_SLOT_CNT * BF16_FP16_ELEM_BYTES) + (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_CALC_FP32_SLOT_CNT * static_cast<int64_t>(sizeof(float)));
    const int64_t budgetBytes = static_cast<int64_t>(ubSize) - FN_UB_RESERVED_BYTES;
    const int64_t ubLimitedBaseDim = AlignDownInt64(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS);
    return std::min<int64_t>(MAX_DIM_TILE_SIZE, ubLimitedBaseDim);
}

inline DimTileChoice ChooseFnTokenFirstBaseDimChoice(int64_t dim)
{
    if (dim <= 0 || dim > MAX_DIM_TILE_SIZE) {
        return {};
    }
    DimTileChoice choice;
    choice.baseDim = dim;
    choice.baseDimCnt = 1;
    choice.gridSize = 1;
    return choice;
}

inline DimTileChoice ChooseFnTokenDimCoSplitBaseDimChoice(gert::TilingContext *context, int64_t dim, uint64_t ubSize,
                                                          uint32_t coreNum)
{
    if (dim <= 0) {
        return {};
    }

    const int64_t ubLimitedBaseDim = ComputeFnUbLimitedBaseDim(ubSize);
    if (ubLimitedBaseDim <= 0) {
        OP_LOGD(context, "FnDimCoSplit: UB budget is too small to form a valid baseDim.");
        return {};
    }

    DimTileChoice result;
    result.baseDim = ubLimitedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;

    if (coreNum == 0 || result.baseDimCnt <= 1 || result.baseDimCnt >= static_cast<int64_t>(coreNum) ||
        (coreNum % result.baseDimCnt == 0)) {
        OP_LOGD(context,
                "FnDimCoSplit: dim[%ld], ubLimitedBaseDim[%ld], baseDimCnt[%ld], coreNum[%u], adjusted[%d].", dim,
                result.baseDim, result.baseDimCnt, coreNum, 0);
        return result;
    }

    int64_t adjustedBaseDimCnt = result.baseDimCnt;
    while (adjustedBaseDimCnt < static_cast<int64_t>(coreNum) && (coreNum % adjustedBaseDimCnt != 0)) {
        ++adjustedBaseDimCnt;
    }

    if (adjustedBaseDimCnt >= static_cast<int64_t>(coreNum)) {
        OP_LOGD(context,
                "FnDimCoSplit: keep baseDimCnt[%ld] because no divisible adjustment exists under coreNum[%u].",
                result.baseDimCnt, coreNum);
        return result;
    }

    const int64_t adjustedBaseDim = AlignUpInt64(CeilDivInt64(dim, adjustedBaseDimCnt), DIM_ALIGN_ELEMS);
    if (adjustedBaseDim <= 0 || adjustedBaseDim > ubLimitedBaseDim || adjustedBaseDim > MAX_DIM_TILE_SIZE) {
        OP_LOGD(context,
                "FnDimCoSplit: rejected adjusted baseDim[%ld] with baseDimCnt[%ld], ubLimitedBaseDim[%ld].",
                adjustedBaseDim, adjustedBaseDimCnt, ubLimitedBaseDim);
        return result;
    }

    result.baseDim = adjustedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;
    OP_LOGD(context,
            "FnDimCoSplit: dim[%ld], ubLimitedBaseDim[%ld], adjustedBaseDim[%ld], baseDimCnt[%ld], coreNum[%u].",
            dim, ubLimitedBaseDim, result.baseDim, result.baseDimCnt, coreNum);
    return result;
}

inline TokenCoreMappingChoice BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt, int64_t baseDimCnt,
                                                            FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    TokenCoreMappingChoice mapping;
    mapping.tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (tokenBlockCnt <= 0 || mapping.tokenCoreBudget <= 0 || baseDimCnt <= 0) {
        return mapping;
    }

    mapping.tokenBlocksPerCore = CeilDivInt64(tokenBlockCnt, mapping.tokenCoreBudget);
    mapping.tokenCoreTailCnt =
        tokenBlockCnt - (std::max<int64_t>(0, mapping.tokenBlocksPerCore - 1) * mapping.tokenCoreBudget);
    if (mapping.tokenCoreTailCnt <= 0) {
        mapping.tokenCoreTailCnt = mapping.tokenCoreBudget;
    }
    mapping.blockDim = mapping.tokenCoreBudget * baseDimCnt;
    return mapping;
}

inline FnTokenSeqRangePlan BuildFnTokenSeqRangePlan(const int64_t *qslData, int64_t batch, int64_t tokenBlockSize,
                                                    int64_t tokenBlockCnt)
{
    FnTokenSeqRangePlan plan;
    if (qslData == nullptr || batch <= 0 || tokenBlockSize <= 0 || tokenBlockCnt <= 0 ||
        tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
        return plan;
    }

    plan.enabled = true;
    plan.rangeCount = tokenBlockCnt;
    int64_t seq = 0;
    for (int64_t tokenTileId = 0; tokenTileId < tokenBlockCnt; ++tokenTileId) {
        const int64_t tokenStart = tokenTileId * tokenBlockSize;
        const int64_t tokenEnd = tokenStart + tokenBlockSize;

        while (seq < batch && qslData[seq + 1] <= tokenStart) {
            ++seq;
        }

        int64_t endSeq = seq;
        while (endSeq < batch && qslData[endSeq] < tokenEnd) {
            ++endSeq;
        }

        plan.tokenTileStartSeq[tokenTileId] = seq;
        plan.tokenTileEndSeq[tokenTileId] = endSeq;
    }
    return plan;
}

inline VarlenTokenTileChoice ChooseUnifiedFnTokenBlockPlan(gert::TilingContext *context,
                                                           const CausalConv1dTilingData &tiling,
                                                           const DimTileChoice &baseDimChoice,
                                                           FnExecutionPlan fnExecutionPlan,
                                                           uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    if ((tiling.inputMode != 0 && tiling.inputMode != 1) || tiling.batch <= 0 || tiling.cuSeqlen <= 0 ||
        baseDimChoice.baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return tokenBlockChoice;
    }
    if (tiling.hasNumAcceptedTokens != 0) {
        OP_LOGD(context, "Varlen token tiling disabled: speculative decode still uses the existing seq mapping.");
        return tokenBlockChoice;
    }

    tokenBlockChoice = ChooseFnTokenBlockChoice(tiling.cuSeqlen, baseDimChoice.baseDimCnt, fnExecutionPlan, coreNum);

    OP_LOGD(context,
            "FnTokenTile(plan=%ld): cuSeqlen[%ld], baseDimCnt[%ld], tokenBlockSize[%ld], "
            "tokenBlockCnt[%ld], gridSize[%ld].",
            static_cast<int64_t>(fnExecutionPlan), tiling.cuSeqlen, baseDimChoice.baseDimCnt,
            tokenBlockChoice.tokenBlockSize, tokenBlockChoice.tokenBlockCnt, tokenBlockChoice.gridSize);
    return tokenBlockChoice;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    const int64_t tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (cuSeqlen <= 0 || tokenCoreBudget <= 0) {
        return tokenBlockChoice;
    }

    tokenBlockChoice.enabled = true;
    const int64_t idealBlockSize = CeilDivInt64(cuSeqlen, tokenCoreBudget);
    tokenBlockChoice.tokenBlockSize = (idealBlockSize > 0) ? idealBlockSize : 1;
    tokenBlockChoice.tokenBlockCnt = CeilDivInt64(cuSeqlen, tokenBlockChoice.tokenBlockSize);
    tokenBlockChoice.gridSize = tokenBlockChoice.tokenBlockCnt * baseDimCnt;
    return tokenBlockChoice;
}

inline FnHostPlan ChooseFnHostPlan(gert::TilingContext *context, const CausalConv1dTilingData &tiling, uint64_t ubSize,
                                   uint32_t coreNum)
{
    FnHostPlan plan;
    if ((tiling.inputMode != 0 && tiling.inputMode != 1) || tiling.batch <= 0 || tiling.cuSeqlen <= 0 ||
        tiling.dim <= 0 || coreNum == 0) {
        return plan;
    }

    if (tiling.dim <= MAX_DIM_TILE_SIZE) {
        plan.caseKind = FN_TILING_CASE_TOKEN_FIRST;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBS;
        plan.baseDimChoice = ChooseFnTokenFirstBaseDimChoice(tiling.dim);
    } else {
        plan.caseKind = FN_TILING_CASE_TOKEN_DIM_CO_SPLIT;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBSD;
        plan.baseDimChoice = ChooseFnTokenDimCoSplitBaseDimChoice(context, tiling.dim, ubSize, coreNum);
    }

    if (plan.baseDimChoice.baseDim <= 0 || plan.baseDimChoice.baseDimCnt <= 0) {
        return {};
    }

    plan.baseDimChoice.gridSize = tiling.batch * plan.baseDimChoice.baseDimCnt;
    plan.tokenBlockChoice =
        ChooseUnifiedFnTokenBlockPlan(context, tiling, plan.baseDimChoice, plan.executionPlan, coreNum);
    if (!plan.tokenBlockChoice.enabled || plan.tokenBlockChoice.tokenBlockSize <= 0 ||
        plan.tokenBlockChoice.tokenBlockCnt <= 0 || plan.tokenBlockChoice.gridSize <= 0) {
        return {};
    }

    plan.tokenCoreMapping = BuildFnTokenCoreMappingChoice(plan.tokenBlockChoice.tokenBlockCnt,
                                                          plan.baseDimChoice.baseDimCnt, plan.executionPlan, coreNum);
    if (plan.tokenCoreMapping.tokenCoreBudget <= 0 || plan.tokenCoreMapping.blockDim <= 0) {
        return {};
    }
    if (plan.tokenCoreMapping.blockDim > static_cast<int64_t>(coreNum)) {
        plan.tokenCoreMapping.blockDim = static_cast<int64_t>(coreNum);
    }
    return plan;
}

} // namespace optiling::causal_conv1d_host

#endif // CAUSAL_CONV1D_TILING_PLANNER_H


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
 * \file causal_conv1d_tiling_utils.h
 * \brief
 */
#ifndef CAUSAL_CONV1D_TILING_UTILS_H
#define CAUSAL_CONV1D_TILING_UTILS_H

#include "tiling_base/tiling_util.h"
#include "../op_kernel/causal_conv1d_tiling_key.h"

namespace optiling::causal_conv1d_host {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t CONV_STATES_INDEX = 3;
constexpr uint32_t QUERY_START_LOC_INDEX = 4;
constexpr uint32_t CACHE_INDICES_INDEX = 5;
constexpr uint32_t INITIAL_STATE_MODE_INDEX = 6;
constexpr uint32_t NUM_ACCEPTED_TOKENS_INDEX = 7;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;
constexpr int64_t ASCENDC_RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;

struct CausalConv1dCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

struct CausalConv1dAttrInfo {
    int64_t activationMode = 0;
    int64_t padSlotId = -1;
    int64_t runMode = 0;
};

struct DimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

struct VarlenTokenTileChoice {
    bool enabled = false;
    int64_t tokenBlockSize = 0;
    int64_t tokenBlockCnt = 0;
    int64_t gridSize = 0;
};

enum FnTilingCaseKind : int64_t {
    FN_TILING_CASE_INVALID = 0,
    FN_TILING_CASE_TOKEN_FIRST = 1,
    FN_TILING_CASE_TOKEN_DIM_CO_SPLIT = 2,
};

struct TokenCoreMappingChoice {
    int64_t tokenCoreBudget = 0;
    int64_t tokenBlocksPerCore = 0;
    int64_t tokenCoreTailCnt = 0;
    int64_t blockDim = 0;
};

constexpr int64_t MAX_FN_TOKEN_SEQ_RANGE_COUNT = 128;

struct FnTokenSeqRangePlan {
    bool enabled = false;
    int64_t rangeCount = 0;
    int64_t tokenTileStartSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
    int64_t tokenTileEndSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
};

struct FnHostPlan {
    FnTilingCaseKind caseKind = FN_TILING_CASE_INVALID;
    FnExecutionPlan executionPlan = FN_EXECUTION_PLAN_INVALID;
    DimTileChoice baseDimChoice;
    VarlenTokenTileChoice tokenBlockChoice;
    TokenCoreMappingChoice tokenCoreMapping;
    FnTokenSeqRangePlan tokenSeqRangePlan;
};

constexpr int64_t DIM_ALIGN_BYTES = 32;
constexpr int64_t BF16_FP16_ELEM_BYTES = 2;
constexpr int64_t DIM_ALIGN_ELEMS = DIM_ALIGN_BYTES / BF16_FP16_ELEM_BYTES;
constexpr int64_t MAX_DIM_TILE_SIZE = 4096;
constexpr int64_t FN_UB_RESERVED_BYTES = 512;
constexpr int64_t RING_SLOT_CNT = 5;
constexpr int64_t FN_OUT_SLOT_CNT = 2;
constexpr int64_t FN_CALC_FP32_SLOT_CNT = 8;

inline uint32_t NormalizeFnPlanTilingKey(uint32_t runModeKey, FnExecutionPlan fnExecutionPlan)
{
    if (runModeKey != CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
    switch (fnExecutionPlan) {
        case FN_EXECUTION_PLAN_CUTBS:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        case FN_EXECUTION_PLAN_CUTBSD:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD;
        default:
            return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
}

inline uint32_t NormalizeWidthTilingKey(uint32_t runModeKey, int32_t width)
{
    if (runModeKey != CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
    switch (width) {
        case 2:
            return CAUSAL_CONV1D_TPL_WIDTH_2;
        case 3:
            return CAUSAL_CONV1D_TPL_WIDTH_3;
        case 4:
            return CAUSAL_CONV1D_TPL_WIDTH_4;
        default:
            return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
}

inline int64_t CeilDivInt64(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t AlignDownInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return (value / align) * align;
}

inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return CeilDivInt64(value, align) * align;
}

inline const char *GetFnTilingCaseName(FnTilingCaseKind caseKind)
{
    switch (caseKind) {
        case FN_TILING_CASE_TOKEN_FIRST:
            return "token_first";
        case FN_TILING_CASE_TOKEN_DIM_CO_SPLIT:
            return "token_dim_co_split";
        default:
            return "invalid";
    }
}

} // namespace optiling::causal_conv1d_host

#endif // CAUSAL_CONV1D_TILING_UTILS_H


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
 * \file causal_conv1d_tiling_validation.h
 * \brief
 */
 #ifndef CAUSAL_CONV1D_TILING_VALIDATION_H
 #define CAUSAL_CONV1D_TILING_VALIDATION_H
 
 #include "tiling_base/tiling_util.h"
 #include "causal_conv1d_tiling_utils.h"
 #include "../op_kernel/causal_conv1d_tiling_data.h"
 
 namespace optiling::causal_conv1d_host {
 
 using namespace Ops::Transformer::OpTiling;
 
 inline ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, uint32_t &coreNum)
 {
     fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
     coreNum = ascendcPlatform.GetCoreNumAiv();
     OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
     OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }
 
 inline ge::graphStatus SetWorkspaceSize(gert::TilingContext *context, size_t workspaceSize)
 {
     size_t *currentWorkspace = context->GetWorkspaceSizes(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
     currentWorkspace[0] = workspaceSize;
     return ge::GRAPH_SUCCESS;
 }
 
 inline ge::graphStatus GetAttrsInfo(gert::TilingContext *context, CausalConv1dAttrInfo &attrInfo)
 {
     auto attrs = context->GetAttrs();
     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
 
     const int64_t *activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, activationModePtr);
     attrInfo.activationMode = *activationModePtr;
     OP_CHECK_IF(attrInfo.activationMode != 0 && attrInfo.activationMode != 1,
                 OP_LOGE(context, "activationMode only supports 0/1"),
                 return ge::GRAPH_FAILED);
 
     const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, padSlotIdPtr);
     attrInfo.padSlotId = *padSlotIdPtr;
 
     const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
     attrInfo.runMode = (runModePtr == nullptr) ? 0 : *runModePtr;
     OP_CHECK_IF(attrInfo.runMode != 0 && attrInfo.runMode != 1, OP_LOGE(context, "runMode only supports 0/1"),
                 return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }
 
 inline ge::graphStatus ValidateAlignedDim(gert::TilingContext *context, int64_t dim)
 {
     OP_CHECK_IF(dim % DIM_ALIGN_ELEMS != 0,
                 OP_LOGE(context,
                         "dim must satisfy dim %% %ld == 0 for causal_conv1d; "
                         "x/weight/convStates last dimension and bias length must all use the same aligned dim, "
                         "got dim=%ld.",
                         DIM_ALIGN_ELEMS, dim),
                 return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }
 
 inline ge::graphStatus GetShapeDtypeInfo(gert::TilingContext *context, const CausalConv1dAttrInfo &attrInfo,
                                          CausalConv1dTilingData &tiling, bool &hasBias)
 {
     const bool isDecodeMode = (attrInfo.runMode == 1);
     tiling.activationMode = attrInfo.activationMode;
     tiling.padSlotId = attrInfo.padSlotId;
 
     auto xShapePtr = context->GetInputShape(X_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
     auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
 
     int64_t dim = 0;
     int64_t cuSeqlen = 0;
     int64_t seqLen = 0;
     int64_t batch = 0;
     int64_t inputMode = 0;
 
     if (xShape.GetDimNum() == 2) {
         if (isDecodeMode) {
             inputMode = 2;
             batch = xShape.GetDim(0);
             dim = xShape.GetDim(1);
             seqLen = 1;
             cuSeqlen = batch;
             OP_CHECK_IF(batch <= 0 || dim <= 0, OP_LOGE(context, "invalid x shape for 2D decode mode"),
                         return ge::GRAPH_FAILED);
         } else {
             inputMode = 0;
             cuSeqlen = xShape.GetDim(0);
             dim = xShape.GetDim(1);
             seqLen = 0;
             OP_CHECK_IF(dim <= 0 || cuSeqlen < 0, OP_LOGE(context, "invalid x shape for 2D varlen mode"),
                         return ge::GRAPH_FAILED);
         }
     } else if (xShape.GetDimNum() == 3) {
         inputMode = 1;
         batch = xShape.GetDim(0);
         seqLen = xShape.GetDim(1);
         dim = xShape.GetDim(2);
         cuSeqlen = batch * seqLen;
         OP_CHECK_IF(batch <= 0 || dim <= 0 || seqLen <= 0, OP_LOGE(context, "invalid x shape for 3D batch mode"),
                     return ge::GRAPH_FAILED);
     } else {
         OP_LOGE(context, "x must be 2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
         return ge::GRAPH_FAILED;
     }
     OP_CHECK_IF(ValidateAlignedDim(context, dim) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "dim alignment validation failed"),
                 return ge::GRAPH_FAILED);
 
     auto wShapePtr = context->GetInputShape(WEIGHT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
     auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
     OP_CHECK_IF(wShape.GetDimNum() != 2, OP_LOGE(context, "weight must be 2D: (width, dim)"), return ge::GRAPH_FAILED);
     const int64_t width = wShape.GetDim(0);
     const int64_t wDim = wShape.GetDim(1);
     OP_CHECK_IF(wDim != dim, OP_LOGE(context, "weight.shape[1] must equal dim"), return ge::GRAPH_FAILED);
     OP_CHECK_IF(width < 2 || width > 4, OP_LOGE(context, "Only support width in [2,4] now, actually is %ld.", width),
                 return ge::GRAPH_FAILED);
 
     auto sShapePtr = context->GetInputShape(CONV_STATES_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, sShapePtr);
     auto sShape = EnsureNotScalar(sShapePtr->GetStorageShape());
     OP_CHECK_IF(sShape.GetDimNum() != 3, OP_LOGE(context, "convStates must be 3D: (num_cache_lines, state_len, dim)"),
                 return ge::GRAPH_FAILED);
     const int64_t numCacheLines = sShape.GetDim(0);
     const int64_t stateLen = sShape.GetDim(1);
     const int64_t sDim = sShape.GetDim(2);
     OP_CHECK_IF(numCacheLines <= 0, OP_LOGE(context, "convStates.shape[0] (num_cache_lines) must be > 0"),
                 return ge::GRAPH_FAILED);
     OP_CHECK_IF(sDim != dim, OP_LOGE(context, "convStates.shape[2] must equal dim"), return ge::GRAPH_FAILED);
     OP_CHECK_IF(stateLen < (width - 1), OP_LOGE(context, "convStates.shape[1] must be >= width-1"),
                 return ge::GRAPH_FAILED);
 
     auto qslShapePtr = context->GetOptionalInputShape(QUERY_START_LOC_INDEX);
     const gert::CompileTimeTensorDesc *qslDesc = context->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
     bool qslAbsent = true;
     int64_t qslSize = 0;
     if (qslShapePtr != nullptr) {
         const auto qslStorageShape = qslShapePtr->GetStorageShape();
         const int64_t qslDimNum = qslStorageShape.GetDimNum();
         qslAbsent = (qslDimNum == 0) || (qslDimNum == 1 && qslStorageShape.GetDim(0) <= 0);
         if (!qslAbsent) {
             auto qslShape = EnsureNotScalar(qslStorageShape);
             OP_CHECK_IF(qslShape.GetDimNum() != 1, OP_LOGE(context, "queryStartLoc must be 1D"),
                         return ge::GRAPH_FAILED);
             qslSize = qslShape.GetDim(0);
             OP_CHECK_IF(qslSize < 1, OP_LOGE(context, "queryStartLoc.size must be >= 1"), return ge::GRAPH_FAILED);
             OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
             OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "queryStartLoc dtype must be int64"),
                         return ge::GRAPH_FAILED);
         }
     }
 
     if (qslAbsent) {
         OP_CHECK_IF(inputMode == 0, OP_LOGE(context, "queryStartLoc is required in 2D varlen mode (inputMode=0)"),
                     return ge::GRAPH_FAILED);
         qslSize = batch + 1;
     }
 
     OP_CHECK_IF(cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                 OP_LOGE(context, "cuSeqlen is too large for int32 indexing, got %ld", cuSeqlen),
                 return ge::GRAPH_FAILED);
 
     const int64_t *qslData = nullptr;
     if (!qslAbsent) {
         const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INDEX);
         qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
         if (qslData != nullptr) {
             OP_CHECK_IF(qslData[0] != 0, OP_LOGE(context, "queryStartLoc[0] must be 0"), return ge::GRAPH_FAILED);
             OP_CHECK_IF(qslData[qslSize - 1] != cuSeqlen,
                         OP_LOGE(context, "queryStartLoc[last] must equal cuSeqlen, got %ld vs %ld",
                                 qslData[qslSize - 1], cuSeqlen),
                         return ge::GRAPH_FAILED);
             for (int64_t i = 0; i + 1 < qslSize; ++i) {
                 const int64_t cur = qslData[i];
                 const int64_t nxt = qslData[i + 1];
                 OP_CHECK_IF(cur < 0 || cur > cuSeqlen,
                             OP_LOGE(context, "queryStartLoc[%ld] out of range: %ld (cuSeqlen=%ld)", i, cur, cuSeqlen),
                             return ge::GRAPH_FAILED);
                 OP_CHECK_IF(
                     nxt < 0 || nxt > cuSeqlen,
                     OP_LOGE(context, "queryStartLoc[%ld] out of range: %ld (cuSeqlen=%ld)", i + 1, nxt, cuSeqlen),
                     return ge::GRAPH_FAILED);
                 OP_CHECK_IF(
                     nxt < cur,
                     OP_LOGE(context,
                             "queryStartLoc must be non-decreasing, got queryStartLoc[%ld]=%ld queryStartLoc[%ld]=%ld",
                             i, cur, i + 1, nxt),
                     return ge::GRAPH_FAILED);
             }
         }
     }
 
     if (!qslAbsent && isDecodeMode && inputMode == 2) {
         const int64_t batchFromQsl = qslSize - 1;
         if (batchFromQsl != batch) {
             inputMode = 0;
             cuSeqlen = xShape.GetDim(0);
             batch = batchFromQsl;
             seqLen = 0;
             OP_CHECK_IF(dim <= 0 || cuSeqlen < 0 || batch < 0,
                         OP_LOGE(context, "invalid x/queryStartLoc shapes for 2D varlen decode mode"),
                         return ge::GRAPH_FAILED);
         }
     }
 
     if (inputMode == 0) {
         batch = qslSize - 1;
     }
     if (!qslAbsent && (inputMode == 1 || inputMode == 2)) {
         OP_CHECK_IF(qslSize != batch + 1, OP_LOGE(context, "queryStartLoc.size must equal batch + 1"),
                     return ge::GRAPH_FAILED);
     }
     if (isDecodeMode) {
         const int64_t decodeSeqLen = (inputMode == 1) ? seqLen : 1;
         OP_CHECK_IF(decodeSeqLen < 1, OP_LOGE(context, "decode mode requires seqlen >= 1, actual is %ld", decodeSeqLen),
                     return ge::GRAPH_FAILED);
     }
 
     tiling.hasCacheIndices = 0;
     bool ciAbsent = true;
     auto ciShapePtr = context->GetOptionalInputShape(CACHE_INDICES_INDEX);
     if (ciShapePtr != nullptr) {
         const auto ciStorageShape = ciShapePtr->GetStorageShape();
         const int64_t ciDimNum = ciStorageShape.GetDimNum();
         ciAbsent = (ciDimNum == 0) || (ciDimNum == 1 && ciStorageShape.GetDim(0) <= 0);
         if (!ciAbsent) {
             auto ciShape = EnsureNotScalar(ciStorageShape);
             OP_CHECK_IF(ciShape.GetDimNum() != 1, OP_LOGE(context, "cacheIndices must be 1D"), return ge::GRAPH_FAILED);
             OP_CHECK_IF(ciShape.GetDim(0) != batch, OP_LOGE(context, "cacheIndices.size must equal batch"),
                         return ge::GRAPH_FAILED);
             tiling.hasCacheIndices = 1;
 
             const gert::Tensor *ciTensor = context->GetOptionalInputTensor(CACHE_INDICES_INDEX);
             const int64_t *ciData = (ciTensor != nullptr) ? ciTensor->GetData<int64_t>() : nullptr;
             if (ciData != nullptr) {
                 for (int64_t i = 0; i < batch; ++i) {
                     const int64_t v = ciData[i];
                     if (v == tiling.padSlotId) {
                         continue;
                     }
                     OP_CHECK_IF(v > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                                 OP_LOGE(context, "cacheIndices[%ld]=%ld exceeds int32 range", i, v),
                                 return ge::GRAPH_FAILED);
                     OP_CHECK_IF(v < 0 || v >= numCacheLines,
                                 OP_LOGE(context,
                                         "cacheIndices[%ld]=%ld out of range [0, num_cache_lines=%ld), padSlotId=%ld", i,
                                         v, numCacheLines, tiling.padSlotId),
                                 return ge::GRAPH_FAILED);
                 }
             }
         }
     }
     if (ciAbsent) {
         OP_CHECK_IF(numCacheLines < batch,
                     OP_LOGE(context,
                             "cacheIndices is absent, requires convStates.shape[0] (num_cache_lines) >= batch for "
                             "identity mapping, got num_cache_lines=%ld batch=%ld",
                             numCacheLines, batch),
                     return ge::GRAPH_FAILED);
     }
 
     tiling.hasInitialStateMode = 0;
     auto ismShapePtr = context->GetOptionalInputShape(INITIAL_STATE_MODE_INDEX);
     if (ismShapePtr != nullptr) {
         const auto ismStorageShape = ismShapePtr->GetStorageShape();
         const int64_t ismDimNum = ismStorageShape.GetDimNum();
         const bool ismAbsent = (ismDimNum == 0) || (ismDimNum == 1 && ismStorageShape.GetDim(0) <= 0);
         if (!ismAbsent) {
             OP_CHECK_IF(isDecodeMode,
                         OP_LOGE(context, "initialStateMode is only supported in runMode=0 (fn/prefill)"),
                         return ge::GRAPH_FAILED);
             auto ismShape = EnsureNotScalar(ismStorageShape);
             OP_CHECK_IF(ismShape.GetDimNum() != 1, OP_LOGE(context, "initialStateMode must be 1D"),
                         return ge::GRAPH_FAILED);
             OP_CHECK_IF(ismShape.GetDim(0) != batch, OP_LOGE(context, "initialStateMode.size must equal batch"),
                         return ge::GRAPH_FAILED);
 
             const gert::Tensor *ismTensor = context->GetOptionalInputTensor(INITIAL_STATE_MODE_INDEX);
             const int64_t *ismData = (ismTensor != nullptr) ? ismTensor->GetData<int64_t>() : nullptr;
             if (ismData != nullptr) {
                 bool hasNonZeroInitialStateMode = false;
                 for (int64_t i = 0; i < batch; ++i) {
                     const int64_t v = ismData[i];
                     OP_CHECK_IF(v != 0 && v != 1,
                                 OP_LOGE(context, "initialStateMode[%ld]=%ld is invalid (only supports 0/1)", i, v),
                                 return ge::GRAPH_FAILED);
                     hasNonZeroInitialStateMode = hasNonZeroInitialStateMode || (v != 0);
                 }
                 tiling.hasInitialStateMode = hasNonZeroInitialStateMode ? 1 : 0;
             } else {
                 tiling.hasInitialStateMode = 1;
             }
         }
     }
 
     tiling.hasNumAcceptedTokens = 0;
     auto natShapePtr = context->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX);
     if (natShapePtr != nullptr) {
         const auto natStorageShape = natShapePtr->GetStorageShape();
         const int64_t natDimNum = natStorageShape.GetDimNum();
         const bool natAbsent = (natDimNum == 0) || (natDimNum == 1 && natStorageShape.GetDim(0) <= 0);
         if (!natAbsent) {
             OP_CHECK_IF(!isDecodeMode,
                         OP_LOGE(context, "numAcceptedTokens is only supported in runMode=1 (decode/update)"),
                         return ge::GRAPH_FAILED);
             auto natShape = EnsureNotScalar(natStorageShape);
             OP_CHECK_IF(natShape.GetDimNum() != 1, OP_LOGE(context, "numAcceptedTokens must be 1D"),
                         return ge::GRAPH_FAILED);
             OP_CHECK_IF(natShape.GetDim(0) != batch, OP_LOGE(context, "numAcceptedTokens.size must equal batch"),
                         return ge::GRAPH_FAILED);
 
             if (inputMode == 1) {
                 const int64_t reqStateLen = (width - 1) + (seqLen - 1);
                 OP_CHECK_IF(stateLen < reqStateLen,
                             OP_LOGE(context,
                                     "spec decode requires stateLen >= (width-1) + (seqlen-1), got stateLen=%ld req=%ld",
                                     stateLen, reqStateLen),
                             return ge::GRAPH_FAILED);
             }
 
             const gert::Tensor *natTensor = context->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX);
             const int64_t *natData = (natTensor != nullptr) ? natTensor->GetData<int64_t>() : nullptr;
             if (natData != nullptr) {
                 for (int64_t i = 0; i < batch; ++i) {
                     const int64_t a = natData[i];
                     OP_CHECK_IF(a < 0, OP_LOGE(context, "numAcceptedTokens[%ld]=%ld is invalid (must be >= 0)", i, a),
                                 return ge::GRAPH_FAILED);
 
                     if (inputMode == 2) {
                         OP_CHECK_IF(
                             a > 1,
                             OP_LOGE(context, "numAcceptedTokens[%ld]=%ld exceeds decode 2D token count (1)", i, a),
                             return ge::GRAPH_FAILED);
                     } else if (inputMode == 1) {
                         OP_CHECK_IF(a > seqLen,
                                     OP_LOGE(context, "numAcceptedTokens[%ld]=%ld exceeds seqlen=%ld in 3D update", i, a,
                                             seqLen),
                                     return ge::GRAPH_FAILED);
                     } else if (inputMode == 0 && qslData != nullptr) {
                         const int64_t lenI = qslData[i + 1] - qslData[i];
                         OP_CHECK_IF(a > lenI,
                                     OP_LOGE(context, "numAcceptedTokens[%ld]=%ld exceeds varlen segment length=%ld", i,
                                             a, lenI),
                                     return ge::GRAPH_FAILED);
                     }
                 }
             }
 
             tiling.hasNumAcceptedTokens = 1;
         }
     }
 
     tiling.hasBias = 0;
     hasBias = false;
     auto biasShapePtr = context->GetOptionalInputShape(BIAS_INDEX);
     if (biasShapePtr != nullptr) {
         const auto biasStorageShape = biasShapePtr->GetStorageShape();
         const int64_t biasDimNum = biasStorageShape.GetDimNum();
         const bool biasAbsent = (biasDimNum == 0) || (biasDimNum == 1 && biasStorageShape.GetDim(0) <= 0);
         if (!biasAbsent) {
             auto biasShape = EnsureNotScalar(biasStorageShape);
             OP_CHECK_IF(biasShape.GetDimNum() != 1, OP_LOGE(context, "bias must be 1D: (dim,)"),
                         return ge::GRAPH_FAILED);
             OP_CHECK_IF(biasShape.GetDim(0) != dim, OP_LOGE(context, "bias.size must equal dim"),
                         return ge::GRAPH_FAILED);
             tiling.hasBias = 1;
             hasBias = true;
         }
     }
 
     auto xDesc = context->GetInputDesc(X_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
     const ge::DataType xDtype = xDesc->GetDataType();
     OP_CHECK_IF(xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
                 OP_LOGE(context, "x dtype only supports bf16/fp16"),
                 return ge::GRAPH_FAILED);
 
     auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
     OP_CHECK_IF(wDesc->GetDataType() != xDtype, OP_LOGE(context, "weight dtype must equal x dtype"),
                 return ge::GRAPH_FAILED);
 
     if (hasBias) {
         auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
         OP_CHECK_IF(biasDesc->GetDataType() != xDtype, OP_LOGE(context, "bias dtype must equal x dtype"),
                     return ge::GRAPH_FAILED);
     }
 
     auto sDesc = context->GetInputDesc(CONV_STATES_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, sDesc);
     OP_CHECK_IF(sDesc->GetDataType() != xDtype, OP_LOGE(context, "convStates dtype must equal x dtype"),
                 return ge::GRAPH_FAILED);
 
     if (!qslAbsent) {
         auto qslDesc2 = context->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc2);
         OP_CHECK_IF(qslDesc2->GetDataType() != ge::DT_INT64, OP_LOGE(context, "queryStartLoc dtype must be int64"),
                     return ge::GRAPH_FAILED);
     }
     if (tiling.hasCacheIndices == 1) {
         auto ciDesc = context->GetOptionalInputDesc(CACHE_INDICES_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, ciDesc);
         OP_CHECK_IF(ciDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "cacheIndices dtype must be int64"),
                     return ge::GRAPH_FAILED);
     }
     if (tiling.hasInitialStateMode == 1) {
         auto ismDesc = context->GetOptionalInputDesc(INITIAL_STATE_MODE_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, ismDesc);
         OP_CHECK_IF(ismDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "initialStateMode dtype must be int64"),
                     return ge::GRAPH_FAILED);
     }
     if (tiling.hasNumAcceptedTokens == 1) {
         OP_CHECK_IF(width != 4, OP_LOGE(context, "numAcceptedTokens is only supported for width=4 currently"),
                     return ge::GRAPH_FAILED);
         auto natDesc = context->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, natDesc);
         OP_CHECK_IF(natDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "numAcceptedTokens dtype must be int64"),
                     return ge::GRAPH_FAILED);
     }
 
     tiling.dim = dim;
     tiling.cuSeqlen = cuSeqlen;
     tiling.seqLen = seqLen;
     tiling.inputMode = inputMode;
     tiling.width = width;
     tiling.stateLen = stateLen;
     tiling.numCacheLines = numCacheLines;
     tiling.batch = batch;
     return ge::GRAPH_SUCCESS;
 }
 
 }
 
 #endif
 

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
 * \file causal_conv1d_tiling.cpp
 */

 #include "tiling_base/tiling_templates_registry.h"
 #include "causal_conv1d_tiling_utils.h"
 #include "causal_conv1d_tiling_planner.h"
 #include "causal_conv1d_tiling_validation.h"
 
 namespace optiling {
 
 using namespace Ops::Transformer::OpTiling;
 using namespace causal_conv1d_host;
 
 static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext *context)
 {
     uint64_t ubSize = 0;
     uint32_t coreNum = 0;
     OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
 
     CausalConv1dTilingData *tiling = context->GetTilingData<CausalConv1dTilingData>();
     OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
     OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dTilingData), 0, sizeof(CausalConv1dTilingData)) != EOK,
                 OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
 
     CausalConv1dAttrInfo attrInfo;
     OP_CHECK_IF(GetAttrsInfo(context, attrInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrsInfo error"),
                 return ge::GRAPH_FAILED);
     bool hasBias = false;
     OP_CHECK_IF(GetShapeDtypeInfo(context, attrInfo, *tiling, hasBias) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetShapeDtypeInfo error"), return ge::GRAPH_FAILED);
 
     const int64_t &dim = tiling->dim;
     const int64_t &batch = tiling->batch;
     OP_CHECK_IF(dim <= 0 || batch <= 0, OP_LOGE(context, "dim/batch must be positive"), return ge::GRAPH_FAILED);
 
     const uint32_t runModeKey = static_cast<uint32_t>(attrInfo.runMode);
     const bool &isFn = (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN);
     const bool &hasActivation = (attrInfo.activationMode != 0);
     const char *plannerModeTag = "update";
     DimTileChoice baseDimChoice;
     FnExecutionPlan fnExecutionPlan = FN_EXECUTION_PLAN_INVALID;
     FnHostPlan fnHostPlan;
     const int64_t *qslData = nullptr;
     if (isFn && tiling->inputMode == 0) {
         const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INDEX);
         qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
     }
 
     if (isFn) {
         fnHostPlan = ChooseFnHostPlan(context, *tiling, ubSize, coreNum);
         plannerModeTag = GetFnTilingCaseName(fnHostPlan.caseKind);
         baseDimChoice = fnHostPlan.baseDimChoice;
         fnExecutionPlan = fnHostPlan.executionPlan;
     } else {
         baseDimChoice = ChooseCanonicalUpdateBaseDimChoice(context, tiling->batch, tiling->dim, coreNum);
     }
 
     OP_CHECK_IF(baseDimChoice.baseDim <= 0 || baseDimChoice.baseDimCnt <= 0 || baseDimChoice.gridSize <= 0,
                 OP_LOGE(context, "invalid dim tile size selection"), return ge::GRAPH_FAILED);
 
     int64_t effectiveGridSize = baseDimChoice.gridSize;
 
     if (isFn) {
         OP_CHECK_IF(fnHostPlan.caseKind == FN_TILING_CASE_INVALID || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID ||
                         !fnHostPlan.tokenBlockChoice.enabled || fnHostPlan.tokenBlockChoice.tokenBlockSize <= 0 ||
                         fnHostPlan.tokenBlockChoice.tokenBlockCnt <= 0 || fnHostPlan.tokenBlockChoice.gridSize <= 0 ||
                         fnHostPlan.tokenCoreMapping.tokenCoreBudget <= 0 || fnHostPlan.tokenCoreMapping.blockDim <= 0,
                     OP_LOGE(context, "runMode=0 must resolve a valid unified token tiling plan"),
                     return ge::GRAPH_FAILED);
 
         tiling->tokenBlockSize = fnHostPlan.tokenBlockChoice.tokenBlockSize;
         tiling->tokenBlockCnt = fnHostPlan.tokenBlockChoice.tokenBlockCnt;
         effectiveGridSize = fnHostPlan.tokenBlockChoice.gridSize;
         if (tiling->inputMode == 0) {
             fnHostPlan.tokenSeqRangePlan =
                 BuildFnTokenSeqRangePlan(qslData, tiling->batch, tiling->tokenBlockSize, tiling->tokenBlockCnt);
             if (fnHostPlan.tokenSeqRangePlan.enabled) {
                 tiling->hasExplicitTokenSeqRanges = 1;
                 tiling->explicitTokenSeqRangeCount = fnHostPlan.tokenSeqRangePlan.rangeCount;
                 for (int64_t i = 0; i < fnHostPlan.tokenSeqRangePlan.rangeCount; ++i) {
                     tiling->tokenTileStartSeq[i] = fnHostPlan.tokenSeqRangePlan.tokenTileStartSeq[i];
                     tiling->tokenTileEndSeq[i] = fnHostPlan.tokenSeqRangePlan.tokenTileEndSeq[i];
                 }
             } else if (qslData != nullptr && tiling->tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
                 OP_LOGD(context,
                         "FnTokenSeqRanges disabled: tokenBlockCnt[%ld] exceeds fixed tiling capacity[%ld].",
                         tiling->tokenBlockCnt, MAX_FN_TOKEN_SEQ_RANGE_COUNT);
             }
         }
         OP_LOGD(context,
                 "FnHostPlan(case=%s): inputMode[%ld], dim[%ld], cuSeqlen[%ld], baseDim[%ld], baseDimCnt[%ld], "
                 "tokenCoreBudget[%ld], tokenBlockSize[%ld], tokenBlockCnt[%ld], tokenBlocksPerCore[%ld], "
                 "tokenCoreTailCnt[%ld], explicitSeqRanges[%ld], baseGrid[%ld], phase1Grid[%ld], mappedBlockDim[%ld].",
                 plannerModeTag, tiling->inputMode, tiling->dim, tiling->cuSeqlen, baseDimChoice.baseDim,
                 baseDimChoice.baseDimCnt, fnHostPlan.tokenCoreMapping.tokenCoreBudget,
                 fnHostPlan.tokenBlockChoice.tokenBlockSize, fnHostPlan.tokenBlockChoice.tokenBlockCnt,
                 fnHostPlan.tokenCoreMapping.tokenBlocksPerCore, fnHostPlan.tokenCoreMapping.tokenCoreTailCnt,
                 tiling->hasExplicitTokenSeqRanges,
                 baseDimChoice.gridSize, fnHostPlan.tokenBlockChoice.gridSize, fnHostPlan.tokenCoreMapping.blockDim);
     }
 
     uint32_t blockDim =
         (effectiveGridSize < static_cast<int64_t>(coreNum)) ? static_cast<uint32_t>(effectiveGridSize) : coreNum;
     if (isFn) {
         const int64_t mappedBlockDim = (effectiveGridSize < fnHostPlan.tokenCoreMapping.blockDim) ? effectiveGridSize : fnHostPlan.tokenCoreMapping.blockDim;
         OP_CHECK_IF(mappedBlockDim <= 0, OP_LOGE(context, "invalid mapped blockDim for runMode=0"),
                     return ge::GRAPH_FAILED);
         blockDim = static_cast<uint32_t>(mappedBlockDim);
     }
 
     OP_LOGD(context,
             "Tiling result: mode[%s], batch[%ld], dim[%ld], baseDim[%ld], baseDimCnt[%ld], gridSize[%ld], "
             "effectiveGrid[%ld], blockDim[%u], coreNum[%u], tokenTiling[%ld,%ld], hasActivation[%d], hasBias[%d], "
             "fnPlan[%ld].",
             plannerModeTag, batch, dim, baseDimChoice.baseDim, baseDimChoice.baseDimCnt, baseDimChoice.gridSize,
             effectiveGridSize, blockDim, coreNum, tiling->tokenBlockSize, tiling->tokenBlockCnt,
             static_cast<int32_t>(hasActivation), static_cast<int32_t>(hasBias), static_cast<int64_t>(fnExecutionPlan));
 
     context->SetBlockDim(blockDim);
     tiling->baseDim = baseDimChoice.baseDim;
     tiling->baseDimCnt = baseDimChoice.baseDimCnt;
     const uint32_t fnPlanKey = NormalizeFnPlanTilingKey(runModeKey, fnExecutionPlan);
     const uint32_t widthKey = NormalizeWidthTilingKey(runModeKey, static_cast<int32_t>(tiling->width));
     if (isFn && tiling->hasInitialStateMode != 0) {
         constexpr int64_t kDtypeSize = 2;
         constexpr int64_t kSyncBytesPerBlock = 32;
         const int64_t historyCount = (tiling->width - 1 > 0) ? tiling->width - 1 : 0;
         const int64_t syncWorkspaceSize = static_cast<int64_t>(blockDim) * kSyncBytesPerBlock;
         const int64_t snapshotWorkspaceSize = tiling->batch * historyCount * tiling->dim * kDtypeSize;
         const int64_t workspaceSize =
             ASCENDC_RESERVED_WORKSPACE_SIZE + syncWorkspaceSize + snapshotWorkspaceSize;
         OP_CHECK_IF(SetWorkspaceSize(context, static_cast<size_t>(workspaceSize)) != ge::GRAPH_SUCCESS,
                     OP_LOGE(context, "SetWorkspaceSize error"), return ge::GRAPH_FAILED);
         OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                     OP_LOGE(context, "SetScheduleMode(1) error"), return ge::GRAPH_FAILED);
         tiling->hasInitStateWorkspace = 1;
     } else {
         OP_CHECK_IF(SetWorkspaceSize(context, 0) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspaceSize error"),
                     return ge::GRAPH_FAILED);
         tiling->hasInitStateWorkspace = 0;
     }
 
     const uint64_t tilingKey = GET_TPL_TILING_KEY(runModeKey, widthKey, fnPlanKey);
     context->SetTilingKey(tilingKey);
     return ge::GRAPH_SUCCESS;
 }
 
 static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext *context)
 {
     OP_LOGD(context, "Enter TilingParseForCausalConv1d.");
     return ge::GRAPH_SUCCESS;
 }
 
 IMPL_OP_OPTILING(CausalConv1d)
     .Tiling(CausalConv1dTilingFunc)
     .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);
 
 }
 

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
 * \file math_util.h
 * \brief
 */

#ifndef TILING_MATMUL_MATH_UTIL_H
#define TILING_MATMUL_MATH_UTIL_H

#include <array>
#include <cstdint>
#include <vector>
#include <utility>
namespace matmul_tiling {
class MathUtil {
public:
    static bool IsEqual(float leftValue, float rightValue);
    template<typename T>
    static auto CeilDivision(T num1, T num2) -> T
    {
        if (num2 == 0) {
            return 0;
        }
        return static_cast<T>((static_cast<int64_t>(num1) + static_cast<int64_t>(num2) - 1) /
            static_cast<int64_t>(num2));
    }
    template<typename T>
    static auto Align(T num1, T num2) -> T
    {
        return CeilDivision(num1, num2) * num2;
    }
    static int32_t AlignDown(int32_t num1, int32_t num2);
    static bool CheckMulOverflow(int32_t a, int32_t b, int32_t &c);
    static int32_t MapShape(int32_t shape, bool roundUpFlag = true);
    static void AddFactor(std::vector<int32_t> &dimsFactors, int32_t dim);
    static void GetFactorCnt(const int32_t shape, int32_t &factorCnt, const int32_t factorStart,
        const int32_t factorEnd);
    static void GetFactorLayerCnt(const int32_t shape, int32_t &factorCnt, const int32_t factorStart,
        const int32_t factorEnd);
    static bool CheckFactorNumSatisfy(const int32_t dim);
    static int32_t FindBestSingleCore(const int32_t oriShape, const int32_t mappedShape, const int32_t coreNum,
        bool isKDim);
    static void GetFactors(std::vector<int32_t> &factorList, int32_t srcNum, int32_t minFactor, int32_t maxFactor);
    static void GetFactors(std::vector<int32_t> &factorList, int32_t srcNum, int32_t maxFactor);
    static void GetBlockFactors(std::vector<int32_t> &factorList, const int32_t oriShape, const int32_t mpShape,
        const int32_t coreNum, const int32_t maxNum);
    static int32_t GetNonFactorMap(std::vector<int32_t> &factorList, int32_t srcNum, int32_t maxFactor);
    static std::vector<std::pair<int, int>> GetFactorPairs(int32_t num);
    static std::pair<int32_t, int32_t> DivideIntoMainAndTail(int32_t num, int32_t divisor);
};
} // namespace matmul_tiling
#endif // _MATH_UTIL_H_


add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        causal_conv1d_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME CausalConv1d
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE causal_conv1d ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe tiling function implementation file
 * Create: 2025-07-22
 * Note:
 * History: 2025-07-13 create FusedDeepMoe infer function file
 */

#if defined(__DAV_C310__)
#include <cstdint>
#include "../ops_log.h"
#include "../ops_error.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace ge {
constexpr const char *OPS_UTILS_LOG_SUB_MOD_NAME = "FUSED_DEEP_MOE";
constexpr const char *OPS_UTILS_LOG_PACKAGE_TYPE = "CAM_OPS";
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_SHARE_OUTPUT_INDEX = 1;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS = 2;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 4;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 5;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_TWO = 2;

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const char *nodeName = context->GetNodeName();
    const gert::Shape *expandXShape = context->GetInputShape(EXPAND_X_INDEX);
    const gert::Shape *expertIdsShape = context->GetInputShape(EXPERT_IDS_INDEX);
    gert::Shape *expandXOutShape = context->GetOutputShape(OUTPUT_X_INDEX);
    gert::Shape *shareOutputShape = context->GetOutputShape(OUTPUT_SHARE_OUTPUT_INDEX);
    gert::Shape *expertTokenNumsShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS);
    if (expandXShape == nullptr || expertIdsShape == nullptr || expandXOutShape == nullptr ||
        shareOutputShape == nullptr || expertTokenNumsShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (expandXShape->GetDimNum() < DIM_TWO || expertIdsShape->GetDimNum() < DIM_ONE) {
        return GRAPH_FAILED;
    }

    int bs = expertIdsShape->GetDim(0);
    int h = expandXShape->GetDim(1);

    expandXOutShape->SetDimNum(expandXShape->GetDimNum());
    expandXOutShape->SetDim(0, bs);
    expandXOutShape->SetDim(1, h);

    shareOutputShape->SetDimNum(expandXShape->GetDimNum());
    shareOutputShape->SetDim(0, bs);
    shareOutputShape->SetDim(1, h);

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_SIZE_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    OPS_ERR_IF(moeExpertNumPtr == nullptr, OPS_LOG_E(nodeName, "moeExpertNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epRankSizePtr == nullptr, OPS_LOG_E(nodeName, "epRankSizePtr is nullptr."), return ge::GRAPH_FAILED);
    uint32_t epRankSize = static_cast<uint32_t>(*epRankSizePtr);
    uint32_t moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);

    expertTokenNumsShape->SetDim(0, moeExpertNum / epRankSize);
    expertTokenNumsShape->SetDimNum(1);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto expandXDataType = context->GetInputDataType(EXPAND_X_INDEX);
    context->SetOutputDataType(OUTPUT_X_INDEX, expandXDataType);
    context->SetOutputDataType(OUTPUT_SHARE_OUTPUT_INDEX, expandXDataType);
    context->SetOutputDataType(OUTPUT_EXPERT_TOKEN_NUMS, ge::DT_INT64);
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedDeepMoe).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ge
#endif  // defined(__DAV_C310__)

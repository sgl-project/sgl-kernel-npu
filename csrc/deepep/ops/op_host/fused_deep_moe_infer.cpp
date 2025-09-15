/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe tiling function implementation file
 * Author: Guo Ren
 * Create: 2025-07-22
 * Note:
 * History: 2025-07-13 create FusedDeepMoe infer function file
 */

#include <cstdint>
#include "error_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace ge {
    constexpr uint32_t EXPAND_X_INDEX = 0;
    constexpr uint32_t EXPERT_IDS_INDEX = 1;
    constexpr uint32_t OUTPUT_X_INDEX = 0;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* expandXShape = context->GetInputShape(EXPAND_X_INDEX);
    const gert::Shape* expertIdsShape = context->GetInputShape(EXPERT_IDS_INDEX);
    gert::Shape* expandXOutShape = context->GetOutputShape(OUTPUT_X_INDEX);
    
    if (expandXShape == nullptr || expertIdsShape == nullptr || expandXOutShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (expandXShape->GetDimNum() < 2 || expertIdsShape->GetDimNum() < 1) {
        return GRAPH_FAILED;
    }
    
    int bs = expertIdsShape->GetDim(0);
    int h = expandXShape->GetDim(1);

    expandXOutShape->SetDimNum(expandXShape->GetDimNum());
    expandXOutShape->SetDim(0, bs);
    expandXOutShape->SetDim(1, h);
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto expandXDataType = context->GetInputDataType(EXPAND_X_INDEX);
    context->SetOutputDataType(OUTPUT_X_INDEX, expandXDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedDeepMoe)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}
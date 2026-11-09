// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeSparseAttentionScore(gert::InferShapeContext *context)
{
    const auto *query = context->GetInputShape(0);
    *context->GetOutputShape(0) = *query;
    auto *lse = context->GetOutputShape(1);
    lse->SetDimNum(3);
    lse->SetDim(0, query->GetDim(0));
    lse->SetDim(1, query->GetDim(1));
    lse->SetDim(2, 1);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeSparseAttentionScore(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SglSparseAttentionScore)
    .InferShape(InferShapeSparseAttentionScore)
    .InferDataType(InferDataTypeSparseAttentionScore);
}  // namespace ops

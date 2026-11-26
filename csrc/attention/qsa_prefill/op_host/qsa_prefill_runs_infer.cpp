// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeQsaPrefillRuns(gert::InferShapeContext *context)
{
    const auto *q = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    auto *lse = context->GetOutputShape(1);
    if (!q || !out || !lse || q->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    *out = *q;
    lse->SetDimNum(2);
    lse->SetDim(0, q->GetDim(0));
    lse->SetDim(1, q->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQsaPrefillRuns(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglQsaPrefillRuns).InferShape(InferShapeQsaPrefillRuns).InferDataType(InferDataTypeQsaPrefillRuns);

}  // namespace ops

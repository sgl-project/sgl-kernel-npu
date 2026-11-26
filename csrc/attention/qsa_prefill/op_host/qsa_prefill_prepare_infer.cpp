// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeQsaPrefillPrepare(gert::InferShapeContext *context)
{
    const auto *blocks = context->GetInputShape(0);
    auto *slots = context->GetOutputShape(0);
    auto *runs = context->GetOutputShape(1);
    auto *counts = context->GetOutputShape(2);
    if (!blocks || !slots || !runs || !counts || blocks->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = blocks->GetDim(0);
    constexpr int64_t kOutWidth = 2064;
    slots->SetDimNum(2);
    slots->SetDim(0, rows);
    slots->SetDim(1, kOutWidth);
    runs->SetDimNum(2);
    runs->SetDim(0, rows);
    runs->SetDim(1, kOutWidth);
    counts->SetDimNum(1);
    counts->SetDim(0, rows * 16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQsaPrefillPrepare(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglQsaPrefillPrepare)
    .InferShape(InferShapeQsaPrefillPrepare)
    .InferDataType(InferDataTypeQsaPrefillPrepare);

}  // namespace ops

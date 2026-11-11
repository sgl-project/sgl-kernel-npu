/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // attn_out has the query shape; softmax_lse stays empty because
    // return_softmax_lse=true is rejected by the Torch adapter.
    const gert::Shape *qShape = context->GetInputShape(0);
    gert::Shape *attnOutShape = context->GetOutputShape(0);
    gert::Shape *lseShape = context->GetOutputShape(1);
    if (!qShape || !attnOutShape || !lseShape) {
        return ge::GRAPH_FAILED;
    }
    *attnOutShape = *qShape;

    lseShape->SetDimNum(1);
    lseShape->SetDim(0, 0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglSparseAttnSharedkv).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops

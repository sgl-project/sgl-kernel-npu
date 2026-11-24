// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#include "register/op_impl_registry.h"

namespace ops {

// y is the seventh input and the only output; the Kernel writes it in place.
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(6);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) {
        return ge::GRAPH_FAILED;
    }
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(6));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglSgemmvShrink).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops

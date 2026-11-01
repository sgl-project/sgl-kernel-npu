// BSD 3-Clause License
// Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

// Split from sgl-kernel-npu csrc/sampler/apply_token_bitmask/op_host/apply_token_bitmask.cpp.
// Interface metadata and computation are preserved.

#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglApplyTokenBitMask).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops

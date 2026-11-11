// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Framework adaptation of csrc/mega_chunk_gdn; PTO computation is unchanged.

#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeMegaChunkGdn(gert::InferShapeContext *context)
{
    constexpr size_t inputs[] = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        const auto *input = context->GetInputShape(inputs[i]);
        auto *output = context->GetOutputShape(i);
        if (!input || !output) return ge::GRAPH_FAILED;
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeMegaChunkGdn(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(9));
    context->SetOutputDataType(1, context->GetInputDataType(10));
    context->SetOutputDataType(2, context->GetInputDataType(11));
    context->SetOutputDataType(3, context->GetInputDataType(12));
    context->SetOutputDataType(4, context->GetInputDataType(13));
    context->SetOutputDataType(5, context->GetInputDataType(14));
    context->SetOutputDataType(6, context->GetInputDataType(15));
    context->SetOutputDataType(7, context->GetInputDataType(16));
    context->SetOutputDataType(8, context->GetInputDataType(17));
    context->SetOutputDataType(9, context->GetInputDataType(18));
    context->SetOutputDataType(10, context->GetInputDataType(19));
    context->SetOutputDataType(11, context->GetInputDataType(20));
    context->SetOutputDataType(12, context->GetInputDataType(22));
    context->SetOutputDataType(13, context->GetInputDataType(23));
    context->SetOutputDataType(14, context->GetInputDataType(24));
    context->SetOutputDataType(15, context->GetInputDataType(25));
    context->SetOutputDataType(16, context->GetInputDataType(26));
    context->SetOutputDataType(17, context->GetInputDataType(27));
    context->SetOutputDataType(18, context->GetInputDataType(28));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SglMegaChunkGdn).InferShape(InferShapeMegaChunkGdn).InferDataType(InferDataTypeMegaChunkGdn);
}  // namespace ops

// Shape/type inference does not read tensor values or depend on Torch.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeCausalConv1d(gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *state = context->GetInputShape(2);
    auto *y = context->GetOutputShape(0);
    auto *stateRef = context->GetOutputShape(1);
    if (!x || !state || !y || !stateRef) return ge::GRAPH_FAILED;
    *y = *x;
    *stateRef = *state;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeCausalConv1d(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(2));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglCausalConv1d).InferShape(InferShapeCausalConv1d).InferDataType(InferDataTypeCausalConv1d);
}  // namespace ops

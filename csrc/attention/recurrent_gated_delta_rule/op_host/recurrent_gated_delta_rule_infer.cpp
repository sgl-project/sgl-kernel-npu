// Inference uses only tensor metadata, never Torch or device tensor values.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeRecurrentGatedDeltaRule(gert::InferShapeContext *context)
{
    const auto *mix = context->GetInputShape(0);
    const auto *state = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    auto *stateRef = context->GetOutputShape(1);
    if (!mix || !state || !out || !stateRef || mix->GetDimNum() != 3 || state->GetDimNum() != 4)
        return ge::GRAPH_FAILED;
    *out = gert::Shape({mix->GetDim(0), mix->GetDim(1), state->GetDim(1), state->GetDim(2)});
    *stateRef = *state;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRecurrentGatedDeltaRule(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglRecurrentGatedDeltaRule)
    .InferShape(InferShapeRecurrentGatedDeltaRule)
    .InferDataType(InferDataTypeRecurrentGatedDeltaRule);
}  // namespace ops

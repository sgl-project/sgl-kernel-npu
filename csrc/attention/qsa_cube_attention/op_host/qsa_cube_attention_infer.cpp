#include <register/op_impl_registry.h>

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    auto query = context->GetInputShape(0);
    auto output = context->GetOutputShape(0);
    if (!query || !output) {
        return ge::GRAPH_FAILED;
    }
    *output = *query;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferType(gert::InferDataTypeContext *context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(SglQsaCubeAttention).InferShape(InferShape).InferDataType(InferType);
}  // namespace ops

/*!
 * \file dispatch_ffn_infer.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {
const size_t ATTR_GROUP = 0;
const size_t ATTR_RANK_SIZE = 1;
const size_t SUPPORT_DIM_SIZE = 2;

static ge::graphStatus InferShapeDispatchFFNCombine(gert::InferShapeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDispatchFFNCombine(gert::InferDataTypeContext *context)
{
    // auto d_type = context->GetInputDataType(0);
    // context->SetOutputDataType(0, d_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DispatchFFNCombine)
    .InferShape(InferShapeDispatchFFNCombine)
    .InferDataType(InferDataTypeDispatchFFNCombine);
}  // namespace ops

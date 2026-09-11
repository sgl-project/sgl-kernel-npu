









/*!
 * \file sparse_attn_sharedkv_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t RETURN_SOFTMAX_LSE_INDEX = 8;

ge::graphStatus InferShapeKvQuantSparseAttnSharedkv(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("KvQuantSparseAttnSharedkv", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    auto attr = context->GetAttrs();
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (returnSoftmaxLse) {
        *softmaxLseShape = *queryShape;
        auto lastDimIdx = softmaxLseShape->GetDimNum() - 1;
        softmaxLseShape->SetDim(lastDimIdx, 1);
    } else {
        softmaxLseShape->SetDimNum(1);
        softmaxLseShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvQuantSparseAttnSharedkv(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("KvQuantSparseAttnSharedkv", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(KvQuantSparseAttnSharedkv).InferShape(InferShapeKvQuantSparseAttnSharedkv).InferDataType(InferDataTypeKvQuantSparseAttnSharedkv);
} // namespace ops

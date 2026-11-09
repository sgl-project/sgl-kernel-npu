/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
// Shape inference for sparse flash attention. The output shapes mirror the
// legacy Torch host wrapper's make_outputs().

#include <cstring>

#include "register/op_impl_registry.h"

namespace ops {

namespace {

constexpr int64_t kTndDimensions = 3;
constexpr int64_t kBsndDimensions = 4;
constexpr uint32_t kLayoutQueryAttrIndex = 2;
constexpr uint32_t kLayoutKvAttrIndex = 3;
constexpr uint32_t kReturnSoftmaxLseAttrIndex = 8;

void SetEmptyShape(gert::Shape *shape)
{
    shape->SetDimNum(1);
    shape->SetDim(0, 0);
}

}  // namespace

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *queryShape = context->GetInputShape(0);
    const gert::Shape *keyShape = context->GetInputShape(1);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    gert::Shape *softmaxMaxShape = context->GetOutputShape(1);
    gert::Shape *softmaxSumShape = context->GetOutputShape(2);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (!queryShape || !keyShape || !attentionOutShape || !softmaxMaxShape || !softmaxSumShape || !attrs) {
        return ge::GRAPH_FAILED;
    }
    const char *layoutQuery = attrs->GetStr(kLayoutQueryAttrIndex);
    const char *layoutKv = attrs->GetStr(kLayoutKvAttrIndex);
    const bool *returnSoftmaxLse = attrs->GetAttrPointer<bool>(kReturnSoftmaxLseAttrIndex);
    if (!layoutQuery || !layoutKv || !returnSoftmaxLse) {
        return ge::GRAPH_FAILED;
    }
    const bool tnd = std::strcmp(layoutQuery, "TND") == 0;
    if (!tnd && std::strcmp(layoutQuery, "BSND") != 0) {
        return ge::GRAPH_FAILED;
    }

    *attentionOutShape = *queryShape;
    if (!*returnSoftmaxLse) {
        SetEmptyShape(softmaxMaxShape);
        SetEmptyShape(softmaxSumShape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t kvHeads = 0;
    if (tnd) {
        if (queryShape->GetDimNum() != kTndDimensions || keyShape->GetDimNum() < kTndDimensions) {
            return ge::GRAPH_FAILED;
        }
        kvHeads = std::strcmp(layoutKv, "PA_BSND") == 0 ? keyShape->GetDim(2) : keyShape->GetDim(1);
        if (kvHeads <= 0 || queryShape->GetDim(1) % kvHeads != 0) {
            return ge::GRAPH_FAILED;
        }
        softmaxMaxShape->SetDimNum(kTndDimensions);
        softmaxMaxShape->SetDim(0, kvHeads);
        softmaxMaxShape->SetDim(1, queryShape->GetDim(0));
        softmaxMaxShape->SetDim(2, queryShape->GetDim(1) / kvHeads);
    } else {
        if (queryShape->GetDimNum() != kBsndDimensions || keyShape->GetDimNum() < kBsndDimensions) {
            return ge::GRAPH_FAILED;
        }
        kvHeads = keyShape->GetDim(2);
        if (kvHeads <= 0 || queryShape->GetDim(2) % kvHeads != 0) {
            return ge::GRAPH_FAILED;
        }
        softmaxMaxShape->SetDimNum(kBsndDimensions);
        softmaxMaxShape->SetDim(0, queryShape->GetDim(0));
        softmaxMaxShape->SetDim(1, kvHeads);
        softmaxMaxShape->SetDim(2, queryShape->GetDim(1));
        softmaxMaxShape->SetDim(3, queryShape->GetDim(2) / kvHeads);
    }
    *softmaxSumShape = *softmaxMaxShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglSparseFlashAttention).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops

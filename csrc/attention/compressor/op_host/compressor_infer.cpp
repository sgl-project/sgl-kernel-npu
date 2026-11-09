/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include <algorithm>

namespace ops {
static ge::graphStatus InferCompressorShape(gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *weight = context->GetInputShape(5);
    const auto *attrs = context->GetAttrs();
    auto *out = context->GetOutputShape(0);
    auto *stateOut = context->GetOutputShape(1);
    const auto *state = context->GetInputShape(3);
    if (!x || !weight || !attrs || !out || !stateOut || !state || weight->GetDimNum() != 1) return ge::GRAPH_FAILED;
    const auto *ratio = attrs->GetAttrPointer<int64_t>(1);
    if (!ratio || *ratio <= 0) return ge::GRAPH_FAILED;
    if (x->GetDimNum() == 3) {
        *out = gert::Shape({x->GetDim(0), (x->GetDim(1) + *ratio - 1) / *ratio, weight->GetDim(0)});
    } else if (x->GetDimNum() == 2) {
        const auto *cu = context->GetOptionalInputShape(9);
        if (!cu || cu->GetDimNum() != 1 || cu->GetDim(0) < 1) return ge::GRAPH_FAILED;
        const int64_t tokens = x->GetDim(0);
        *out = gert::Shape({std::min(tokens, tokens / *ratio + cu->GetDim(0) - 1), weight->GetDim(0)});
    } else {
        return ge::GRAPH_FAILED;
    }
    *stateOut = *state;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferCompressorDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(3));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglCompressor).InferShape(InferCompressorShape).InferDataType(InferCompressorDataType);

}  // namespace ops

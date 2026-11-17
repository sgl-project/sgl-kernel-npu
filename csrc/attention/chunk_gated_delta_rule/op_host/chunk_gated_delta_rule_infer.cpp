/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Provenance: cann/ops-transformer attention/chunk_gated_delta_rule/ -> sgl-kernel-npu
// csrc/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule.cpp -> attention CANN copy. Original import revision is
// unknown; this copy preserves the local computation.

#include "register/op_impl_registry.h"

namespace ops {

// Output tensors preserve the value/state shapes; the optional result follows chunk_state.
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *value = context->GetInputShape(2);
    const auto *state = context->GetInputShape(4);
    auto *out = context->GetOutputShape(0);
    auto *finalState = context->GetOutputShape(1);
    if (!value || !state || !out || !finalState) return ge::GRAPH_FAILED;
    *out = *value;
    *finalState = *state;
    const auto *chunk = context->GetInputShape(7);
    auto *chunkResult = context->GetOutputShape(2);
    if (chunk) {
        if (!chunkResult) return ge::GRAPH_FAILED;
        *chunkResult = *chunk;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(2));
    context->SetOutputDataType(1, context->GetInputDataType(4));
    const auto chunkType = context->GetInputDataType(7);
    if (chunkType != ge::DT_UNDEFINED) context->SetOutputDataType(2, chunkType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglChunkGatedDeltaRule).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops

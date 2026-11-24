/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeChunkKdaFwd(gert::InferShapeContext *context)
{
    const auto *attrs = context->GetAttrs();
    if (!attrs) return ge::GRAPH_FAILED;
    const auto *batch = attrs->GetInt(0);
    const auto *seqlen = attrs->GetInt(1);
    const auto *q_heads = attrs->GetInt(2);
    const auto *v_heads = attrs->GetInt(3);
    const auto *k_dim = attrs->GetInt(4);
    const auto *v_dim = attrs->GetInt(5);
    const auto *chunk_size = attrs->GetInt(6);
    const auto *seq_num = attrs->GetInt(7);
    const auto *total_chunks = attrs->GetInt(8);
    const auto *input_rank = attrs->GetInt(9);
    if (!batch || !seqlen || !q_heads || !v_heads || !k_dim || !v_dim || !chunk_size || !seq_num || !total_chunks ||
        !input_rank)
        return ge::GRAPH_FAILED;
    if (auto *output = context->GetOutputShape(0)) *output = gert::Shape({*batch, *seqlen, *v_heads, *v_dim});
    if (auto *output = context->GetOutputShape(1)) *output = gert::Shape({*seq_num, *v_heads, *k_dim, *v_dim});
    if (auto *output = context->GetOutputShape(2)) *output = gert::Shape({*batch, *v_heads, *seqlen, *k_dim});
    if (auto *output = context->GetOutputShape(3)) *output = gert::Shape({*batch, *v_heads, *seqlen, *chunk_size});
    if (auto *output = context->GetOutputShape(4)) *output = gert::Shape({*batch, *v_heads, *seqlen, *chunk_size});
    if (auto *output = context->GetOutputShape(5)) *output = gert::Shape({*batch, *v_heads, *seqlen, *k_dim});
    if (auto *output = context->GetOutputShape(6)) *output = gert::Shape({*batch, *v_heads, *seqlen, *v_dim});
    if (auto *output = context->GetOutputShape(7)) *output = gert::Shape({*batch, *v_heads, *seqlen, *k_dim});
    if (auto *output = context->GetOutputShape(8)) *output = gert::Shape({*batch, *v_heads, *seqlen, *k_dim});
    if (auto *output = context->GetOutputShape(9)) *output = gert::Shape({*batch, *v_heads, *seqlen, *v_dim});
    if (auto *output = context->GetOutputShape(10))
        *output = gert::Shape({*batch, *v_heads, *total_chunks, *k_dim, *v_dim});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeChunkKdaFwd(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_BF16);
    context->SetOutputDataType(4, ge::DT_BF16);
    context->SetOutputDataType(5, ge::DT_BF16);
    context->SetOutputDataType(6, ge::DT_BF16);
    context->SetOutputDataType(7, ge::DT_BF16);
    context->SetOutputDataType(8, ge::DT_BF16);
    context->SetOutputDataType(9, ge::DT_BF16);
    context->SetOutputDataType(10, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SglChunkKdaFwd).InferShape(InferShapeChunkKdaFwd).InferDataType(InferDataTypeChunkKdaFwd);
}  // namespace ops

// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "register/op_impl_registry.h"

namespace ops {

namespace {
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t Q_NOPE_DIM = 512;
constexpr int64_t Q_FULL_DIM = 576;

int64_t CacheMode(gert::InferShapeContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *cacheMode = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    return cacheMode ? *cacheMode : 1;
}
}  // namespace

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // hidden_state: [N, hidden]; wuk: [headNum, 128, 512]; caches keep their shapes.
    const gert::Shape *hiddenShape = context->GetInputShape(0);
    const gert::Shape *wukShape = context->GetInputShape(12);
    const gert::Shape *kvCacheShape = context->GetInputShape(13);
    const gert::Shape *kvCacheRopeShape = context->GetInputShape(14);
    gert::Shape *qOut0 = context->GetOutputShape(0);
    gert::Shape *kvCacheOut0 = context->GetOutputShape(1);
    gert::Shape *qOut1 = context->GetOutputShape(2);
    gert::Shape *kvCacheOut1 = context->GetOutputShape(3);
    if (!hiddenShape || !wukShape || !kvCacheShape || !kvCacheRopeShape || !qOut0 || !kvCacheOut0 || !qOut1 ||
        !kvCacheOut1) {
        return ge::GRAPH_FAILED;
    }
    if (hiddenShape->GetDimNum() != 2 || wukShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = hiddenShape->GetDim(0);
    const int64_t headNum = wukShape->GetDim(0);

    qOut0->SetDimNum(3);
    qOut0->SetDim(0, tokens);
    qOut0->SetDim(1, headNum);
    qOut0->SetDim(2, CacheMode(context) == 0 ? Q_FULL_DIM : Q_NOPE_DIM);
    *kvCacheOut0 = *kvCacheShape;

    qOut1->SetDimNum(3);
    qOut1->SetDim(0, tokens);
    qOut1->SetDim(1, headNum);
    qOut1->SetDim(2, HEAD_DIM);
    *kvCacheOut1 = *kvCacheRopeShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto hidden = context->GetInputDataType(0);
    const auto kvCache = context->GetInputDataType(13);
    const auto kvCacheRope = context->GetInputDataType(14);
    const bool int8Cache = kvCache == ge::DT_INT8;
    context->SetOutputDataType(0, int8Cache ? ge::DT_INT8 : hidden);
    context->SetOutputDataType(1, kvCache);
    context->SetOutputDataType(2, hidden);
    context->SetOutputDataType(3, kvCacheRope);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SglMlaPreprocess).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;

constexpr uint32_t STATE_CACHE_INPUT_INDEX = 3;

constexpr uint32_t APE_INPUT_INDEX = 4;
constexpr uint32_t NORM_WEIGHT_INPUT_INDEX = 5;
constexpr uint32_t ROPE_SIN_INPUT_INDEX = 6;
constexpr uint32_t ROPE_COS_INPUT_INDEX = 7;

// INPUT(OPTION)
constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 8;

constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 9;
constexpr uint32_t SEQ_USED_INPUT_INDEX = 10;
constexpr uint32_t START_POS_INPUT_INDEX = 11;

// ATTR
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 0;
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
constexpr uint32_t COFF_ATTR_INDEX = 2;
constexpr uint32_t NORM_EPS_ATTR_INDEX = 3;
constexpr uint32_t ROTARY_MODE_ATTR_INDEX = 4;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 5;
constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 6;

// OUTPUT
constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;

// ATTR DEFAULT VALUE
constexpr uint32_t CMP_RATIO_VALUE = 4;
constexpr uint32_t COFF_VALUE = 1;

struct CompressorProtoShapeParam {
    bool isBsMerge{false};
    int64_t B{0};
    int64_t T{0};
    int64_t S{0};
    int64_t Sr{0};
    int64_t H{0};
    int64_t D{0};
};

// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

ge::graphStatus GetCompressorShapeDim(const gert::InferShapeContext *context, CompressorProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX);  // (B, S, H) | (T, H)
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);  // (coff * D, H)
    if (wkvShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto wgateShape = context->GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);  // (coff * D, H)
    if (wgateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto stateCacheShape = context->GetRequiredInputShape(
        STATE_CACHE_INPUT_INDEX);  // (block_num, block_size, 2 * coff * D) | (B, tokrn_size, 2 * coff * D)
    if (stateCacheShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto apeShape = context->GetRequiredInputShape(APE_INPUT_INDEX);  // (r, coff * D)
    if (apeShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto normWeightShape = context->GetRequiredInputShape(NORM_WEIGHT_INPUT_INDEX);  // (D)
    if (normWeightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ropeSinShape =
        context->GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);  // (B, ceil(S / r), rD) | (min(T, T/r + B), rD)
    if (ropeSinShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ropeCosShape =
        context->GetRequiredInputShape(ROPE_COS_INPUT_INDEX);  // (B, ceil(S / r), rD) | (min(T, T/r + B), rD)
    if (ropeCosShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto stateBlockTableShape =
        context->GetRequiredInputShape(STATE_BLOCK_TABLE_INPUT_INDEX);  // (B, sMax/block_size) | (B, )
    if (stateBlockTableShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto cuSeqlensShape = context->GetRequiredInputShape(CU_SEQ_LEN_INPUT_INDEX);  // (B+1,)
    if (cuSeqlensShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto seqUsedShape = context->GetRequiredInputShape(SEQ_USED_INPUT_INDEX);  // (B,)
    if (seqUsedShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto startPosShape = context->GetRequiredInputShape(START_POS_INPUT_INDEX);  // (B,)
    if (startPosShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (xShape->GetDimNum() == DIM_NUM_3) {  // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {  // T
        shapeParam.isBsMerge = true;
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = normWeightShape->GetDim(DIM_INDEX_0);
    shapeParam.Sr = ropeSinShape->GetDim(DIM_INDEX_1);

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorShapeDim(const CompressorProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX);  // query: (B, S, N, Hckv) | (T, N, Hckv)
    if (cmpKvShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attr = context->GetAttrs();
    const uint32_t *cmpRatioPtr = attr->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);
    uint32_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const uint32_t *coffPtr = attr->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    uint32_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;
    // Set output shape
    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3);  // (B, Sr, H)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.H);
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2);  // (T, N, Hckv)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.H);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressor(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressor(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    CompressorProtoShapeParam shapeParam{};
    auto apiRet = GetCompressorShapeDim(context, shapeParam);
    if (apiRet != GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    apiRet = SetCompressorShapeDim(shapeParam, context);
    if (apiRet != GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Compressor).InferShape(InferShapeCompressor).InferDataType(InferDataTypeCompressor);
}  // namespace ops

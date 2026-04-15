/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
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
static constexpr uint32_t BSA_LAYOUT_DIM0 = 0;
static constexpr uint32_t BSA_LAYOUT_DIM1 = 1;
static constexpr uint32_t BSA_LAYOUT_DIM2 = 2;
static constexpr uint32_t BSA_LAYOUT_DIM3 = 3;
static constexpr uint32_t BSA_QUERY_INDEX = 0;
static constexpr uint32_t BSA_VALUE_INDEX = 2;
static constexpr int32_t BSA_UNKNOWN_DIMS = -2;
static constexpr uint32_t BSA_DIM_NUMS_1 = 1;
static constexpr uint32_t BSA_LAYOUT_BNSD_BSND_DIMS = 4;
static constexpr uint32_t BSA_LAYOUT_TND_DIMS = 3;
static constexpr uint32_t BSA_LAYOUT_SH_DIMS = 2;
static constexpr uint32_t BSA_LAYOUT_BSH_DIMS = 3;
static constexpr uint32_t BSA_LAYOUT_NSD_DIMS = 3;
static constexpr uint32_t BSA_LAYOUT_BNSD_DIMS = 4;
static constexpr uint32_t BSA_LAYOUT_BSND_DIMS = 4;
static constexpr uint32_t BSA_ATTR_NUM_HEADS_INDEX = 0;
static constexpr uint32_t BSA_ATTR_NUM_KV_HEADS_INDEX = 5;
static constexpr uint32_t BSA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t BSA_ATTR_INPUT_LAYOUT_INDEX = 4;
static constexpr uint32_t BSA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 5;
static constexpr uint32_t BSA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 6;
static constexpr uint32_t BSA_QUANT_SCALE2_INDEX = 10;
} // namespace ops
namespace ops {
static ge::graphStatus InferShapeBlockSparseAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        // OPS_LOG_E("BlockSparseAttention", "Context for inferring shape is nullptr!");
        return ge::GRAPH_FAILED;
    }
    // OPS_LOG_D(context->GetNodeName(), "Enter BlockSparseAttention inferShape impl.");
    // query shape : (B, S, H)
    const gert::Shape *queryShape = context->GetInputShape(BSA_QUERY_INDEX);
    // OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)

    // value shape
    const gert::Shape *valueShape = context->GetInputShape(BSA_VALUE_INDEX);
    // OPS_LOG_E_IF_NULL(context, valueShape, return ge::GRAPH_FAILED)

    // attentionOut: (B, S, H)
    gert::Shape *attentionOutShape = context->GetOutputShape(BSA_ATTENTION_OUT_INDEX);
    // OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)

    *attentionOutShape = *queryShape;

    // UNKNOWN DIM
    if (((queryShape->GetDimNum() == BSA_DIM_NUMS_1) && (queryShape->GetDim(BSA_LAYOUT_DIM0) == BSA_UNKNOWN_DIMS)) ||
        ((valueShape->GetDimNum() == BSA_DIM_NUMS_1) && (valueShape->GetDim(BSA_LAYOUT_DIM0) == BSA_UNKNOWN_DIMS))) {
        attentionOutShape->SetDimNum(BSA_DIM_NUMS_1);
        (*attentionOutShape)[BSA_LAYOUT_DIM0] = BSA_UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    // Get attr
    auto attrs = context->GetAttrs();
    // OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)
    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(BSA_ATTR_INPUT_LAYOUT_INDEX);
    // OPS_LOG_E_IF_NULL(context, inputLayoutPtr, return ge::GRAPH_FAILED)
    const int64_t *numHeadsPtr = attrs->GetInt(BSA_ATTR_NUM_HEADS_INDEX);
    // OPS_LOG_E_IF_NULL(context, numHeadsPtr, return ge::GRAPH_FAILED)
    const int64_t *numKeyValueHeadsPtr = attrs->GetInt(BSA_ATTR_NUM_KV_HEADS_INDEX);
    // OPS_LOG_E_IF_NULL(context, numKeyValueHeadsPtr, return ge::GRAPH_FAILED)

    // KV_N除零保护, 当KV_N为零时KV_N = Q_N
    if (*numHeadsPtr == 0) {
        // OPS_LOG_E(context->GetNodeName(), "numHeads can not be 0!");
        return ge::GRAPH_FAILED;
    }
    int64_t numKeyValueHeads = (*numKeyValueHeadsPtr == 0) ? *numHeadsPtr : *numKeyValueHeadsPtr;
    // Set output shape
    if (strcmp(inputLayoutPtr, "BNSD_BSND") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_BNSD_BSND_DIMS) {
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[BSA_LAYOUT_DIM3];
        outputD = (outputD == 0 || (*queryShape)[BSA_LAYOUT_DIM3] == 0) ? (*queryShape)[BSA_LAYOUT_DIM3] : outputD;
        (*attentionOutShape)[BSA_LAYOUT_DIM0] = (*queryShape)[BSA_LAYOUT_DIM0];
        (*attentionOutShape)[BSA_LAYOUT_DIM1] = (*queryShape)[BSA_LAYOUT_DIM2];
        (*attentionOutShape)[BSA_LAYOUT_DIM2] = (*queryShape)[BSA_LAYOUT_DIM1];
        (*attentionOutShape)[BSA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "TND") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_TND_DIMS || valueShape->GetDimNum() != BSA_LAYOUT_TND_DIMS) {
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[BSA_LAYOUT_DIM2];
        outputD = (outputD == 0 || (*queryShape)[BSA_LAYOUT_DIM2] == 0) ? (*queryShape)[BSA_LAYOUT_DIM2] : outputD;
        (*attentionOutShape)[BSA_LAYOUT_DIM0] = (*queryShape)[BSA_LAYOUT_DIM0];
        (*attentionOutShape)[BSA_LAYOUT_DIM1] = (*queryShape)[BSA_LAYOUT_DIM1];
        (*attentionOutShape)[BSA_LAYOUT_DIM2] = outputD;
    } else if (strcmp(inputLayoutPtr, "NTD_TND") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_TND_DIMS || valueShape->GetDimNum() != BSA_LAYOUT_TND_DIMS) {
            return ge::GRAPH_FAILED;
        }
        (*attentionOutShape)[BSA_LAYOUT_DIM0] = (*queryShape)[BSA_LAYOUT_DIM1];
        (*attentionOutShape)[BSA_LAYOUT_DIM1] = (*queryShape)[BSA_LAYOUT_DIM0];
        (*attentionOutShape)[BSA_LAYOUT_DIM2] = (*valueShape)[BSA_LAYOUT_DIM2];
    } else if (strcmp(inputLayoutPtr, "SH") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_SH_DIMS) {
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BSH") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_BSH_DIMS) {
            return ge::GRAPH_FAILED;
        }
        if (valueShape->GetDim(BSA_LAYOUT_DIM2) != -1) {
            int64_t outputH = (*valueShape)[BSA_LAYOUT_DIM2] / numKeyValueHeads * (*numHeadsPtr);
            outputH = (outputH == 0 || (*queryShape)[BSA_LAYOUT_DIM2] == 0) ? (*queryShape)[BSA_LAYOUT_DIM2] : outputH;
            (*attentionOutShape)[BSA_LAYOUT_DIM0] = (*queryShape)[BSA_LAYOUT_DIM0];
            (*attentionOutShape)[BSA_LAYOUT_DIM1] = (*queryShape)[BSA_LAYOUT_DIM1];
            (*attentionOutShape)[BSA_LAYOUT_DIM2] = outputH;
        }
    } else if (strcmp(inputLayoutPtr, "NSD") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_NSD_DIMS) {
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BNSD") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_BNSD_DIMS) {
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[BSA_LAYOUT_DIM3];
        outputD = (outputD == 0 || (*queryShape)[BSA_LAYOUT_DIM3] == 0) ? (*queryShape)[BSA_LAYOUT_DIM3] : outputD;
        (*attentionOutShape)[BSA_LAYOUT_DIM0] = (*queryShape)[BSA_LAYOUT_DIM0];
        (*attentionOutShape)[BSA_LAYOUT_DIM1] = (*queryShape)[BSA_LAYOUT_DIM1];
        (*attentionOutShape)[BSA_LAYOUT_DIM2] = (*queryShape)[BSA_LAYOUT_DIM2];
        (*attentionOutShape)[BSA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "BSND") == 0) {
        if (queryShape->GetDimNum() != BSA_LAYOUT_BSND_DIMS) {
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[BSA_LAYOUT_DIM3];
        outputD = (outputD == 0 || (*queryShape)[BSA_LAYOUT_DIM3] == 0) ? (*queryShape)[BSA_LAYOUT_DIM3] : outputD;
        (*attentionOutShape)[BSA_LAYOUT_DIM0] = (*queryShape)[BSA_LAYOUT_DIM0];
        (*attentionOutShape)[BSA_LAYOUT_DIM1] = (*queryShape)[BSA_LAYOUT_DIM1];
        (*attentionOutShape)[BSA_LAYOUT_DIM2] = (*queryShape)[BSA_LAYOUT_DIM2];
        (*attentionOutShape)[BSA_LAYOUT_DIM3] = outputD;
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlockSparseAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // default set q's dtype as BSA's output type
    ge::DataType outputType = context->GetInputDataType(BSA_QUERY_INDEX);
    if (context->GetOptionalInputDataType(BSA_QUANT_SCALE2_INDEX) != ge::DT_UNDEFINED) { // 10 is quant_scale2's index
        outputType = ge::DT_INT8;
    } else if (outputType == ge::DT_INT8) {
        outputType = ge::DT_FLOAT16;
    }
    // attention_out, outidx:0
    context->SetOutputDataType(BSA_ATTENTION_OUT_INDEX, outputType);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockSparseAttention)
    .InferShape(InferShapeBlockSparseAttention)
    .InferDataType(InferDataTypeBlockSparseAttention)
    .InputsDataDependency({BSA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX, BSA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX});
} // namespace ops
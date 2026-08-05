/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file minimax_indexer_tiling.cpp
 * \brief
 */

#include "minimax_indexer_tiling.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace sglang::MIHost {

#define OPS_LOG_E(opName, logInfo) (std::string(opName) + ": " + logInfo)
// -------------------------- MIInfoParser Member Functions -------------------------------------
ge::graphStatus MIInfoParser::CheckRequiredInOutExistence() const
{
    TORCH_CHECK(opParamInfo_.query.shape != nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"));
    TORCH_CHECK(opParamInfo_.query.desc != nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"));
    TORCH_CHECK(opParamInfo_.key.shape != nullptr, OPS_LOG_E(opName_, "Shape of tensor key is nullptr"));
    TORCH_CHECK(opParamInfo_.key.desc != nullptr, OPS_LOG_E(opName_, "Desc of tensor key is nullptr"));
    TORCH_CHECK(opParamInfo_.weights.shape != nullptr, OPS_LOG_E(opName_, "Shape of tensor weights is nullptr"));
    TORCH_CHECK(opParamInfo_.weights.desc != nullptr, OPS_LOG_E(opName_, "Desc of tensor weights is nullptr"));
    TORCH_CHECK(opParamInfo_.attenOut.shape != nullptr, OPS_LOG_E(opName_, "Shape of tensor attenOut is nullptr"));
    TORCH_CHECK(opParamInfo_.attenOut.desc != nullptr, OPS_LOG_E(opName_, "Desc of tensor attenOut is nullptr"));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::CheckRequiredAttrExistence() const
{
    TORCH_CHECK(opParamInfo_.layOut != nullptr, OPS_LOG_E(opName_, "attr layout_query is nullptr"));
    TORCH_CHECK(opParamInfo_.layOutKey != nullptr, OPS_LOG_E(opName_, "attr layout_key is nullptr"));
    TORCH_CHECK(opParamInfo_.sparseCount != nullptr, OPS_LOG_E(opName_, "attr sparse_count is nullptr"));
    TORCH_CHECK(opParamInfo_.sparseMode != nullptr, OPS_LOG_E(opName_, "attr sparse_mode is nullptr"));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetOpName()
{
    TORCH_CHECK(context_ != nullptr, OPS_LOG_E("MinimaxIndexer", "opName got from TilingContext is nullptr"));
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetNpuInfo()
{
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    TORCH_CHECK(aivNum != 0 && aicNum != 0, OPS_LOG_E(opName_, "num of core obtained is 0"));

    socVersion_ = ascendcPlatform.GetSocVersion();
    TORCH_CHECK(socVersion_ == platform_ascendc::SocVersion::ASCEND910B ||
                    socVersion_ == platform_ascendc::SocVersion::ASCEND910_93,
                OPS_LOG_E(opName_, "soc version does not support "), (int32_t)socVersion_);

    TORCH_CHECK(context_->GetWorkspaceSizes(1) != nullptr, OPS_LOG_E(opName_, "workspaceSize got from ge is nullptr"));

    return ge::GRAPH_SUCCESS;
}

void MIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.reqToToken.tensor = context_->GetOptionalInputTensor(REQ_TO_TOKEN_INDEX);
    opParamInfo_.reqToToken.desc = context_->GetOptionalInputDesc(REQ_TO_TOKEN_INDEX);
    opParamInfo_.reqPoolIndices.tensor = context_->GetOptionalInputTensor(REQ_POOL_INDEX);
    opParamInfo_.reqPoolIndices.desc = context_->GetOptionalInputDesc(REQ_POOL_INDEX);
}

void MIInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    GetOptionalInputParaInfo();
}

void MIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(MINIMAX_INDEXER_OUTPUT);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(MINIMAX_INDEXER_OUTPUT);
}

ge::graphStatus MIInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    TORCH_CHECK(attrs != nullptr, OPS_LOG_E(context_->GetNodeName(), "attrs got from context is nullptr"));

    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);

    TORCH_CHECK((std::string(opParamInfo_.layOutKey) == "PA_BSND") ||
                    (std::string(opParamInfo_.layOut) == std::string(opParamInfo_.layOutKey)),
                OPS_LOG_E(opName_, "under non-PA conditions, layout_query and layout_key should be equal."));
    TORCH_CHECK((std::string(opParamInfo_.layOutKey) == "PA_BSND") || (std::string(opParamInfo_.layOutKey) == "BSND") ||
                    (std::string(opParamInfo_.layOutKey) == "TND"),
                OPS_LOG_E(opName_, "input attr layout_key only supported PA_BSND, BSND or TND"));

    TORCH_CHECK((std::string(opParamInfo_.layOut) == "BSND") || (std::string(opParamInfo_.layOut) == "TND"),
                OPS_LOG_E(opName_, "input attr layout_query only supported BSND or TND"));
    TORCH_CHECK(*opParamInfo_.sparseCount > 0 && *opParamInfo_.sparseCount <= SPARSE_LIMIT,
                OPS_LOG_E(opName_, "input attr sparse_count must > 0 and <= 2048."));
    TORCH_CHECK(*opParamInfo_.sparseMode == 0 || *opParamInfo_.sparseMode == SPARSE_MODE_LOWER,
                OPS_LOG_E(opName_, "input attr sparse_mode only supported 0 or 3."));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    GetAndCheckAttrParaInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_) && (inputKType_ == weightsType_);
    TORCH_CHECK(inDTypeAllEqual,
                OPS_LOG_E(opName_, "The data types of the input query, key, and weights must be the same."));
    TORCH_CHECK((inputQType_ == ge::DT_FLOAT16) || (inputQType_ == ge::DT_BF16),
                OPS_LOG_E(opName_, "The data types of the input query, key, and weights must be float16 or bfloat16."));

    TORCH_CHECK(outputType_ == ge::DT_INT32,
                OPS_LOG_E(opName_, "The data types of the output sparse_indices must be int32."));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetQueryKeyAndOutLayout()
{
    // get query/key layout base values
    const map<string, DataLayout> layoutMap = {{"BSND", DataLayout::BSND},
                                               {"TND", DataLayout::TND},
                                               {"PA_BSND", DataLayout::BnBsND}};

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second;
    }

    std::string layoutKey(opParamInfo_.layOutKey);
    auto itKey = layoutMap.find(layoutKey);
    if (itKey != layoutMap.end()) {
        kLayout_ = itKey->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::BnBsND) {
        // Fused direct mode: req_to_token + req_pool_indices substitute block_table.
        bool directMode = (opParamInfo_.reqToToken.tensor != nullptr) ||
                          (opParamInfo_.reqPoolIndices.tensor != nullptr);
        if (directMode) {
            TORCH_CHECK(opParamInfo_.reqToToken.tensor != nullptr && opParamInfo_.reqPoolIndices.tensor != nullptr,
                        OPS_LOG_E(opName_,
                                  "direct mode requires both req_to_token and req_pool_indices, block_table optional"));
        } else {
            TORCH_CHECK(opParamInfo_.blockTable.tensor != nullptr,
                        OPS_LOG_E(opName_, "key layout only supported PA_BSND, input block_table must not be null"));
        }
        TORCH_CHECK(
            opParamInfo_.actualSeqLengths.tensor != nullptr,
            OPS_LOG_E(opName_, "key layout only supported PA_BSND, input actual_seq_lengths_key must not be null"));
        TORCH_CHECK(opParamInfo_.blockTable.tensor == nullptr ||
                        opParamInfo_.blockTable.desc->GetDataType() == ge::DT_INT32,
                    OPS_LOG_E(opName_, "input block_table data type only support int32"));
    } else if (kLayout_ == DataLayout::TND) {
        TORCH_CHECK(opParamInfo_.actualSeqLengths.tensor != nullptr,
                    OPS_LOG_E(opName_, "when layout_key is TND, input actual_seq_lengths_key must not be null"));
    }

    TORCH_CHECK(opParamInfo_.actualSeqLengths.tensor == nullptr ||
                    opParamInfo_.actualSeqLengths.desc->GetDataType() == ge::DT_INT32,
                OPS_LOG_E(opName_, "input actual_seq_lengths_key data type only support int32"));

    TORCH_CHECK(opParamInfo_.actualSeqLengths.tensor == nullptr ||
                    opParamInfo_.actualSeqLengths.desc->GetDataType() == ge::DT_INT32,
                OPS_LOG_E(opName_, "input actual_seq_lengths_key data type only support int32"));

    if (qLayout_ == DataLayout::TND) {
        TORCH_CHECK(opParamInfo_.actualSeqLengthsQ.tensor != nullptr,
                    OPS_LOG_E(opName_, "when layout_query is TND, input actual_seq_lengths_query must not be null"));
    }

    TORCH_CHECK(opParamInfo_.actualSeqLengthsQ.tensor == nullptr ||
                    opParamInfo_.actualSeqLengthsQ.desc->GetDataType() == ge::DT_INT32,
                OPS_LOG_E(opName_, "input actual_seq_lengths_query data type only support int32"));

    TORCH_CHECK(kLayout_ == DataLayout::BnBsND || opParamInfo_.blockTable.tensor == nullptr,
                OPS_LOG_E(opName_, "when key layout is not PA_BSND, input block_table must be null"));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::CheckShapeDim()
{
    TORCH_CHECK(opParamInfo_.blockTable.tensor == nullptr ||
                    opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() == DIM_NUM_TWO,
                OPS_LOG_E(opName_, "the dim num of block_table's shape should be 2"));

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t qExpectShapeDim = DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        qExpectShapeDim = DIM_NUM_THREE;
    }
    if (kLayout_ == DataLayout::TND) {
        kExpectShapeDim = DIM_NUM_THREE;
    }

    TORCH_CHECK(kShapeDim == kExpectShapeDim, opName_, ": the dim num of key's shape should be ", kExpectShapeDim,
                ", but now is ", kShapeDim);

    TORCH_CHECK(qShapeDim == qExpectShapeDim, opName_, ": the dim num of query's shape should be ", qExpectShapeDim,
                ", but now is ", qShapeDim);

    TORCH_CHECK(outShapeDim == qExpectShapeDim, opName_, ": the dim num of sparse_indices's shape should be ",
                qExpectShapeDim, ", but now is ", outShapeDim);

    TORCH_CHECK(weightsShapeDim == qExpectShapeDim - 1, opName_, ": the dim num of weights's shape should be ",
                qExpectShapeDim - 1, ", but now is ", weightsShapeDim);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    TORCH_CHECK(size > 0,
                actualSeqLenName + "'s shape size should be greater than 0, instead of " + std::to_string(size));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    TORCH_CHECK(n2Size_ >= 1, opName_, ": key numhead (N2) must be >= 1, got ", n2Size_, ".");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetGSize()
{
    TORCH_CHECK(n1Size_ % n2Size_ == 0, opName_, ": input query's head_num ", n1Size_,
                " can not be a multiple of key's head_num ", n2Size_);
    gSize_ = n1Size_ / n2Size_;
    TORCH_CHECK(gSize_ >= 2 && gSize_ % 2 == 0, opName_, ": N1/N2 (gSize) must be even and >= 2 for head-split, got ", gSize_,
                " (N1=", n1Size_, ", N2=", n2Size_, ").");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetBatchSize()
{
    // get batch size base value
    // 1. non-TND: use query batch_size dimension;
    // 2. TND: actual_seq_lens_q required, its length is the B-axis size
    if ((qLayout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
    } else {  // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus MIInfoParser::GetHeadDim()
{
    // use query D dimension as base
    uint32_t dIndex = DIM_IDX_TWO;
    // determine D dimension index based on layout
    switch (qLayout_) {
        case DataLayout::TND:
            // TND format: [Total, N, D] -> D is dim 2
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND format: [Batch, SeqLen, N, D] -> D is dim 3
            dIndex = DIM_IDX_THREE;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    TORCH_CHECK(headDim_ == HEAD_DIM_LIMIT, OPS_LOG_E(opName_, "input query's last dim head_dim only support 128."));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    // OPS_LOG_I(context_->GetNodeName(), "blockSize_ is %d", blockSize_);
    TORCH_CHECK(blockSize_ % 16 == 0 && blockSize_ > 0 && blockSize_ <= 1024,
                OPS_LOG_E(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    TORCH_CHECK((blockCount_ != 0), OPS_LOG_E(opName_, "input key's block_count cannot be 0."));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    const int64_t s2SizeTemp = static_cast<int64_t>(maxBlockNumPerBatch_) * static_cast<int64_t>(blockSize_);
    if (s2SizeTemp > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = static_cast<uint32_t>(s2SizeTemp);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::GetS2Size()
{
    // get S2 base value
    // 1. BATCH_CONTINUOUS: from key S-axis
    // 3. PAGE_ATTENTION: S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::BnBsND) {
        return GetS2SizeForPageAttention();
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    } else if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MIInfoParser::ValidateInputShapesMatch()
{
    /*
    TND:
    query [T,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [T,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize],
    out [T,N2,topk]
    ----------------------
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [BatchSize,S1,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] optional
    out [BatchSize,S1,N2,topk]
    */
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        // -----------------------check BatchSize-------------------
        // bSize_ from actual_seq_lens_q
        TORCH_CHECK(opParamInfo_.actualSeqLengths.tensor != nullptr, opName_, ": actualSeqLengths tensor is null");

        TORCH_CHECK((opParamInfo_.actualSeqLengths.tensor->GetShapeSize() == bSize_) &&
                        (opParamInfo_.blockTable.tensor == nullptr ||
                         opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) == bSize_),
                    opName_,
                    ": TND case input actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 are ", bSize_,
                    ", ", opParamInfo_.actualSeqLengths.tensor->GetShapeSize(), ", ",
                    opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0), " respectively, they must be same.");

        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        TORCH_CHECK((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) == qTsize) &&
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) == qTsize),
                    opName_, ": TND case input query, weights, sparse_indices dim 0 are ", qTsize, ", ",
                    opParamInfo_.weights.shape->GetStorageShape().GetDim(0), ", ",
                    opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0), " respectively, they must be same.");
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ from query
        TORCH_CHECK((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) == bSize_) &&
                        ((opParamInfo_.blockTable.tensor == nullptr) ||
                         (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) == bSize_)) &&
                        ((opParamInfo_.actualSeqLengths.tensor == nullptr) ||
                         (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() == bSize_)) &&
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) == bSize_),
                    OPS_LOG_E(opName_,
                              "BSND case input query, weight, actual_seq_lengths_key, block_table, sparse_indices dim "
                              "0 must be same."));

        TORCH_CHECK((opParamInfo_.actualSeqLengthsQ.tensor == nullptr) ||
                        (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() == bSize_),
                    opName_, ": BSND case input query, actual_seq_lengths_query dim 0 are ", bSize_, ", ",
                    opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize(), " respectively, they must be same");

        // -----------------------check S1-------------------
        TORCH_CHECK((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) == s1Size_) &&
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) == s1Size_),
                    opName_, ": BSND case input query, weight, sparse_indices dim 1 are ", s1Size_, ", ",
                    opParamInfo_.weights.shape->GetStorageShape().GetDim(1), ", ",
                    opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1), ", they must be same.");
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }

    // -----------------------check N1-------------------
    TORCH_CHECK(opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) == n1Size_,
                OPS_LOG_E(opName_, "input query, weight shape dim N1 must be same."));

    // -----------------------check D-------------------
    uint32_t keyDDim = kLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    TORCH_CHECK(opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) == headDim_,
                OPS_LOG_E(opName_, "input query, key shape last dim must be same."));

    // -----------------------check output head dim (MiniMax: per query-head)---
    // MiniMax indexer output is [B, S1, N1, topk]: one topk block-index list per
    // (lightning summed over the GQA group and used N2=1 output; MiniMax keeps every
    // require the output's head axis to equal N1 (num query heads = 64).
    TORCH_CHECK(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) == n1Size_,
                OPS_LOG_E(opName_, "MiniMax output sparse_indices head dim must equal query head num N1."));

    // -----------------------check sparse_count-------------------
    // Fused causal-local append emits topk+1 rows (local block at slot topk).
    int64_t outLastDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1);
    TORCH_CHECK(outLastDim == *opParamInfo_.sparseCount || outLastDim == *opParamInfo_.sparseCount + 1,
                OPS_LOG_E(opName_, "output sparse_indices shape last dim must be same as attr sparse_count "
                                   "(or sparse_count+1 with append_local)."));

    return ge::GRAPH_SUCCESS;
}

void MIInfoParser::GenerateInfo(MITilingInfo &miInfo)
{
    miInfo.opName = opName_;
    miInfo.opParamInfo = opParamInfo_;
    miInfo.socVersion = socVersion_;

    miInfo.bSize = bSize_;
    miInfo.n1Size = n1Size_;
    miInfo.n2Size = n2Size_;
    miInfo.s1Size = s1Size_;
    miInfo.s2Size = s2Size_;
    miInfo.gSize = gSize_;

    miInfo.inputQType = inputQType_;
    miInfo.inputKType = inputKType_;
    miInfo.outputType = outputType_;

    miInfo.blockSize = blockSize_;
    miInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    miInfo.pageAttentionFlag = layOutKeyStr == "PA_BSND" ? true : false;
    miInfo.sparseMode = *opParamInfo_.sparseMode;
    miInfo.sparseCount = *opParamInfo_.sparseCount;

    miInfo.inputQLayout = qLayout_;
    miInfo.inputKLayout = kLayout_;
}

ge::graphStatus MIInfoParser::ParseAndCheck(MITilingInfo &miInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(miInfo);

    return ge::GRAPH_SUCCESS;
}

// -------------------------- TilingPrepare -------------------------------------
static ge::graphStatus TilingPrepareForMinimaxIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// -------------------------- MinimaxIndexerTiling -----------------------------------
ge::graphStatus MinimaxIndexerTiling::DoTiling(MITilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    // Adaptive core count: the per-head LD merge is scalar on core 0 and costs
    // O(usedCoreNum). Too many cores on short context (few blocks/core) makes the
    // merge dominate the block-parallelism win. Target ~16 blocks/core, clamped to
    // [1, blockDim], so B=1 8k (64 blocks) uses 4 cores, 32k (256) uses 16, etc.
    uint32_t estTotalBlocks = tilingInfo->bSize * tilingInfo->maxBlockNumPerBatch;
    uint32_t targetCores = (estTotalBlocks + 15) / 16;  // ceil(estTotalBlocks/16)
    if (targetCores < 1) targetCores = 1;
    if (targetCores > blockDim) targetCores = blockDim;

    // -------------set workspacesize-----------------
    // MiniMax multi-core WS: mm1Res(double-buffered scores) + per-head topk16 partials
    // (values + indices) for the LD merge. s2BaseSize == blockSize (one Cube tile/block).
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;   // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;       // double-buffered
    constexpr uint32_t M_BASE_SIZE = 512;       // M-axis base-block size (s1*g, S1=1 -> 64 active)
    const uint32_t blockSize = static_cast<uint32_t>(tilingInfo->blockSize);
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // mm1Res: [aic, 2, M_BASE, blockSize*2] float (s2BaseSize = 2*blockSize, 2 blocks/tile)
    workspaceSize += M_BASE_SIZE * blockSize * 2 * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    // vec1Res (topk values) + vec1Param (topk indices): [aic, B, G, topk]
    uint32_t partialSize = tilingInfo->bSize * tilingInfo->gSize * tilingInfo->sparseCount;
    workspaceSize += partialSize * sizeof(float) * aicNum;    // values
    workspaceSize += partialSize * sizeof(int32_t) * aicNum;  // indices
    context_->SetWorkspaceSizes(workspaceSize);

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(GE_DATATYPE_TO_KEY(tilingInfo->inputQType));
    uint32_t inputKType = static_cast<uint32_t>(GE_DATATYPE_TO_KEY(tilingInfo->inputKType));
    uint32_t outputType = static_cast<uint32_t>(GE_DATATYPE_TO_KEY(tilingInfo->outputType));
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey = (inputQType << 24) | (inputKType << 16) | (outputType << 12) | (pageAttentionFlag << 8) |
                         (inputQLayout << 4) | inputKLayout;

    // -------------set tilingdata-----------------
    MITilingData tilingData = {
        .bSize = tilingInfo->bSize,
        .n2Size = tilingInfo->n2Size,
        .gSize = tilingInfo->gSize,
        .s1Size = tilingInfo->s1Size,
        .s2Size = static_cast<uint32_t>(tilingInfo->s2Size),
        .sparseCount = tilingInfo->sparseCount,
        .usedCoreNum = targetCores,  // adaptive: ~16 blocks/core, clamped to [1, blockDim]
        .blockSize = tilingInfo->blockSize,
        .maxBlockNumPerBatch = tilingInfo->maxBlockNumPerBatch,
        .sparseMode = tilingInfo->sparseMode,
        .tilingKey = tilingKey,
    };
    // MiniMax extras (init/local/smScale) are injected by HOST_API after DoTiling.
    tilingData.initBlocks = 0U;
    tilingData.localBlocks = 0U;
    tilingData.smScaleLog2e = 0.0f;

    // TopK-API tiling for the LD merge path (ProcessLD replaces its scalar
    // running-min topk-16 with the vectorized TopK API). The merge runs when
    // numBlocks > topk; partPerHead = aicNum_*topk in the kernel, and
    // aicNum_ == GetBlockNum() == usedCoreNum (== targetCores here), so
    // partPerHead = targetCores * sparseCount. inner is 8-float (32B) aligned
    // per TopKInfo requirements. TopkTiling is computed by TopKTilingFunc and
    // flattened (SaveToBuffer) into the raw mirror carried in MITilingData;
    // topkTmpSize is the explicit-tmp UB scratch size (GetTopKMaxMinTmpSize max).
    {
        const uint32_t partPerHead = targetCores * tilingInfo->sparseCount;
        // TopK on 910B tiles the inner axis in 64-element groups; a non-÷64 tail
        // is mishandled (drops candidates in the partial tail tile, e.g. the
        // local-sentinel block). The kernel pads [partPerHead..paddedLen) with
        // -inf so TopK always runs on a ÷64 inner. paddedLen must match the
        // kernel's Align(aicNum_*topk, 64) (aicNum_ == GetBlockNum() == targetCores).
        const int32_t topkInner = static_cast<int32_t>(((partPerHead + 63U) / 64U) * 64U);
        optiling::TopkTiling topkHost;
        TopKTilingFunc(ascendcPlatform, topkInner, /*outter=*/1, static_cast<int32_t>(tilingInfo->sparseCount),
                       /*dataTypeSize=*/4U, /*isInitIndex=*/true, TopKMode::TOPK_NORMAL, /*isLargest=*/true, topkHost);
        topkHost.SaveToBuffer(&tilingData.topkTiling, sizeof(MITopkTilingRaw));
        uint32_t topkTmpMax = 0U;
        uint32_t topkTmpMin = 0U;
        GetTopKMaxMinTmpSize(ascendcPlatform, topkInner, 1, /*isReuseSource=*/false, /*isInitIndex=*/true,
                             TopKMode::TOPK_NORMAL, /*isLargest=*/true, /*dataTypeSize=*/4U, topkTmpMax, topkTmpMin);
        tilingData.topkTmpSize = topkTmpMax;
    }

    tilingData_ = tilingData;
    return ge::GRAPH_SUCCESS;
}

const MITilingData &MinimaxIndexerTiling::GetTilingData() const
{
    return tilingData_;
}
}  // namespace sglang::MIHost

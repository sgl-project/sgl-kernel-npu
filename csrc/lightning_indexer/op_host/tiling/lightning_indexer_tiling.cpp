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
 * \file lightning_indexer_tiling.cpp
 * \brief
 */

#include "lightning_indexer_tiling.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace sglang::LIHost {

#define OPS_LOG_E(opName, logInfo) (std::string(opName) + ": " + logInfo)
// --------------------------LIInfoParser类成员函数定义-------------------------------------
ge::graphStatus LIInfoParser::CheckRequiredInOutExistence() const
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

ge::graphStatus LIInfoParser::CheckRequiredAttrExistence() const
{
    TORCH_CHECK(opParamInfo_.layOut != nullptr, OPS_LOG_E(opName_, "attr layout_query is nullptr"));
    TORCH_CHECK(opParamInfo_.layOutKey != nullptr, OPS_LOG_E(opName_, "attr layout_key is nullptr"));
    TORCH_CHECK(opParamInfo_.sparseCount != nullptr, OPS_LOG_E(opName_, "attr sparse_count is nullptr"));
    TORCH_CHECK(opParamInfo_.sparseMode != nullptr, OPS_LOG_E(opName_, "attr sparse_mode is nullptr"));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetOpName()
{
    TORCH_CHECK(context_ != nullptr, OPS_LOG_E("LightningIndexer", "opName got from TilingContext is nullptr"));
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetNpuInfo()
{
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    TORCH_CHECK(aivNum != 0 && aivNum != 0, OPS_LOG_E(opName_, "num of core obtained is 0"));

    socVersion_ = ascendcPlatform.GetSocVersion();
    TORCH_CHECK(socVersion_ == platform_ascendc::SocVersion::ASCEND910B ||
                    socVersion_ == platform_ascendc::SocVersion::ASCEND910_93,
                OPS_LOG_E(opName_, "soc version does not support "), (int32_t)socVersion_);

    TORCH_CHECK(context_->GetWorkspaceSizes(1) != nullptr, OPS_LOG_E(opName_, "workspaceSize got from ge is nullptr"));

    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void LIInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    GetOptionalInputParaInfo();
}

void LIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(LIGHTNING_INDEXER);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(LIGHTNING_INDEXER);
}

ge::graphStatus LIInfoParser::GetAndCheckAttrParaInfo()
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

ge::graphStatus LIInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    GetAndCheckAttrParaInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckInOutDataType()
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

ge::graphStatus LIInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
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

ge::graphStatus LIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::BnBsND) {
        TORCH_CHECK(opParamInfo_.blockTable.tensor != nullptr,
                    OPS_LOG_E(opName_, "key layout only supported PA_BSND, input block_table must not be null"));
        TORCH_CHECK(
            opParamInfo_.actualSeqLengths.tensor != nullptr,
            OPS_LOG_E(opName_, "key layout only supported PA_BSND, input actual_seq_lengths_key must not be null"));
        TORCH_CHECK(opParamInfo_.blockTable.desc->GetDataType() == ge::DT_INT32,
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

ge::graphStatus LIInfoParser::CheckShapeDim()
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

ge::graphStatus LIInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    TORCH_CHECK(size > 0,
                actualSeqLenName + "'s shape size should be greater than 0, instead of " + std::to_string(size));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    TORCH_CHECK(n2Size_ == 1, opName_, ": key shape", n2Index, " is numhead, only support 1.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetGSize()
{
    TORCH_CHECK(n1Size_ % n2Size_ == 0, opName_, ": input query's head_num ", n1Size_,
                " can not be a multiple of key's head_num ", n2Size_);
    gSize_ = n1Size_ / n2Size_;
    TORCH_CHECK(gSize_ == 64, opName_, ": N1 is ", n1Size_, ", N2 is ", n2Size_, ", N1 divided by N2 must equal 64.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
    } else {  // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus LIInfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (qLayout_) {
        case DataLayout::TND:
            // TND格式: [Total, N, D] -> D是第2维(索引2)
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND格式: [Batch, SeqLen, N, D] -> D是第3维(索引3)
            dIndex = DIM_IDX_THREE;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    TORCH_CHECK(headDim_ == HEAD_DIM_LIMIT, OPS_LOG_E(opName_, "input query's last dim head_dim only support 128."));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    // OPS_LOG_I(context_->GetNodeName(), "blockSize_ is %d", blockSize_);
    TORCH_CHECK(blockSize_ % 16 == 0 && blockSize_ > 0 && blockSize_ <= 1024,
                OPS_LOG_E(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    TORCH_CHECK((blockCount_ != 0), OPS_LOG_E(opName_, "input key's block_count cannot be 0."));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::BnBsND) {
        return GetS2SizeForPageAttention();
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    } else if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::ValidateInputShapesMatch()
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
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于act_seq_q
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
        // bSize_ 来源于query
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

    // -----------------------check N2-------------------
    TORCH_CHECK(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) == n2Size_,
                OPS_LOG_E(opName_, "input query and output sparse_indices shape n2 dim must be same."));

    // -----------------------check sparse_count-------------------
    TORCH_CHECK(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) == *opParamInfo_.sparseCount,
                OPS_LOG_E(opName_, "output sparse_indices shape last dim must be same as attr sparse_count."));

    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GenerateInfo(LITilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;

    liInfo.bSize = bSize_;
    liInfo.n1Size = n1Size_;
    liInfo.n2Size = n2Size_;
    liInfo.s1Size = s1Size_;
    liInfo.s2Size = s2Size_;
    liInfo.gSize = gSize_;

    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.outputType = outputType_;

    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    liInfo.pageAttentionFlag = layOutKeyStr == "PA_BSND" ? true : false;
    liInfo.sparseMode = *opParamInfo_.sparseMode;
    liInfo.sparseCount = *opParamInfo_.sparseCount;

    liInfo.inputQLayout = qLayout_;
    liInfo.inputKLayout = kLayout_;
}

ge::graphStatus LIInfoParser::ParseAndCheck(LITilingInfo &liInfo)
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

    GenerateInfo(liInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForLightningIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------LightningIndexerTiling类成员函数定义-----------------------
ge::graphStatus LightningIndexerTiling::DoTiling(LITilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;          // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;              // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;              // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;             // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;           // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;           // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8;  // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;       // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;         // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;               // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;           // TopK选取个数
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    // Decode流程(LD)需要Workspace大小
    // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
    // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
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
    LITilingData tilingData = {
        .bSize = tilingInfo->bSize,
        .n2Size = tilingInfo->n2Size,
        .gSize = tilingInfo->gSize,
        .s1Size = tilingInfo->s1Size,
        .s2Size = static_cast<uint32_t>(tilingInfo->s2Size),
        .sparseCount = tilingInfo->sparseCount,
        .usedCoreNum = blockDim,
        .blockSize = tilingInfo->blockSize,
        .maxBlockNumPerBatch = tilingInfo->maxBlockNumPerBatch,
        .sparseMode = tilingInfo->sparseMode,
        .tilingKey = tilingKey,
    };

    tilingData_ = tilingData;
    return ge::GRAPH_SUCCESS;
}

const LITilingData &LightningIndexerTiling::GetTilingData() const
{
    return tilingData_;
}
}  // namespace sglang::LIHost

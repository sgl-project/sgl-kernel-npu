/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_tiling.cpp
 * \brief
 */

#include "lightning_indexer_tiling.h"
#include "../op_kernel/lightning_indexer_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {
// --------------------------LIInfoParser类成员函数定义-------------------------------------
ge::graphStatus LIInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.weights.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.weights.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.valuesOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor values output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.valuesOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor values output is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layOut == nullptr, OPS_LOG_E(opName_, "attr layout_query is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.layOutKey == nullptr, OPS_LOG_E(opName_, "attr layout_key is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.sparseCount == nullptr, OPS_LOG_E(opName_, "attr sparse_count is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.blockLen == nullptr, OPS_LOG_E(opName_, "attr block_len is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.qBlockLen == nullptr, OPS_LOG_E(opName_, "attr q_block_len is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.initNum == nullptr, OPS_LOG_E(opName_, "attr init_num is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.localNum == nullptr, OPS_LOG_E(opName_, "attr local_num is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparse_mode is nullptr"),
               return ge::GRAPH_FAILED);

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
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("LightningIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr, OPS_LOG_E(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
        OPS_LOG_E(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr, OPS_LOG_E(opName_, "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.curSeqLengthsQ.tensor = context_->GetOptionalInputTensor(CUR_SEQ_Q_INDEX);
    opParamInfo_.curSeqLengthsQ.desc = context_->GetOptionalInputDesc(CUR_SEQ_Q_INDEX);
    opParamInfo_.curSeqLengths.tensor = context_->GetOptionalInputTensor(CUR_SEQ_K_INDEX);
    opParamInfo_.curSeqLengths.desc = context_->GetOptionalInputDesc(CUR_SEQ_K_INDEX);
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
    opParamInfo_.valuesOut.desc = context_->GetOutputDesc(LIGHTNING_VALUES);
    opParamInfo_.valuesOut.shape = context_->GetOutputShape(LIGHTNING_VALUES);
}

ge::graphStatus LIInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    opParamInfo_.blockLen = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_LEN_INDEX);
    opParamInfo_.qBlockLen = attrs->GetAttrPointer<int32_t>(ATTR_Q_BLOCK_LEN_INDEX);
    opParamInfo_.initNum = attrs->GetAttrPointer<int32_t>(ATTR_INIT_NUM_INDEX);
    opParamInfo_.localNum = attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_NUM_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    opParamInfo_.returnValue = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);

    if (opParamInfo_.layOut != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.sparseCount != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "selscted count is:%d", *opParamInfo_.sparseCount);
    }
    if (opParamInfo_.blockLen != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "block len is:%d", *opParamInfo_.blockLen);
    }
    if (opParamInfo_.qBlockLen != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "q block len is:%d", *opParamInfo_.qBlockLen);
    }
    if (opParamInfo_.initNum != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "init num is:%d", *opParamInfo_.initNum);
    }
    if (opParamInfo_.localNum != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "local num is:%d", *opParamInfo_.localNum);
    }
    if (opParamInfo_.sparseMode != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "sparse mode is:%d", *opParamInfo_.sparseMode);
    }
    if (opParamInfo_.preTokens != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "pre tokens is:%d", *opParamInfo_.preTokens);
    }
    if (opParamInfo_.nextTokens != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "next tokens is:%d", *opParamInfo_.nextTokens);
    }
    if (opParamInfo_.returnValue != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "return value is:%d", *opParamInfo_.returnValue);
    }

    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");

    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND")
        && (std::string(opParamInfo_.layOut) != std::string(opParamInfo_.layOutKey))),
        OPS_LOG_E(opName_, "under non-PA conditions, layout_query and layout_key should be equal."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND") && (std::string(opParamInfo_.layOutKey) != "BSND")
        && (std::string(opParamInfo_.layOutKey) != "TND")),
        OPS_LOG_E(opName_, "input attr layout_key only supported PA_BSND, BSND or TND"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(((std::string(opParamInfo_.layOut) != "BSND") && (std::string(opParamInfo_.layOut) != "TND")),
               OPS_LOG_E(opName_, "input attr layout_query only supported BSND or TND."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(!(*opParamInfo_.blockLen == 1 || *opParamInfo_.blockLen == 2 || *opParamInfo_.blockLen == 4 || *opParamInfo_.blockLen == 8 || *opParamInfo_.blockLen == 16),
               OPS_LOG_E(opName_, "input attr block_len must in [1, 2, 4, 8, 16]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(!(*opParamInfo_.qBlockLen == 1), // TODO: 暂时关闭q block indexer
               OPS_LOG_E(opName_, "input attr q_block_len must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF((!((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= TOPK_2K)) && (*opParamInfo_.sparseCount > TOPK_8K) && (*opParamInfo_.sparseCount % 128 != 0)),
               OPS_LOG_E(opName_, "input attr sparse_count must in [1 ~ 2048], or must be divisible by 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(!(*opParamInfo_.initNum >= 0 && *opParamInfo_.initNum <= 16 && (*opParamInfo_.initNum) % (*opParamInfo_.blockLen) == 0),
               OPS_LOG_E(opName_, "input attr init_num invalid."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(!(*opParamInfo_.localNum >= 0 && *opParamInfo_.localNum <= 2048 && (*opParamInfo_.localNum) % (*opParamInfo_.blockLen) == 0),
               OPS_LOG_E(opName_, "input attr local_num invalid."), return ge::GRAPH_FAILED);
    OPS_ERR_IF((*opParamInfo_.qBlockLen > 1 && (*opParamInfo_.initNum != 0 || *opParamInfo_.localNum != 0 || *opParamInfo_.blockLen > 1)),
               OPS_LOG_E(opName_, "input attr q_block_len and kv_block_len do not support setting meanwhile."), return ge::GRAPH_FAILED);            
    OPS_ERR_IF(!((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_LOWER)),
               OPS_LOG_E(opName_, "input attr sparse_mode only supported 0 or 3."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    valuesOutType_ = opParamInfo_.valuesOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_);
    OPS_ERR_IF(!inDTypeAllEqual,
               OPS_LOG_E(opName_, "The data types of the input query, key must be the same."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
               OPS_LOG_E(opName_, "The data types of the input query, key must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF((weightsType_ != ge::DT_FLOAT),
               OPS_LOG_E(opName_, "The data types of weights must be float."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(outputType_ != ge::DT_INT32,
               OPS_LOG_E(opName_, "The data types of the output sparse_indices must be int32."),
               return ge::GRAPH_FAILED);
               
    OPS_ERR_IF(valuesOutType_ != inputQType_,
               OPS_LOG_E(opName_, "The data types of the output sparse_values must be same as intputQType."),
               return ge::GRAPH_FAILED);


    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND},
        {"TND", DataLayout::TND},
        {"PA_BSND", DataLayout::BnBsND}
    };

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
        OPS_ERR_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OPS_LOG_E(opName_, "key layout only supported PA_BSND, input block_table must not be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            opParamInfo_.curSeqLengths.tensor == nullptr,
            OPS_LOG_E(opName_, "key layout only supported PA_BSND, input cur_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        OPS_ERR_IF(opParamInfo_.curSeqLengths.tensor == nullptr,
                   OPS_LOG_E(opName_, "when layout_key is TND, input cur_seq_lengths_key must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(opParamInfo_.curSeqLengths.tensor != nullptr &&
               opParamInfo_.curSeqLengths.desc->GetDataType() != ge::DT_INT64,
                   OPS_LOG_E(opName_, "input cur_seq_lengths_key data type only support int64"),
                   return ge::GRAPH_FAILED);
    if (qLayout_ == DataLayout::TND) {
        OPS_ERR_IF(opParamInfo_.curSeqLengthsQ.tensor == nullptr,
                   OPS_LOG_E(opName_, "when layout_query is TND, input cur_seq_lengths_query must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(opParamInfo_.curSeqLengthsQ.tensor != nullptr &&
                   opParamInfo_.curSeqLengthsQ.desc->GetDataType() != ge::DT_INT64,
               OPS_LOG_E(opName_, "input cur_seq_lengths_query data type only support int64"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kLayout_ != DataLayout::BnBsND && opParamInfo_.blockTable.tensor != nullptr,
                   OPS_LOG_E(opName_, "when key layout is not PA_BSND, input block_table must be null"),
                   return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckShapeDim()
{
    OPS_ERR_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                   (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
               OPS_LOG_E(opName_, "the dim num of block_table's shape should be 2"), return ge::GRAPH_FAILED);

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t valuesOutShapeDim = opParamInfo_.valuesOut.shape->GetStorageShape().GetDimNum();
    uint32_t qExpectShapeDim = DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        qExpectShapeDim = DIM_NUM_THREE;
    }
    if (kLayout_ == DataLayout::TND) {
        kExpectShapeDim = DIM_NUM_THREE;
    }
    OPS_ERR_IF(kShapeDim != kExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of key's shape should be %u, but now is %u", kExpectShapeDim, kShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of query's shape should be %u, but now is %u",
                qExpectShapeDim, qShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(outShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of sparse_indices's shape should be %u, but now is %u",
                qExpectShapeDim, outShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(valuesOutShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of sparse_values's shape should be %u, but now is %u",
                qExpectShapeDim, valuesOutShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!(weightsShapeDim == qExpectShapeDim - 1),
               OPS_LOG_E(opName_, "the dim num of weights's shape should be %u, but now is %u", qExpectShapeDim - 1,
                weightsShapeDim),
               return ge::GRAPH_FAILED);

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
    OPS_LOG_I(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetCurSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &curSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize()) - 1;
    if (size <= 0) {
        OPS_LOG_E(opName_, "%s's shape size is %u, it should be greater than 0.", curSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OPS_LOG_I(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OPS_ERR_IF(n2Size_ != 1, OPS_LOG_E(opName_, "key shape[%u] is numhead, only support 1.", n2Index),
    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OPS_LOG_E(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    // OPS_ERR_IF(gSize_ != 64, OPS_LOG_E(opName_, "N1 is %u, N2 is %u, N1 divided by N2 must equal 64.",
    //     n1Size_, n2Size_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, cur_seq_lens_q必须传入, 以cur_seq_lens_q数组的长度 -1 为B轴大小
    if ((qLayout_ == DataLayout::TND)) {
        return GetCurSeqLenSize(bSize_, opParamInfo_.curSeqLengthsQ.tensor, "input cur_seq_lengths_query");
    } else { // BSND
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
            OPS_LOG_E(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OPS_ERR_IF(headDim_ != HEAD_DIM_LIMIT, OPS_LOG_E(opName_, "input query's last dim head_dim only support 128."),
               return ge::GRAPH_FAILED);

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
    OPS_LOG_I(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OPS_ERR_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
               OPS_LOG_E(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OPS_ERR_IF((blockCount_ == 0),
                OPS_LOG_E(opName_, "input key's block_count cannot be 0."),
                return ge::GRAPH_FAILED);
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
    OPS_LOG_I(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
              maxBlockNumPerBatch_, blockSize_, s2Size_);
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
        OPS_ERR_IF(
            (opParamInfo_.curSeqLengths.tensor->GetShapeSize() - 1 != bSize_) ||
                ((opParamInfo_.blockTable.tensor != nullptr) &&
                (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
            OPS_LOG_E(
                opName_,
                "TND case input cur_seq_lengths_query, cur_seq_lengths_key, block_table dim 0 are %u, %u, %u respectively, they must be same.",
                bSize_, opParamInfo_.curSeqLengths.tensor->GetShapeSize() - 1,
                opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)),
            return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize) ||
                       (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) != qTsize),
                   OPS_LOG_E(opName_, "TND case input query, weights, sparse_indices, sparse_values dim 0 are %u, %u, %u respectively, they must be same.",
                        qTsize, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                        opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0), opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0)),
                   return ge::GRAPH_FAILED);
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                       ((opParamInfo_.blockTable.tensor != nullptr) &&
                       (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)) ||
                       ((opParamInfo_.curSeqLengths.tensor != nullptr) &&
                        (opParamInfo_.curSeqLengths.tensor->GetShapeSize() != bSize_)) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_) ||
                       (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) != bSize_),
                   OPS_LOG_E(opName_, "BSND case input query, weight, actual_seq_lengths_key, block_table, sparse_indices, sparse_values dim 0 must be same."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((opParamInfo_.curSeqLengthsQ.tensor != nullptr) &&
                       (opParamInfo_.curSeqLengthsQ.tensor->GetShapeSize() != bSize_),
                   OPS_LOG_E(opName_, "BSND case input query, actual_seq_lengths_query dim 0 are %u, %u respectively, they must be same",
                        bSize_, opParamInfo_.curSeqLengthsQ.tensor->GetShapeSize()),
                   return ge::GRAPH_FAILED);
        // -----------------------check S1-------------------
        OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                       (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                   OPS_LOG_E(opName_, "BSND case input query, weight, sparse_indices, sparse_values dim 1 are %u, %u, %u, they must be same.",
                        s1Size_, opParamInfo_.weights.shape->GetStorageShape().GetDim(1),
                        opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)),
                   return ge::GRAPH_FAILED);
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
               OPS_LOG_E(opName_, "input query, weight shape dim N1 must be same."), return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    uint32_t keyDDim = kLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    OPS_ERR_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) != headDim_),
               OPS_LOG_E(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OPS_ERR_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_) ||
               (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
               OPS_LOG_E(opName_, "input query and output sparse_indices, sparse_values shape n2 dim must be same."),
               return ge::GRAPH_FAILED);
    // -----------------------check sparse_count-------------------
    OPS_ERR_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount) ||
               (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
               OPS_LOG_E(opName_, "output sparse_indices, sparse_values shape last dim must be same as attr sparse_count."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GenerateInfo(LITilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.platformInfo = platformInfo_;
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
    liInfo.blockLen = *opParamInfo_.blockLen;
    liInfo.qBlockLen = *opParamInfo_.qBlockLen;
    liInfo.initNum = *opParamInfo_.initNum;
    liInfo.localNum = *opParamInfo_.localNum;
    liInfo.preTokens = *opParamInfo_.preTokens;
    liInfo.nextTokens = *opParamInfo_.nextTokens;
    liInfo.returnValue = static_cast<int8_t>(*opParamInfo_.returnValue); // attr  --> bool -> int_8

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
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    // Decode流程(LD)需要Workspace大小
    // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
    // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_blockLen(tilingInfo->blockLen);
    tilingData_.set_qBlockLen(tilingInfo->qBlockLen);
    tilingData_.set_initNum(tilingInfo->initNum);
    tilingData_.set_localNum(tilingInfo->localNum);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_preTokens(tilingInfo->preTokens);
    tilingData_.set_nextTokens(tilingInfo->nextTokens);
    tilingData_.set_returnValue(tilingInfo->returnValue);

    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForLightningIndexer(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LightningIndexer", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    LITilingInfo liInfo;
    LIInfoParser LIInfoParser(context);
    if (LIInfoParser.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LightningIndexerTiling liTiling(context);
    return liTiling.DoTiling(&liInfo);
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(MlpLightningIndexer)
    .Tiling(TilingForLightningIndexer)
    .TilingParse<LICompileInfo>(TilingPrepareForLightningIndexer);

} // namespace optiling

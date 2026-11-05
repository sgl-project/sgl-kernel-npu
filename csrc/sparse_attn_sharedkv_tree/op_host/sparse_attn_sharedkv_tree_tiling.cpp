/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_tree_tiling.cpp
 * \brief
 */

#include "sparse_attn_sharedkv_tree_tiling.h"
#include "../op_kernel/sparse_attn_sharedkv_tree_template_tiling_key.h"
#include <sstream>

#include <cstdio>
#define OP_LOGE(opName, fmt, ...) do { fprintf(stderr, "[tree][%s] ", (opName)); fprintf(stderr, fmt, ##__VA_ARGS__); fprintf(stderr, "\n"); } while (0)
#define OP_LOGI(opName, fmt, ...) ((void)0)
#define OPS_REPORT_VECTOR_INNER_ERR(opName, message) do { fprintf(stderr, "[tree][%s] %s\n", (opName), (message)); } while (0)
#define OP_CHECK_IF(condition, logExpr, action) \
    do {                                        \
        if (condition) {                        \
            logExpr;                            \
            action;                             \
        }                                       \
    } while (false)

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

static const std::string QUERY_NAME = "query";
static const std::string ORI_KV_NAME = "ori_kv";
static const std::string CMP_KV_NAME = "cmp_kv";
static const std::string ORI_SPARSE_INDICES = "ori_sparse_indices";
static const std::string CMP_SPARSE_INDICES = "cmp_sparse_indices";
static const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
static const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
static const std::string SINKS_NAME = "sinks";
static const std::string METADATA_NAME = "metadata";
static const std::string ATTEN_OUT_NAME = "attn_out";
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_KV_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {CMP_KV_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_SPARSE_INDICES,             {ge::DT_INT32}},
    {CMP_SPARSE_INDICES,               {ge::DT_INT32}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_BLOCK_TABLE_NAME,         {ge::DT_INT32}},
    {CMP_BLOCK_TABLE_NAME,            {ge::DT_INT32}},
    {SINKS_NAME,                    {ge::DT_FLOAT}},
    {METADATA_NAME,                    {ge::DT_INT32}}
};

const std::map<std::string, std::vector<SASTreeLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,            {SASTreeLayout::BSND, SASTreeLayout::TND}},
    {ORI_KV_NAME,               {SASTreeLayout::PA_ND, SASTreeLayout::BSND, SASTreeLayout::TND}},
    {CMP_KV_NAME,             {SASTreeLayout::PA_ND, SASTreeLayout::BSND, SASTreeLayout::TND}},
    {ATTEN_OUT_NAME,         {SASTreeLayout::BSND, SASTreeLayout::TND}},
    {ORI_SPARSE_INDICES,         {SASTreeLayout::BSND, SASTreeLayout::TND}},
    {CMP_SPARSE_INDICES,         {SASTreeLayout::BSND, SASTreeLayout::TND}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

std::string SASTreeLayoutToSerialString(SASTreeLayout layout)
{
    switch (layout) {
        case SASTreeLayout::BSND: return "BSND";
        case SASTreeLayout::TND: return "TND";
        case SASTreeLayout::PA_ND: return "PA_ND";
        default: return "UNKNOWN";
    }
}

struct SASTreeCompileInfo {
    int64_t core_num;
};

static const std::map<SASTreeLayout, std::vector<SASTreeAxis>> SASTree_LAYOUT_AXIS_MAP = {
    {SASTreeLayout::BSND, {SASTreeAxis::B, SASTreeAxis::S, SASTreeAxis::N, SASTreeAxis::D}},
    {SASTreeLayout::TND, {SASTreeAxis::T, SASTreeAxis::N, SASTreeAxis::D}},
    {SASTreeLayout::PA_ND, {SASTreeAxis::Bn, SASTreeAxis::Bs, SASTreeAxis::N, SASTreeAxis::D}},
};

static const std::map<SASTreeLayout, size_t> SASTree_LAYOUT_DIM_MAP = {
    {SASTreeLayout::BSND, DIM_NUM_FOUR},
    {SASTreeLayout::TND, DIM_NUM_THREE},
    {SASTreeLayout::PA_ND, DIM_NUM_FOUR},
};

static std::string SASTreeDataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("sparseAttnSharedkv", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

// --------------------------SASTreeInfoParser类成员函数定义-------------------------------------
ge::graphStatus SASTreeInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.q.shape == nullptr, OP_LOGE(opName_, "Shape of tensor q is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.q.desc == nullptr, OP_LOGE(opName_, "Desc of tensor q is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr, OP_LOGE(opName_, "tensor of ori_Kv is nullptr"),
                return ge::GRAPH_FAILED);
    if (kvLayout_ == SASTreeLayout::PA_ND) {
            OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor == nullptr, OP_LOGE(opName_, "tensor of ori_block_table is nullptr"),
                return ge::GRAPH_FAILED);
    }
    if (perfMode_ == SASTreeTemplateMode::CFA_TEMPLATE_MODE){
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor == nullptr, OP_LOGE(opName_, "tensor of cmp_kv is nullptr"),
                    return ge::GRAPH_FAILED);
    }
    if (perfMode_ == SASTreeTemplateMode::SCFA_TEMPLATE_MODE){
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor == nullptr, OP_LOGE(opName_, "tensor of cmp_kv is nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor == nullptr, OP_LOGE(opName_, "cmp_sparse_indices is nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::CheckUnrequiredParaExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("SparseAttnSharedkvTree", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetNpuInfo()
{
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
        OP_LOGE(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SASTreeInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.oriKv.tensor = context_->GetOptionalInputTensor(ORI_KV_INDEX);
    opParamInfo_.oriKv.desc = context_->GetOptionalInputDesc(ORI_KV_INDEX);
    opParamInfo_.cmpKv.tensor = context_->GetOptionalInputTensor(CMP_KV_INDEX);
    opParamInfo_.cmpKv.desc = context_->GetOptionalInputDesc(CMP_KV_INDEX);
    opParamInfo_.oriSparseIndices.tensor = context_->GetOptionalInputTensor(ORI_SPARSE_INDICES_INDEX);
    opParamInfo_.oriSparseIndices.desc = context_->GetOptionalInputDesc(ORI_SPARSE_INDICES_INDEX);
    opParamInfo_.cmpSparseIndices.tensor = context_->GetOptionalInputTensor(CMP_SPARSE_INDICES_INDEX);
    opParamInfo_.cmpSparseIndices.desc = context_->GetOptionalInputDesc(CMP_SPARSE_INDICES_INDEX);
    opParamInfo_.oriBlockTable.tensor = context_->GetOptionalInputTensor(ORI_BLOCK_TABLE_INDEX);
    opParamInfo_.oriBlockTable.desc = context_->GetOptionalInputDesc(ORI_BLOCK_TABLE_INDEX);
    opParamInfo_.cmpBlockTable.tensor = context_->GetOptionalInputTensor(CMP_BLOCK_TABLE_INDEX);
    opParamInfo_.cmpBlockTable.desc = context_->GetOptionalInputDesc(CMP_BLOCK_TABLE_INDEX);
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.cuSeqLensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.seqUsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.seqUsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.cuSeqLensKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqLensKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqLensCmpKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_CMP_KV_INDEX);
    opParamInfo_.cuSeqLensCmpKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_CMP_KV_INDEX);
    opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.cmpTreeKv.tensor = context_->GetOptionalInputTensor(CMP_TREE_KV_INDEX);
    opParamInfo_.cmpTreeKv.desc = context_->GetOptionalInputDesc(CMP_TREE_KV_INDEX);
    opParamInfo_.treeAncestors.tensor = context_->GetOptionalInputTensor(TREE_ANCESTORS_INDEX);
    opParamInfo_.treeAncestors.desc = context_->GetOptionalInputDesc(TREE_ANCESTORS_INDEX);
    opParamInfo_.treePrefix.tensor = context_->GetOptionalInputTensor(TREE_PREFIX_INDEX);
    opParamInfo_.treePrefix.desc = context_->GetOptionalInputDesc(TREE_PREFIX_INDEX);

}

void SASTreeInfoParser::GetInputParaInfo()
{
    opParamInfo_.q.desc = context_->GetInputDesc(Q_INDEX);
    opParamInfo_.q.shape = context_->GetInputShape(Q_INDEX);
    GetOptionalInputParaInfo();
}

void SASTreeInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
}

ge::graphStatus SASTreeInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.oriMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_MASK_MODE_INDEX);
    opParamInfo_.cmpMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_MASK_MODE_INDEX);
    opParamInfo_.oriKvStride = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_KV_STRIDE_INDEX);
    opParamInfo_.cmpKvStride = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_KV_STRIDE_INDEX);
    opParamInfo_.oriWinLeft = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_WIN_LEFT_INDEX);
    opParamInfo_.oriWinRight = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_WIN_RIGHT_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKv = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetInOutDataType()
{
    qType_ = opParamInfo_.q.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
    if (opParamInfo_.oriKv.desc != nullptr) {
        oriKvType_ = opParamInfo_.oriKv.desc->GetDataType();
    }
    if (opParamInfo_.cmpKv.desc != nullptr) {
        cmpKvType_ = opParamInfo_.cmpKv.desc->GetDataType();
    }
    if (opParamInfo_.oriSparseIndices.desc != nullptr) {
        oriSparseIndicesType_ = opParamInfo_.oriSparseIndices.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetSASTemplateMode(SASTreeTilingInfo &sasInfo)
{
    if (opParamInfo_.oriKv.desc != nullptr) {
        if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr) {
            perfMode_ = SASTreeTemplateMode::SCFA_TEMPLATE_MODE;
        } else if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor == nullptr) {
            perfMode_ = SASTreeTemplateMode::CFA_TEMPLATE_MODE;
        } else if (opParamInfo_.cmpKv.desc == nullptr && opParamInfo_.cmpSparseIndices.tensor == nullptr) {
            perfMode_ = SASTreeTemplateMode::SWA_TEMPLATE_MODE;
        } else {
            OP_LOGE(opName_, "When cmp_sparse_indices is not nullptr, cmp_kv cannot be nullptr.");
            return ge::GRAPH_FAILED;
        }
        if (sasInfo.perfMode == SASTreeTemplateMode::CFA_TEMPLATE_MODE || sasInfo.perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE) {
            if (kvLayout_ == SASTreeLayout::TND && opParamInfo_.cuSeqLensCmpKv.tensor == nullptr) {
                OP_LOGE(opName_, "the layout_kv is %s, seqlens_cmp_kv must be provided.", SASTreeLayoutToSerialString(kvLayout_).c_str());
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(opName_, "ori_kv is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetQueryAndOutLayout()
{
    // 获取q和attnOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<SASTreeLayout, SASTreeLayout>> layoutMap = {
        {"BSND",        {SASTreeLayout::BSND, SASTreeLayout::BSND}},
        {"TND",         {SASTreeLayout::TND, SASTreeLayout::TND }},
    };
    std::string layout(opParamInfo_.layoutQ);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
        oriSparseIndicesLayout_ = qLayout_;
        cmpSparseIndicesLayout_ = qLayout_;
    } else {
        OP_LOGE(opName_, "layout of q is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (qLayout_ == SASTreeLayout::BSND){
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor != nullptr,
                    OP_LOGE(opName_, "when q's layout is BSND, cu_seqlens_q is null."),
                    return ge::GRAPH_FAILED);
    }
    if (qLayout_ == SASTreeLayout::TND){
        OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor != nullptr,
                    OP_LOGE(opName_, "when q's layout is TND, seqused_q is null."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetKvLayout()
{
    const map<string, SASTreeLayout> layoutKVMap = {
        {"PA_ND",     SASTreeLayout::PA_ND},
        {"BSND",     SASTreeLayout::BSND},
        {"TND",     SASTreeLayout::TND},
    };
    std::string layout(opParamInfo_.layoutKv);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE(opName_, "layout_kv is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================
bool SASTreeInfoParser::HasAxis(const SASTreeAxis &axis, const SASTreeLayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = SASTree_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SASTree_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SASTreeAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = SASTree_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == SASTree_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t SASTreeInfoParser::GetAxisIdx(const SASTreeAxis &axis, const SASTreeLayout &layout) const
{
    const std::vector<SASTreeAxis>& axes = SASTree_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t SASTreeInfoParser::GetAxisNum(const gert::Shape &shape, const SASTreeAxis &axis,const SASTreeLayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

uint32_t SASTreeInfoParser::GetSparseIndexWidth(const gert::Shape &shape, const SASTreeLayout &layout) const
{
    if (layout == SASTreeLayout::TND && shape.GetDimNum() == DIM_NUM_THREE) {
        return shape.GetDim(DIM_NUM_THREE - 1);
    }
    if (layout == SASTreeLayout::BSND && shape.GetDimNum() == DIM_NUM_FOUR) {
        return shape.GetDim(DIM_NUM_FOUR - 1);
    }
    return 0;
}

void SASTreeInfoParser::SetSASShape()
{
    qShape_ = opParamInfo_.q.shape->GetStorageShape();
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvShape_ = opParamInfo_.oriKv.tensor->GetStorageShape();
    } else {
        OP_LOGE(opName_, "q tensor is nullptr, please check input parameters.");
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvShape_ = opParamInfo_.cmpKv.tensor->GetStorageShape();
    }
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        oriSparseIndicesShape_ = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        hasOriSparseIndices_ = true;
        oriSparseIndexWidth_ = GetSparseIndexWidth(oriSparseIndicesShape_, oriSparseIndicesLayout_);
    }
    if (perfMode_ == SASTreeTemplateMode::SCFA_TEMPLATE_MODE)
    {
        if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
            cmpSparseIndicesShape_ = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
            uint32_t cmpSparseIndicesT = GetAxisNum(cmpSparseIndicesShape_, SASTreeAxis::T, cmpSparseIndicesLayout_);
        } else {
            OP_LOGE(opName_, "cmp_sparse_indices tensor is nullptr, please check input parameters.");
        }
    }
}

ge::graphStatus SASTreeInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(qShape_, SASTreeAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetN2Size()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(oriKvShape_, SASTreeAxis::N, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        uint32_t cmpKvN2Size_ = GetAxisNum(cmpKvShape_, SASTreeAxis::N, kvLayout_);
        if (perfMode_ == SASTreeTemplateMode::SCFA_TEMPLATE_MODE){
            uint32_t cmpSparseIndicesN2Size_ = GetAxisNum(cmpSparseIndicesShape_, SASTreeAxis::N, cmpSparseIndicesLayout_);
            OP_CHECK_IF(cmpKvN2Size_ != n2Size_ || n2Size_ != cmpSparseIndicesN2Size_,
            OP_LOGE(opName_, "N2 size check failed! Expected ori_kv's N2(%u) == cmp_sparse_indices's N2(%u).", n2Size_, cmpSparseIndicesN2Size_),
            return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(cmpKvN2Size_ != n2Size_,
                    OP_LOGE(opName_, "N2 size check failed! Expected cmp_kv's N2(%u) ==ori_kv's N2(%u).", cmpKvN2Size_, n2Size_),
                    return ge::GRAPH_FAILED);
        n2Size_ = cmpKvN2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    SASTreeLayout &layout, const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE(opName_, "when layout of q is %s, %s must be provided.",
            SASTreeLayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cuSeqLensQ");
}

ge::graphStatus SASTreeInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SASTreeLayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(qShape_, SASTreeAxis::B, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SASTreeLayout::TND) ? GetAxisNum(qShape_, SASTreeAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetKVTSize()
{
    // 获取KV的T基准值
    // 1、非TND时, 以KV的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_ori_kv和actual_seq_lens_cmp_kv必须传入, 以actual_seq_lens_ori_kv数组的长度为B轴大小(当前接口只传入oriseq，先以oriseq算出cmpseq)
    orikvTSize_ = (kvLayout_ == SASTreeLayout::TND) ? GetAxisNum(oriKvShape_, SASTreeAxis::T, kvLayout_) : 0;
    // 入参接口信息可以从GetOptionalInputParaInfo()函数中获取
    // cmpkvTSize_ = (kvLayout_ == SASTreeLayout::TND) ? GetAxisNum(cmpKvShape_, SASTreeAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SASTreeLayout::TND) {
        s1Size_ = GetAxisNum(qShape_, SASTreeAxis::T, qLayout_);
    } else { // BSND
        s1Size_ = GetAxisNum(qShape_, SASTreeAxis::S, qLayout_);
    }
    if (perfMode_ == SASTreeTemplateMode::SCFA_TEMPLATE_MODE){
        if (cmpSparseIndicesLayout_ == SASTreeLayout::TND) {
            uint32_t cmpSparseIndicesT = GetAxisNum(cmpSparseIndicesShape_, SASTreeAxis::T, cmpSparseIndicesLayout_);
            OP_CHECK_IF(cmpSparseIndicesT != s1Size_,
            OP_LOGE(opName_, "T size check failed !"),
            return ge::GRAPH_FAILED);
        } else{
            uint32_t cmpSparseIndicesS1 = GetAxisNum(cmpSparseIndicesShape_, SASTreeAxis::S, cmpSparseIndicesLayout_);
            OP_CHECK_IF(cmpSparseIndicesS1 != s1Size_,
                        OP_LOGE(opName_, "s1 size check failed !"),
                        return ge::GRAPH_FAILED);
        }
    }
    if (hasOriSparseIndices_) {
        if (oriSparseIndicesLayout_ == SASTreeLayout::TND) {
            uint32_t oriSparseIndicesT = GetAxisNum(oriSparseIndicesShape_, SASTreeAxis::T, oriSparseIndicesLayout_);
            OP_CHECK_IF(oriSparseIndicesT != s1Size_,
                        OP_LOGE(opName_, "ori_sparse_indices T size should equal query T size, but got %u and %u.",
                                 oriSparseIndicesT, s1Size_),
                        return ge::GRAPH_FAILED);
        } else {
            uint32_t oriSparseIndicesS1 = GetAxisNum(oriSparseIndicesShape_, SASTreeAxis::S, oriSparseIndicesLayout_);
            OP_CHECK_IF(oriSparseIndicesS1 != s1Size_,
                        OP_LOGE(opName_, "ori_sparse_indices S size should equal query S size, but got %u and %u.",
                                 oriSparseIndicesS1, s1Size_),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.oriBlockTable.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, block_table must be provided.", SASTreeLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t oriDimNum = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (oriDimNum != DIM_NUM_TWO) {
        OP_LOGE(opName_, "the dim num of ori_block_table is %u, it should be %u.", oriDimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1) < 0) {
        OP_LOGE(opName_, "%s's second dimension(%lld) should be non-negative number.",
            ORI_BLOCK_TABLE_NAME.c_str(), opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    oriMaxBlockNumPerBatch_ = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1);

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        uint32_t cmpDimNum = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDimNum();
        if (cmpDimNum != DIM_NUM_TWO) {
            OP_LOGE(opName_, "the dim num of cmp_block_table is %u, it should be %u.", cmpDimNum, DIM_NUM_TWO);
            return ge::GRAPH_FAILED;
        }
        if (qLayout_ == SASTreeLayout::TND) {
            if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0) != bSize_ - 1) {
                OP_LOGE(opName_, "cmp_block_table's first dimension(%u) should be equal to query's B(%u).",
                    opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1), bSize_ - 1);
                return ge::GRAPH_FAILED;
            }
        } else if (qLayout_ == SASTreeLayout::BSND) {
            if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0) != bSize_) {
                OP_LOGE(opName_, "cmp_block_table's first dimension(%u) should be equal to query's B(%u).",
                    opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1), bSize_);
                return ge::GRAPH_FAILED;
            }
        }
        if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
            OP_LOGE(opName_, "%s's second dimension(%lld) should be greater than 0",
                CMP_BLOCK_TABLE_NAME.c_str(), opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1));
            return ge::GRAPH_FAILED;
        }
        cmpMaxBlockNumPerBatch_ = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetBlockSize()
{
    oriBlockSize_ = GetAxisNum(oriKvShape_, SASTreeAxis::Bs, kvLayout_);
    cmpBlockSize_ = GetAxisNum(cmpKvShape_, SASTreeAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = oriMaxBlockNumPerBatch_ * oriBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetS2SizeForTND()
{
    if (opParamInfo_.cuSeqLensKv.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, seqlens_ori_kv must be provided.", SASTreeLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    // if (opParamInfo_.sequsedKv.tensor == nullptr) {
    //     OP_LOGE(opName_, "the layout_kv is %s, sequsedKv must be provided.", SASTreeLayoutToSerialString(kvLayout_).c_str());
    //     return ge::GRAPH_FAILED;
    // }
    // 这里返回累加和的最大值
    s2Size_ = GetAxisNum(oriKvShape_, SASTreeAxis::T, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetS2Size()
{
    // 获取S2基准值:PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    // 1、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    // 2、BSND时, S2直接获取
    if (kvLayout_ == SASTreeLayout::BSND) {
        if (opParamInfo_.oriKv.tensor != nullptr) {
            s2Size_ = GetAxisNum(oriKvShape_, SASTreeAxis::S, kvLayout_);
            return ge::GRAPH_SUCCESS;
        }
        if (opParamInfo_.cmpKv.tensor != nullptr) {
            s2Size_ = GetAxisNum(cmpKvShape_, SASTreeAxis::S, kvLayout_);
            return ge::GRAPH_SUCCESS;
        }
        return ge::GRAPH_FAILED;
    }
    return (kvLayout_ == SASTreeLayout::PA_ND) ? GetS2SizeForPageAttention() : GetS2SizeForTND();
}

ge::graphStatus SASTreeInfoParser::GetQHeadDim()
{
    qHeadDim_ = GetAxisNum(qShape_, SASTreeAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetValueHeadDim()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvHeadDim_ = GetAxisNum(oriKvShape_, SASTreeAxis::D, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvHeadDim_ = GetAxisNum(cmpKvShape_, SASTreeAxis::D, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus SASTreeInfoParser::GetSparseBlockCount()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        sparseBlockCount_ = GetAxisNum(cmpSparseIndicesShape_, SASTreeAxis::K, cmpSparseIndicesLayout_);
    }
    if (hasOriSparseIndices_) {
        oriSparseIndexWidth_ = GetSparseIndexWidth(oriSparseIndicesShape_, oriSparseIndicesLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetSinks()
{
    if (opParamInfo_.sinks.tensor == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", SINKS_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (qLayout_ == SASTreeLayout::TND) {
        if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
            if (opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_) {
                OP_LOGE(opName_, "cu_seqlens_q's dimension should be equal to %u.", bSize_);
                return ge::GRAPH_FAILED;
            }
            actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize(); // cuSeqLensQ shape is B+1
            OP_CHECK_IF(actualLenDimsQ_ == 0,
                        OP_LOGE(opName_, "cu_seqlens_q cannot be empty tensor."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_LOGE(opName_, "When layout_q is TND,  input cu_seqlens_q must be provided");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (opParamInfo_.seqUsedQ.tensor != nullptr) {
            actualLenDimsQ_ = opParamInfo_.seqUsedQ.tensor->GetShapeSize();
        }
    }
    if (kvLayout_ != SASTreeLayout::PA_ND && kvLayout_ != SASTreeLayout::BSND && kvLayout_ != SASTreeLayout::TND) {
        OP_LOGE(opName_, "ori_kv and cmp_kv only support PA_ND, BSND and TND layout.");
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ == SASTreeLayout::PA_ND) {
        if (opParamInfo_.sequsedKv.tensor != nullptr) {
            if (qLayout_ == SASTreeLayout::BSND){
                if (opParamInfo_.sequsedKv.tensor->GetShapeSize() != bSize_) {
                    OP_LOGE(opName_, "seqused_kv's dimension should be equal to %u, but got %ld.",
                        bSize_, opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (opParamInfo_.sequsedKv.tensor->GetShapeSize() != (bSize_ - 1)) {
                    OP_LOGE(opName_, "seqused_kv's dimension should be equal to %u (bSize - 1), but got %ld.",
                        (bSize_ - 1), opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            }
            OP_CHECK_IF(opParamInfo_.sequsedKv.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(opName_, "seqused_kv's dtype must be DT_INT32."),
                        return ge::GRAPH_FAILED);
            actualLenDimsKV_ = opParamInfo_.sequsedKv.tensor->GetShapeSize();
            OP_CHECK_IF(actualLenDimsKV_ == 0,
                        OP_LOGE(opName_, "seqused_kv cannot be empty tensor."),
                        return ge::GRAPH_FAILED);
        } else {
                OP_LOGE(opName_, "When kv layout is PA_ND, input sequsedKv must be provided");
                return ge::GRAPH_FAILED;
        }
    } else if (kvLayout_ == SASTreeLayout::TND) {
        if (opParamInfo_.cuSeqLensKv.tensor != nullptr) {
            if (qLayout_ == SASTreeLayout::BSND){
                if (opParamInfo_.cuSeqLensKv.tensor->GetShapeSize() != bSize_ + 1) {
                    OP_LOGE(opName_, "cuSeqLensKv's dimension should be equal to %u (bSize + 1), but got %ld.",
                        (bSize_ + 1), opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (opParamInfo_.cuSeqLensKv.tensor->GetShapeSize() != (bSize_)) {
                    OP_LOGE(opName_, "cuSeqLensKv's dimension should be equal to %u, but got %ld.",
                        bSize_, opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            }
            actualLenDimsKV_ = opParamInfo_.cuSeqLensKv.tensor->GetShapeSize();
        } else {
            OP_LOGE(opName_, "When kv layout is TND, input cuSeqLensKv must be provided");
            return ge::GRAPH_FAILED;
        }
    }
    if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.seqUsedQ.tensor->GetShapeSize();
    } else if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize(); // cuSeqLensQ shape is B+1
    }
    return ge::GRAPH_SUCCESS;
}

void SASTreeInfoParser::GenerateInfo(SASTreeTilingInfo &sasInfo)
{
    sasInfo.opName = opName_;
    sasInfo.platformInfo = platformInfo_;
    sasInfo.opParamInfo = opParamInfo_;
    sasInfo.socVersion = socVersion_;

    sasInfo.bSize = bSize_;
    sasInfo.n1Size = n1Size_;
    sasInfo.n2Size = n2Size_;
    sasInfo.s1Size = s1Size_;
    sasInfo.s2Size = s2Size_;
    sasInfo.gSize = gSize_;
    sasInfo.qHeadDim = qHeadDim_;
    sasInfo.oriKvHeadDim = oriKvHeadDim_;
    sasInfo.cmpKvHeadDim = cmpKvHeadDim_;
    sasInfo.qTSize = qTSize_;
    sasInfo.sparseBlockCount = sparseBlockCount_;
    sasInfo.oriWinLeft = oriWinLeft_;
    sasInfo.oriWinRight = oriWinRight_;
    sasInfo.qType = qType_;
    sasInfo.oriKvType = oriKvType_;
    sasInfo.cmpKvType = cmpKvType_;
    sasInfo.oriSparseIndicesType = oriSparseIndicesType_;
    sasInfo.outputType = outputType_;
    sasInfo.perfMode = perfMode_;

    if (kvLayout_ == SASTreeLayout::PA_ND) {
        sasInfo.totalBlockNum = (opParamInfo_.oriKv.tensor != nullptr) ?
            opParamInfo_.oriKv.tensor->GetStorageShape().GetDim(0) : 0;
    }
    sasInfo.sparseBlockSize = 1;
    sasInfo.hasOriSparseIndices = hasOriSparseIndices_;
    sasInfo.oriSparseIndexWidth = oriSparseIndexWidth_;
    sasInfo.oriBlockSize = oriBlockSize_;
    sasInfo.cmpBlockSize = cmpBlockSize_;
    sasInfo.blockTypeSize = sizeof(float);
    sasInfo.oriMaxBlockNumPerBatch = oriMaxBlockNumPerBatch_;
    sasInfo.cmpMaxBlockNumPerBatch = cmpMaxBlockNumPerBatch_;

    sasInfo.actualLenDimsQ = actualLenDimsQ_;
    sasInfo.actualLenDimsKV = actualLenDimsKV_;
    sasInfo.maxActualseq = maxActualseq_;
    sasInfo.actualSeqLenFlag = (opParamInfo_.sequsedKv.tensor != nullptr);
    sasInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;

    sasInfo.softmaxScale = *opParamInfo_.softmaxScale;
    sasInfo.cmpRatio = *opParamInfo_.cmpRatio;
    sasInfo.oriMaskMode = *opParamInfo_.oriMaskMode;
    sasInfo.cmpMaskMode = *opParamInfo_.cmpMaskMode;
    sasInfo.oriKvStride = *opParamInfo_.oriKvStride;
    sasInfo.cmpKvStride = *opParamInfo_.cmpKvStride;
    sasInfo.oriWinLeft = *opParamInfo_.oriWinLeft;
    sasInfo.oriWinRight = *opParamInfo_.oriWinRight;

    sasInfo.qLayout = qLayout_;
    sasInfo.oriSparseIndicesLayout = oriSparseIndicesLayout_;
    sasInfo.cmpSparseIndicesLayout = cmpSparseIndicesLayout_;
    sasInfo.kvLayout = kvLayout_;
    sasInfo.outLayout = outLayout_;
    sasInfo.returnSoftmaxLse = *opParamInfo_.returnSoftmaxLse;
}

ge::graphStatus SASTreeInfoParser::Parse(SASTreeTilingInfo &sasInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("SparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence() ||
        ge::GRAPH_SUCCESS != CheckUnrequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetSASTemplateMode(sasInfo)) {
        return ge::GRAPH_FAILED;
    }

    SetSASShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetQHeadDim() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount() ||
        ge::GRAPH_SUCCESS != GetSinks()
        ) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(sasInfo);
    return ge::GRAPH_SUCCESS;
}

void SASTreeTilingCheck::Init()
{
    opName_ = sasInfo_.opName;
    platformInfo_ = sasInfo_.platformInfo;
    opParamInfo_ = sasInfo_.opParamInfo;
    socVersion_ = sasInfo_.socVersion;
    bSize_ = sasInfo_.bSize;
    n1Size_ = sasInfo_.n1Size;
    n2Size_ = sasInfo_.n2Size;
    s1Size_ = sasInfo_.s1Size;
    s2Size_ = sasInfo_.s2Size;
    gSize_ = sasInfo_.gSize;
    qHeadDim_ = sasInfo_.qHeadDim;
    oriKvHeadDim_ = sasInfo_.oriKvHeadDim;
    cmpKvHeadDim_ = sasInfo_.cmpKvHeadDim;
    oriBlockSize_ = sasInfo_.oriBlockSize;
    cmpBlockSize_ = sasInfo_.cmpBlockSize;
    qTSize_ = sasInfo_.qTSize;
    qType_ = sasInfo_.qType;
    oriKvType_ = sasInfo_.oriKvType;
    cmpKvType_ = sasInfo_.cmpKvType;
    oriSparseIndicesType_ = sasInfo_.oriSparseIndicesType;
    outputType_ = sasInfo_.outputType;
    cmpRatio_ = sasInfo_.cmpRatio;
    qLayout_ = sasInfo_.qLayout;
    oriSparseIndicesLayout_ = sasInfo_.oriSparseIndicesLayout;
    cmpSparseIndicesLayout_ = sasInfo_.cmpSparseIndicesLayout;
    oriWinLeft_ = sasInfo_.oriWinLeft;
    oriWinRight_ = sasInfo_.oriWinRight;
    kvLayout_ = sasInfo_.kvLayout;
    outLayout_ = sasInfo_.outLayout;
    hasOriSparseIndices_ = sasInfo_.hasOriSparseIndices;
    oriSparseIndexWidth_ = sasInfo_.oriSparseIndexWidth;
}

void SASTreeTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << SASTreeDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), SASTreeDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus SASTreeTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
   if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
                    expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
                    LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void SASTreeTilingCheck::LogErrorLayoutSupport(const std::vector<SASTreeLayout> &expectLayoutList,
    const SASTreeLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << SASTreeLayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), SASTreeLayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus SASTreeTilingCheck::CheckLayoutSupport(const SASTreeLayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
                OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
void SASTreeTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void SASTreeTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus SASTreeTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckDimNumInLayoutSupport(const SASTreeLayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = SASTree_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
                SASTreeLayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
                shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaQuery() const
{
    OP_CHECK_IF(opParamInfo_.q.shape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(opName_, "q cannot be empty tensor."),
                return ge::GRAPH_FAILED);
    if (opParamInfo_.q.desc == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", QUERY_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (
        ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.q.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.q.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.q.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriKv() const
{
    const std::vector<size_t> oriKvDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (
        ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriKv.desc, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriKv.tensor->GetShape(), oriKvDimNumList, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.oriKv.tensor->GetShape(), ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaCmpKv() const
{
    if (sasInfo_.perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE ||
        sasInfo_.perfMode == SASTreeTemplateMode::CFA_TEMPLATE_MODE) {
	    const std::vector<size_t> cmpKvDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
        if (
            ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpKv.desc, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpKv.tensor->GetShape(), cmpKvDimNumList, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.cmpKv.tensor->GetShape(), CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
            }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriSparseIndices() const
{
    if (!hasOriSparseIndices_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(kvLayout_ != SASTreeLayout::PA_ND,
                OP_LOGE(opName_, "ori_sparse_indices is only supported with PA_ND kv layout."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qLayout_ != SASTreeLayout::TND,
                OP_LOGE(opName_, "ori_sparse_indices currently supports TND query layout only."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(opName_, "ori_sparse_indices cannot be empty tensor."),
                return ge::GRAPH_FAILED);

    const std::vector<size_t> oriSparseIndicesDimNumList = {DIM_NUM_THREE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriSparseIndices.desc, ORI_SPARSE_INDICES) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(oriSparseIndicesLayout_, ORI_SPARSE_INDICES) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriSparseIndices.tensor->GetShape(),
                                                oriSparseIndicesDimNumList, ORI_SPARSE_INDICES) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(oriSparseIndicesLayout_,
                                                        &opParamInfo_.oriSparseIndices.tensor->GetShape(),
                                                        ORI_SPARSE_INDICES)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t oriSparseIndicesN2Size = opParamInfo_.oriSparseIndices.tensor->GetStorageShape().GetDim(1);
    OP_CHECK_IF(oriSparseIndicesN2Size != n2Size_,
                OP_LOGE(opName_, "ori_sparse_indices N2(%u) should equal kv head num(%u).",
                         oriSparseIndicesN2Size, n2Size_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriSparseIndexWidth_ == 0 || oriSparseIndexWidth_ > SPARSE_LIMIT ||
                    (oriSparseIndexWidth_ % 128U) != 0U,
                OP_LOGE(opName_, "ori_sparse_indices K should be a positive 128-aligned value not greater than %u, got %u.",
                         SPARSE_LIMIT, oriSparseIndexWidth_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaCmpSparseIndices() const
{
    if (sasInfo_.perfMode == optiling::SASTreeTemplateMode::SCFA_TEMPLATE_MODE){
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE(opName_, "when cmp_sparse_indices is not nullptr(SCFA), cmp_sparse_indices cannot be empty tensor."),
                    return ge::GRAPH_FAILED);
        const std::vector<size_t> cmpSparseIndicesDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
        if (
            ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpSparseIndices.desc, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(cmpSparseIndicesLayout_, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpSparseIndices.tensor->GetShape(), cmpSparseIndicesDimNumList, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(cmpSparseIndicesLayout_, &opParamInfo_.cmpSparseIndices.tensor->GetShape(), CMP_SPARSE_INDICES)) {            return ge::GRAPH_FAILED;
        }
        if (cmpSparseIndicesLayout_ == SASTreeLayout::TND)
        {
            OP_CHECK_IF(!(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1) != 512 || \
                          opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1) != 1024),
                        OP_LOGE(opName_, "K should be 512 or 1024, but got: %lld ",
                        opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1)),
                        return ge::GRAPH_FAILED);
        } else{
            OP_CHECK_IF(!(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1) != 512 || \
                          opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1) != 1024),
                        OP_LOGE(opName_, "K should be 512 or 1024, but got: %lld ",
                        opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_FOUR - 1)),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriBlockTable() const
{
    if (kvLayout_ == SASTreeLayout::BSND) {
        return ge::GRAPH_SUCCESS; // BSND 场景不需要使用oriBlockTable
    }
    if(kvLayout_ == SASTreeLayout::TND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(opName_, "ori_block_table cannot be empty tensor."),
                return ge::GRAPH_FAILED);
    const std::vector<size_t> oriBlockTableDimNumList = {DIM_NUM_TWO};
    if (
        ge::GRAPH_SUCCESS != CheckDtypeSupport( opParamInfo_.oriBlockTable.desc, ORI_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriBlockTable.tensor->GetShape(), oriBlockTableDimNumList, ORI_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((oriBlockSize_ <= 0 || oriBlockSize_ > BLOCK_SIZE_LIMIT ||
                (static_cast<uint64_t>(oriBlockSize_) % 16 != 0UL)),
                OP_LOGE(opName_, "ori_block_size should be in range [1, 1024], and be aligned to 16, but got: %d.",
                oriBlockSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaCmpBlockTable() const
{
    if (kvLayout_ == SASTreeLayout::BSND) {
        return ge::GRAPH_SUCCESS; // BSND 场景不需要使用oriBlockTable
    }
    if (kvLayout_ == SASTreeLayout::TND) {
        return ge::GRAPH_SUCCESS;
    }
    if (sasInfo_.perfMode == optiling::SASTreeTemplateMode::SCFA_TEMPLATE_MODE ||
        sasInfo_.perfMode == optiling::SASTreeTemplateMode::CFA_TEMPLATE_MODE){
            const std::vector<size_t> cmpBlockTableDimNumList = {DIM_NUM_TWO};
            if (
                ge::GRAPH_SUCCESS != CheckDtypeSupport( opParamInfo_.cmpBlockTable.desc, CMP_BLOCK_TABLE_NAME) ||
                ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpBlockTable.tensor->GetShape(),
                cmpBlockTableDimNumList, CMP_BLOCK_TABLE_NAME)) {
                return ge::GRAPH_FAILED;
                }
            OP_CHECK_IF((cmpBlockSize_ <= 0 || cmpBlockSize_ > BLOCK_SIZE_LIMIT ||
                        (static_cast<uint64_t>(cmpBlockSize_) % 16 != 0UL)),
                        OP_LOGE(opName_, "cmp_block_size should be in [1, 1024], and be aligned to 16, but got: %d.",
                        cmpBlockSize_),
                        return ge::GRAPH_FAILED);
        }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaSinks() const
{
    OP_CHECK_IF(opParamInfo_.sinks.tensor->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(opName_, "sinks cannot be empty tensor."),
                return ge::GRAPH_FAILED);
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE(opName_, "the dim num of %s is %u, it should be %u.", SINKS_NAME.c_str(),
            opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum(), DIM_NUM_ONE);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0) != n1Size_) {
        OP_LOGE(opName_, "%s's dimension(%ld) should be equal to query head num(%u).", SINKS_NAME.c_str(),
            opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0), n1Size_);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.sinks.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(opName_, "sinks's dtype must be DT_FLOAT."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaMetadata() const
{
    if (opParamInfo_.metadata.tensor == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", METADATA_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT),
	            OP_LOGE(opName_, "input metadata dim 0 must be %u.", METADATA_LIMIT),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE(opName_, "metadata's dtype must be DT_INT32."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaCmpRatio() const
{
    if (sasInfo_.perfMode == optiling::SASTreeTemplateMode::CFA_TEMPLATE_MODE || sasInfo_.perfMode == optiling::SASTreeTemplateMode::SCFA_TEMPLATE_MODE){
        OP_CHECK_IF(cmpRatio_ != 128 && cmpRatio_ != 4,
                    OP_LOGE(opName_, "cmp_ratio should be 128 or 4, but got %u", cmpRatio_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriMaskMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaCmpMaskMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriKvStride() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaCmpKvStride() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriWinLeft() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSingleParaOriWinRight() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckSinglePara() const
{
    if (
        ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSinks() ||
        ge::GRAPH_SUCCESS != CheckSingleParaMetadata() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriMaskMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpMaskMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriKvStride() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKvStride() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriWinLeft() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriWinRight()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
                OP_LOGE(opName_, "%s should not be null", name.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer != nullptr,
                OP_LOGE(opName_, "%s should be null", name.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
    std::map<std::string, const void *> &notExistMap) const
{
    if (CheckExistsByMap(existMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNotExistsByMap(notExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckParaExistence() const
{
    if (kvLayout_ != SASTreeLayout::PA_ND) {
        return ge::GRAPH_SUCCESS;
    }
    std::map<std::string, const void *> ParamExistMap = {
        {"actualSeqLengths", opParamInfo_.sequsedKv.tensor},
        {"oriBlockTable", opParamInfo_.oriBlockTable.tensor},
    };
    std::map<std::string, const void *> ParamNotExistMap = {};
    if (CheckExistenceByMap(ParamExistMap, ParamNotExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckFeatureShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
                OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == SASTreeLayout::TND),
                OP_LOGE(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % 4 != 0,
                OP_LOGE(opName_, "q_head_num should be multiple of 4, but got %u", n1Size_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE(opName_, "kv_head_num should be 1, but got %u", n2Size_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
                OP_LOGE(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(gSize_ % 4 != 0,
	            OP_LOGE(opName_, "group num should be multiple of 4, but got %u", gSize_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qHeadDim_ != DIM_LIMIT,
                OP_LOGE(opName_, "q_head_dim only support %u, but got %u", DIM_LIMIT, qHeadDim_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriKvHeadDim_ != DIM_LIMIT,
                OP_LOGE(opName_, "ori_kv_head_dim only support %u, but got %u", DIM_LIMIT, oriKvHeadDim_),
                return ge::GRAPH_FAILED);
    if (!(sasInfo_.perfMode == SASTreeTemplateMode::SWA_TEMPLATE_MODE)){
        OP_CHECK_IF(cmpKvHeadDim_ != DIM_LIMIT,
                    OP_LOGE(opName_, "cmp_kv_head_dim only support %u, but got %u", DIM_LIMIT, cmpKvHeadDim_),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(!(qType_ == oriKvType_),
                OP_LOGE(opName_, "Head dimension data type check failed! qType[%s] must be the same with oriKvType[%s].",
                SASTreeDataTypeToSerialString(qType_).c_str(),
                SASTreeDataTypeToSerialString(oriKvType_).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(*opParamInfo_.oriMaskMode != 4,
                OP_LOGE(opName_, "ori_mask_mode should be 4, but got %d", *opParamInfo_.oriMaskMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.cmpMaskMode != 3,
                OP_LOGE(opName_, "cmp_mask_mode should be 3, but got %d", *opParamInfo_.cmpMaskMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinLeft_ < 0,
                OP_LOGE(opName_, "ori_win_left should be non-negative, but got %ld", oriWinLeft_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinRight_ < 0,
                OP_LOGE(opName_, "ori_win_right should be non-negative, but got %ld", oriWinRight_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutQuerySupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutQuerySupportList.begin(), layoutQuerySupportList.end(), layoutQuery) ==
                layoutQuerySupportList.end(),
                OP_LOGE(opName_, "layout_q only supports BSND/TND, but got %s", layoutQuery.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckFeatureDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_BF16 && qType_ != ge::DT_FLOAT16,
                OP_LOGE(opName_, "q dtype only support %s and %s, but got %s",
                SASTreeDataTypeToSerialString(ge::DT_BF16).c_str(), SASTreeDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
                SASTreeDataTypeToSerialString(qType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckFeaturePa() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckFeature() const
{
    // TND bSize_ includes the leading cu_seqlens_q sentinel.
    const bool hasTree = opParamInfo_.cmpTreeKv.tensor != nullptr;
    OP_CHECK_IF(hasTree != (opParamInfo_.treeAncestors.tensor != nullptr) ||
                    hasTree != (opParamInfo_.treePrefix.tensor != nullptr),
                OP_LOGE(opName_, "Tree compressed KV, ancestors and prefixes must be supplied together"),
                return ge::GRAPH_FAILED);
    if (hasTree) {
        const auto &rows = opParamInfo_.cmpTreeKv.tensor->GetShape().GetStorageShape();
        const auto &ancestors = opParamInfo_.treeAncestors.tensor->GetShape().GetStorageShape();
        const auto &prefix = opParamInfo_.treePrefix.tensor->GetShape().GetStorageShape();
        OP_CHECK_IF(rows.GetDimNum() != 2 || rows.GetDim(0) <= 0 || rows.GetDim(1) != 512 ||
                        ancestors.GetDimNum() != 2 || ancestors.GetDim(0) != qTSize_ ||
                        ancestors.GetDim(1) <= 0 || prefix.GetDimNum() != 1 || prefix.GetDim(0) != qTSize_,
                    OP_LOGE(opName_, "Tree pools require rows [nodes,512], ancestors [queries,depth], prefix [queries]"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cmpTreeKv.desc->GetDataType() != qType_ ||
                        opParamInfo_.treeAncestors.desc->GetDataType() != ge::DT_INT64 ||
                        opParamInfo_.treePrefix.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(opName_, "Invalid tree pool input dtype"), return ge::GRAPH_FAILED);
    }
    // One query record per tree node. C128 uses full CFA traversal, never Top-K.
    const bool supportedCompression =
        (sasInfo_.perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE && *opParamInfo_.cmpRatio == 4) ||
        (sasInfo_.perfMode == SASTreeTemplateMode::CFA_TEMPLATE_MODE && *opParamInfo_.cmpRatio == 128);
    OP_CHECK_IF(qLayout_ != SASTreeLayout::TND || kvLayout_ != SASTreeLayout::PA_ND ||
                    !supportedCompression || qTSize_ + 1 != bSize_,
                OP_LOGE(opName_, "Tree attention requires TND/PA_ND with C4 Top-K or full C128 CFA"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFeatureShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureDtype() ||
        ge::GRAPH_SUCCESS != CheckFeaturePa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SASTreeTilingCheck::SetSASShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.q.shape->GetStorageShape();
    oriKvShapeCmp_= opParamInfo_.oriKv.tensor->GetShape().GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attnOut.shape->GetStorageShape();
    if (sasInfo_.perfMode == SASTreeTemplateMode::CFA_TEMPLATE_MODE ||
        sasInfo_.perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpKvShapeCmp_= opParamInfo_.cmpKv.tensor->GetShape().GetStorageShape();
    }
    if (sasInfo_.perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpKvSparseIndicesCmp_ = opParamInfo_.cmpSparseIndices.tensor->GetShape().GetStorageShape();
    }
}

ge::graphStatus SASTreeTilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE(opName_, "%s dtype should be the same to %s, but it's %s.", name.c_str(),
            SASTreeDataTypeToSerialString(expectDtype).c_str(),
            SASTreeDataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckOriAndCmpKv() const
{
    OP_CHECK_IF(opParamInfo_.oriKv.tensor->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(opName_, "ori_kv cannot be empty tensor."),
                return ge::GRAPH_FAILED);
    if (sasInfo_.perfMode == SASTreeTemplateMode::CFA_TEMPLATE_MODE ||
        sasInfo_.perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE)
    {
        if (opParamInfo_.cmpKv.tensor->GetStorageShape().GetDim(0) != 0 ) {
            OP_CHECK_IF(opParamInfo_.cmpKv.tensor->GetStorageShape().GetShapeSize() == 0,
                        OP_LOGE(opName_, "cmp_kv cannot be empty tensor."),
                        return ge::GRAPH_FAILED);
        }
        if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(cmpKvType_,
            oriKvType_, CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckAttenOut() const
{
    if (opParamInfo_.attnOut.desc != nullptr && opParamInfo_.attnOut.shape != nullptr) {
        OP_CHECK_IF(opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE(opName_, "attn_out cannot be empty tensor."),
                    return ge::GRAPH_FAILED);
    } else{
        OP_LOGE(opName_, "attn_out cannot be nullptr.");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckActualSeqLensQ() const
{
    if (qLayout_ == SASTreeLayout::TND) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE(opName_, "when q's is TND, cu_seqlens_q cannot be empty tensor."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(opName_, "when q's is TND, cu_seqlens_q's dtype msut be DT_INT32."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckActualSeqLens() const
{
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SASTreeTilingCheck::CheckBlockTable() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::CheckMultiParaConsistency()
{
    SetSASShapeCompare();
    if (
        ge::GRAPH_SUCCESS != CheckOriAndCmpKv() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable())
        {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTreeTilingCheck::Process()
{
    Init();
    if (
        CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS
        )
    {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForSparseAttnSharedkvTree(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

void SparseAttnSharedkvTreeTiling::CalcUbBmm(SASTreeTilingInfo *tilingInfo)
{
    uint32_t cubeMSize = tilingInfo->gSize * tilingInfo->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
}

void SparseAttnSharedkvTreeTiling::SplitBalanced(SASTreeTilingInfo *tilingInfo)
{
    uint32_t s2Size = tilingInfo->s2Size;
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    mBaseSize_ = tilingInfo->gSize;
    headDimAlign_ = Align(tilingInfo->qHeadDim, BYTE_BLOCK);
    CalcUbBmm(tilingInfo);

    tilingData_.baseParams.set_mBaseSize(mBaseSize_);
    tilingData_.baseParams.set_s2BaseSize(sInnerSize_);
    tilingData_.baseParams.set_mmResUbSize(mmResUbSize_);
    tilingData_.baseParams.set_bmm2ResUbSize(bmm2ResUbSize_);
}

// --------------------------SparseAttnSharedkvTreeTiling类成员函数定义-----------------------
ge::graphStatus SparseAttnSharedkvTreeTiling::DoOpTiling(SASTreeTilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    blockDim_ = blockDim;
    OP_LOGI(tilingInfo->opName, "SASTree block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    SplitBalanced(tilingInfo);
    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t VEC1_RES_ELEM_SIZE = 2;        // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t VEC2_RES_ELEM_SIZE = 4;        // 4: fp32
    constexpr uint32_t PRELOAD_NUM = 2;               // preload数量

    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    workspaceSize += PRELOAD_NUM * mmResUbSize_ * MM1_RES_ELEM_SIZE * aicNum;
    workspaceSize += PRELOAD_NUM * mmResUbSize_ * VEC1_RES_ELEM_SIZE * aicNum;
    workspaceSize += PRELOAD_NUM * bmm2ResUbSize_ * MM2_RES_ELEM_SIZE * aicNum;
    workspaceSize += PRELOAD_NUM * bmm2ResUbSize_ * VEC2_RES_ELEM_SIZE * aicNum;
    if (tilingInfo->perfMode == SASTreeTemplateMode::SCFA_TEMPLATE_MODE) {
        workspaceSize += 4 * 512 * 512 * 2 * aicNum; // 4:bufNum 512:s2Size  512:D 2:sizeof(half)
        workspaceSize += 4 * 128 * 4 * (2 * aicNum); // 4:缓存有效mte2 size长度 128:份数 4:512B对齐长度 2:aiv数量
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.baseParams.set_paBlockSize(tilingInfo->blockSize);
    tilingData_.baseParams.set_oriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.baseParams.set_cmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.baseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.baseParams.set_actualLenDimsQ(tilingInfo->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(tilingInfo->actualLenDimsKV);

    tilingData_.baseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(tilingInfo->outLayout));
    tilingData_.baseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.baseParams.set_oriKvStride(tilingInfo->oriKvStride);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.baseParams.set_hasOriSparseIndices(tilingInfo->hasOriSparseIndices);
    tilingData_.baseParams.set_oriSparseIndexWidth(tilingInfo->oriSparseIndexWidth);
    tilingData_.baseParams.set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse);

    tilingData_.cmpParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);
    tilingData_.cmpParams.set_sparseBlockCount(tilingInfo->sparseBlockCount);
    tilingData_.cmpParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.cmpParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);
    tilingData_.cmpParams.set_cmpKvStride(tilingInfo->cmpKvStride);
    const auto &treeInputs = tilingInfo->opParamInfo;
    const bool hasTree = treeInputs.cmpTreeKv.tensor != nullptr;
    tilingData_.cmpParams.set_hasTreeCmpKv(hasTree);
    tilingData_.cmpParams.set_ancestorStride(hasTree ?
        treeInputs.treeAncestors.tensor->GetShape().GetStorageShape().GetDim(1) : 0);
    tilingData_.cmpParams.set_treeNodeCount(hasTree ?
        treeInputs.cmpTreeKv.tensor->GetShape().GetStorageShape().GetDim(0) : 0);
    tilingData_.cmpParams.set_cmpCacheSpan(hasTree ?
        treeInputs.cmpKv.tensor->GetShape().GetStorageShape().GetDim(0) * tilingInfo->cmpKvStride : 0);

    usedCoreNum_ = aicNum;
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);

    // -------------set tilingkey-----------------
    // FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);
    blockDim_ = usedCoreNum_;
    tilingKey_ = GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(tilingInfo->perfMode));

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus RunSparseAttnSharedkvTreeTiling(ge_helper::TilingContext &context,
                                                SparseAttnSharedkvTreeTiling &tiling)
{
    SASTreeTilingInfo sasInfo;
    SASTreeInfoParser sasInfoParser(&context);
    if (sasInfoParser.Parse(sasInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SASTreeTilingCheck sasTilingChecker(sasInfo);
    if (sasTilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tiling.DoOpTiling(&sasInfo);
}
} // namespace optiling

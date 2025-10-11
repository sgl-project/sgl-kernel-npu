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
 * \file sparse_flash_attention_tiling.cc
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_flash_attention_template_tiling_key.h"
#include "sparse_flash_attention_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {SPARSE_INDICES_NAME,       {ge::DT_INT32, ge::DT_INT32}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
};

const std::map<std::string, std::vector<SFALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {SFALayout::BSND, SFALayout::TND}},
    {KEY_NAME,               {SFALayout::BSND, SFALayout::TND, SFALayout::PA_BSND}},
    {VALUE_NAME,             {SFALayout::BSND, SFALayout::TND, SFALayout::PA_BSND}},
    {SPARSE_INDICES_NAME,  {SFALayout::BSND, SFALayout::TND}},
    {ATTEN_OUT_NAME,         {SFALayout::BSND, SFALayout::TND}},
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

struct SparseFlashAttentionCompileInfo {
    int64_t core_num;
};

static const std::map<SFALayout, std::vector<SFAAxis>> SFA_LAYOUT_AXIS_MAP = {
    {SFALayout::BSND, {SFAAxis::B, SFAAxis::S, SFAAxis::N, SFAAxis::D}},
    {SFALayout::TND, {SFAAxis::T, SFAAxis::N, SFAAxis::D}},
    {SFALayout::PA_BSND, {SFAAxis::Bn, SFAAxis::Bs, SFAAxis::N, SFAAxis::D}},
};

static std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string SFADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OPS_LOG_E("FusedInferAttentionScore", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

string SFATensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << SFAShape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << SFAShape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
               .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string SFADebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << SFATensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << SFATensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

std::string SFALayoutToSerialString(SFALayout layout)
{
    switch (layout) {
        case SFALayout::BSND: return "BSND";
        case SFALayout::TND: return "TND";
        case SFALayout::PA_BSND: return "PA_BSND";
        default: return "UNKNOWN";
    }
}

ge::graphStatus SFAMlaTiling::SetBlockDim(uint32_t blockDim)
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::SetTilingKey(uint64_t tilingKey)
{
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize)
{
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::SetTilingData(TilingDef &tilingData)
{
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::GetPlatformInfo()
{
    OPS_ERR_IF(sfaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SFAMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);

    tilingKey_ = GET_TPL_TILING_KEY(0U, layoutQuery, layoutKV, perfMode_ == SFAPerfMode::V_TEMPLATE_MODE);

    OPS_LOG_I(sfaInfo_->opName, "SFA tilingKey_: %lu.", tilingKey_);
}

void SFAMlaTiling::ZeroTensorProcess()
{
    if (sfaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        sfaInfo_->s2Size = 1024;
    }
}

void SFAMlaTiling::InitParams()
{
    if (sfaInfo_->pageAttentionFlag && sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }
    coreNum_ = aicNum_;

    headDimAlign_ = Align(sfaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void SFAMlaTiling::CalcUbBmm()
{
    uint32_t cubeMSize = sfaInfo_->gSize * sfaInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_; 
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}

void SFAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void SFAMlaTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaInfo_->qLayout != SFALayout::TND) {
        if (s2Size == 256) {   // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128; // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (s2Size > 256 && s2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2; // 2:减半
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
}

void SFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = sfaInfo_->gSize; 
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void SFAMlaTiling::Split()
{
    SplitBalanced();
}

void SFAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(sfaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(sfaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(sfaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(sfaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(sfaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(sfaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(sfaInfo_->n1Size / sfaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(sfaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(sfaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(sfaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(sfaInfo_->sparseMode);
    tilingData_.baseParams.set_needInit(sfaInfo_->needInit);
    tilingData_.baseParams.set_sparseBlockSize(sfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(sfaInfo_->sparseBlockCount);
}

// for flash decode
void SFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ * headDimAlign_);   // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ *  // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void SFAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void SFAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void SFAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t SFAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum)
{
    return actCoreNum * 2 * sfaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void SFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum); //balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaInfo_->blockTypeSize);  // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
        if (sfaInfo_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}

void SFAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void SFAMlaTiling::GetWorkspaceSize()
{
    uint32_t mmResElemSize = 4;         // 4:fp32
    uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;       // 4:fp32
    uint32_t qPreProcResElemSize = 0;   // 普通场景不涉及Q预处理
    uint32_t nUpdateElemSize = 4;   // 4:int32
    uint32_t softmaxSumElemSize = 4;   // 4:int32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    uint32_t actCoreNum = coreNum_;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
    //              bufNum  s2Base   D   dRope  sizeOf(half)
    workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum; // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
    // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
    workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数

    CalcFDWorkSpace(actCoreNum);
}

void SFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OPS_LOG_I(sfaInfo_->opName, "SFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus SFAMlaTiling::DoOpTiling(SFATilingInfo *sfaInfo)
{
    sfaInfo_ = sfaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseFlashAttention(gert::TilingContext *context)
{
    SFATilingInfo sfaInfo;
    SFAInfoParser sfaInfoParser(context);
    if (sfaInfoParser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SFATilingCheck tilingChecker(sfaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SFAMlaTiling tiling(context);
    return tiling.DoOpTiling(&sfaInfo);
}

ge::graphStatus TilingPrepareForSparseFlashAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const SFATilingShapeCompareParam &param, const SFALayout &layout) const
{
    if (layout == SFALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == SFALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == SFALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OPS_LOG_E(opName_, "layout %s is unsupported", SFALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CompareShape(SFATilingShapeCompareParam &param,
    const gert::Shape &shape, const SFALayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OPS_LOG_E(opName_,
            "%s shape.dim is %zu, expected shape.dim is %zu, they should be equal.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OPS_LOG_E(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), SFALayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void SFATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << SFADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), SFADataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus SFATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OPS_ERR_IF(it == DTYPE_SUPPORT_MAP.end(),
            OPS_LOG_E(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OPS_ERR_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void SFATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OPS_LOG_E(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void SFATilingCheck::LogErrorAttrValueSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "attr value");
}

ge::graphStatus SFATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorAttrValueSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


void SFATilingCheck::LogErrorLayoutSupport(const std::vector<SFALayout> &expectLayoutList,
    const SFALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << SFALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports layoutQuery %s, but got %s",
        name.c_str(), oss.str().c_str(), SFALayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus SFATilingCheck::CheckLayoutSupport(const SFALayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OPS_ERR_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OPS_LOG_E(opName_, "%s layoutQuery support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OPS_ERR_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaKey() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaValue() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaQueryRope() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.queryRope.desc, QUERY_ROPE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaKeyRope() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.keyRope.desc, KEY_ROPE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaAttenOut() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.attenOut.desc, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(outLayout_, ATTEN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaLayout() const
{
    const std::vector<std::string> layoutQueryList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    if (std::find(layoutQueryList.begin(), layoutQueryList.end(), layoutQuery) == layoutQueryList.end()) {
        OPS_LOG_E(opName_,
            "Layout only supports BSND/TND, but got %s",
            layoutQuery.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaSparseMode() const
{
    OPS_ERR_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OPS_LOG_E(opName_, "sparseMode must == 0/3, but got: %u.", *opParamInfo_.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(topkLayout_, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValue() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() || 
        ge::GRAPH_SUCCESS != CheckSingleParaQueryRope() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyRope() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAttenOut() ||
        ge::GRAPH_SUCCESS != CheckSingleParaLayout() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckRopeExistence()
{
    OPS_ERR_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
        OPS_LOG_E(opName_, "KeyRope is null, but queryRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
        OPS_LOG_E(opName_, "QueryRope is null, but keyRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
        OPS_LOG_E(opName_, "In Mla situation, desc of keyRope and queryRope should not be null"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer != nullptr,
        OPS_LOG_E(opName_, "%s should be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
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

template <typename T>
ge::graphStatus SFATilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OPS_LOG_E(opName_, "Attr %s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OPS_LOG_E(opName_,
                "%s value should be %s, but got %s",
                name.c_str(),
                ossExpect.str().c_str(),
                ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckParaExistenceMlaNoquant() const
{
    std::map<std::string, const void *> mlaNoquantParamExistMap = {
        {"actualSeqLengths", opParamInfo_.actualSeqLengths.tensor},
        {"blockTable", opParamInfo_.blockTable.tensor},
    };
    std::map<std::string, const void *> mlaNoquantParamNotExistMap = {};
    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {};
    if (CheckExistenceByMap(mlaNoquantParamExistMap, mlaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckParaExistenceGqaNoquant() const
{
    std::map<std::string, const void *> gqaNoquantParamExistMap = {};

    std::map<std::string, const void *> gqaNoquantParamNotExistMap = {
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
    };
    if (CheckExistenceByMap(gqaNoquantParamExistMap, gqaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaNoquant();
}

ge::graphStatus SFATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

ge::graphStatus SFATilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const SFALayout &layoutQuery, const std::string &name)
{
    if (tensor == nullptr) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            SFALayoutToSerialString(layoutQuery).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

void SFATilingCheck::SetSFAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
    queryRopeShapeCmp_ = opParamInfo_.queryRope.tensor->GetStorageShape();
    keyRopeShapeCmp_ = opParamInfo_.keyRope.tensor->GetStorageShape();
}

ge::graphStatus SFATilingCheck::CheckQAndQRopeDType()
{
    if (opParamInfo_.query.desc->GetDataType() != inputQType_) {
        OPS_LOG_E(opName_, "query's dtype is %s, it should be %s.",
            SFADataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str(),
            SFADataTypeToSerialString(inputQType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.queryRope.desc->GetDataType() != inputQRopeType_) {
        OPS_LOG_E(opName_, "query's dtype is %s, it should be %s.",
            SFADataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
            SFADataTypeToSerialString(inputQRopeType_).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckQShape()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = qkHeadDim_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, queryShapeCmp_, qLayout_, QUERY_NAME);
}

ge::graphStatus SFATilingCheck::CheckQRopeShape()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = ropeHeadDim_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, queryRopeShapeCmp_, qLayout_, QUERY_ROPE_NAME);
}

ge::graphStatus SFATilingCheck::CheckQAndQRopeShape()
{
    if (ge::GRAPH_SUCCESS != CheckQShape() ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckTopkShape()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = sparseBlockCount_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus SFATilingCheck::CheckQAndQRope()
{
    if (ge::GRAPH_SUCCESS != CheckQAndQRopeDType() ||
        ge::GRAPH_SUCCESS != CheckQAndQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckKVDType()
{
    if (opParamInfo_.key.desc->GetDataType() != inputKvType_) {
        OPS_LOG_E(opName_, "key's dtype is %s, it should be %s.",
            SFADataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str(),
            SFADataTypeToSerialString(inputKvType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.value.desc->GetDataType() != inputKvType_) {
        OPS_LOG_E(opName_, "value's dtype is %s, it should be %s.",
            SFADataTypeToSerialString(opParamInfo_.value.desc->GetDataType()).c_str(),
            SFADataTypeToSerialString(inputKvType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.keyRope.desc->GetDataType() != inputKRopeType_) {
        OPS_LOG_E(opName_, "key_rope's dtype is %s, it should be %s.",
            SFADataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
            SFADataTypeToSerialString(inputKRopeType_).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckKVShapeForBatchContinuous()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.D = qkHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t SFATilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus SFATilingCheck::CheckKVShapeForPageAttention()
{
    uint32_t kvBlockElemNum = 32 / GetTypeSize(inputKvType_);

    int64_t blockNum = keyShapeCmp_.GetDim(0);
    SFATilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.D = qkHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t kRopeBlockElemNum = 32 / GetTypeSize(inputKRopeType_);
    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckKVShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckKVShapeForPageAttention();
    }

    OPS_LOG_E(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus SFATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckKVDType() ||
        ge::GRAPH_SUCCESS != CheckKVShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckAttenOut()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            SFADataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            SFADataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if(GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengths shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensMulti()
{
    uint32_t shapeSizeQ = 0, shapeSizeKV = 0;
    if (qLayout_ == SFALayout::TND) {
        if(GetActualSeqLenSize(shapeSizeKV, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (GetActualSeqLenSize(shapeSizeQ, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (shapeSizeQ == 0 || shapeSizeKV == 0) {
            OPS_LOG_E(opName_, "In TND layout, actualSeqLengthsQ shape size is %u, actualSeqLengths shape size is %u, they should  > 0.",
                shapeSizeQ, shapeSizeKV);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckMuiltPara()
{
    SetSFAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckQAndQRope() ||
        ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensMulti()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoQuantShape() const
{
    OPS_ERR_IF(n2Size_ != 1,
        OPS_LOG_E(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OPS_ERR_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OPS_LOG_E(opName_, "group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != 512,
        OPS_LOG_E(opName_, "qk_head_dim only support 512, but got %u", qkHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != vHeadDim_,
        OPS_LOG_E(opName_, "qk_head_dim[%u] should be equal to v_head_dim[%u]", qkHeadDim_, vHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(ropeHeadDim_ != 64,
        OPS_LOG_E(opName_, "rope_head_dim should be 64, but got %u", ropeHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qTSize_ > 1024 * 1024  / GetTypeSize(inputQType_),
        OPS_LOG_E(opName_, "query dim T should be smaller than 1024 * 1024  / sizeof(query_dtype) = %u, but got %u",
            1024 / GetTypeSize(inputQType_), qTSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OPS_ERR_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OPS_LOG_E(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    OPS_ERR_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OPS_LOG_E(opName_, "query dtype only support %s and %s, but got %s",
            SFADataTypeToSerialString(ge::DT_BF16).c_str(), SFADataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            SFADataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoquantPa() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_ERR_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(blockSize_ % 16 > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) should be 16-aligned.",
        blockSize_), return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(blockSize_ % sparseBlockSize_ > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) must be divided by sparse_block_size(%d), but now the remainder is %d.",
        blockSize_, sparseBlockSize_, blockSize_ % sparseBlockSize_), return ge::GRAPH_FAILED);

    if (qLayout_ == SFALayout::BSND) {
        OPS_ERR_IF(n2Size_ * qkHeadDim_ > 65536,
            OPS_LOG_E(opName_,
                "When input kvcache layout is BSH, the N * D of kvcache is %u, "
                "exceeds the maximum limit (%u) of the datacopy instruction.",
                n2Size_ * qkHeadDim_, COPYND2NZ_SRC_STRIDE_LIMITATION),
            return ge::GRAPH_FAILED);
    }
 
    OPS_ERR_IF(kvLayout_ != SFALayout::PA_BSND,
        OPS_LOG_E(opName_, "SparseFlashAttention is enabled, only supports "
            "key/value's layout is PA_BSND, but now is %s", SFALayoutToSerialString(kvLayout_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaNoquant();
}

ge::graphStatus SFATilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

void SFATilingCheck::Init()
{
    opName_ = sfaInfo_.opName;
    platformInfo_ = sfaInfo_.platformInfo;
    opParamInfo_ = sfaInfo_.opParamInfo;
    socVersion_ = sfaInfo_.socVersion;

    bSize_ = sfaInfo_.bSize;
    n1Size_ = sfaInfo_.n1Size;
    n2Size_ = sfaInfo_.n2Size;
    s1Size_ = sfaInfo_.s1Size;
    s2Size_ = sfaInfo_.s2Size;
    gSize_ = sfaInfo_.gSize;
    qkHeadDim_ = sfaInfo_.qkHeadDim;
    vHeadDim_ = sfaInfo_.vHeadDim;
    ropeHeadDim_ = sfaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = sfaInfo_.maxBlockNumPerBatch;
    qTSize_ = sfaInfo_.qTSize;
    kvTSize_ = sfaInfo_.kvTSize;
    blockSize_ = sfaInfo_.blockSize;
    sparseBlockCount_ = sfaInfo_.sparseBlockCount;
    sparseBlockSize_ = sfaInfo_.sparseBlockSize;

    inputQType_ = sfaInfo_.inputQType;
    inputKvType_ = sfaInfo_.inputKvType;
    inputQRopeType_ = sfaInfo_.inputQRopeType;
    inputKRopeType_ = sfaInfo_.inputKRopeType;
    outputType_ = sfaInfo_.outputType;

    qLayout_ = sfaInfo_.qLayout;
    topkLayout_ = sfaInfo_.topkLayout;
    kvLayout_ = sfaInfo_.kvLayout;
    outLayout_ = sfaInfo_.outLayout;

    kvStorageMode_ = sfaInfo_.kvStorageMode;
    l2CacheSize_ = sfaInfo_.l2CacheSize;
}

ge::graphStatus SFATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckMuiltPara() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool SFAInfoParser::HasAxis(const SFAAxis &axis, const SFALayout &layout) const
{   
    const auto& layoutIt = SFA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SFAAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    return true;
}

size_t SFAInfoParser::GetAxisIdx(const SFAAxis &axis, const SFALayout &layout) const
{
    const std::vector<SFAAxis>& axes = SFA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t SFAInfoParser::GetAxisNum(const gert::Shape &shape, const SFAAxis &axis,const SFALayout &layout) const
{
    return HasAxis(axis, layout) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

ge::graphStatus SFAInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layoutQuery == nullptr, OPS_LOG_E(opName_, "attr layoutQuery is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.layoutKV == nullptr, OPS_LOG_E(opName_, "attr layoutKV is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseBlockSize == nullptr, OPS_LOG_E(opName_, "attr sparseBlockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.scaleValue == nullptr, OPS_LOG_E(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    SFALayout &layout, const std::string &name)
{
    if ((tensor == nullptr)) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            SFALayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ");
}

ge::graphStatus SFAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("SparseFlashAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void SFAInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INPUT_INDEX);
}

void SFAInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void SFAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
}

ge::graphStatus SFAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<uint32_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<uint32_t>(SPARSE_MODE_ATTR_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SFALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, SFAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus SFAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SFALayout::TND) ? GetAxisNum(queryShape_, SFAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetKVTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以key的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    kvTSize_ = (kvLayout_ == SFALayout::TND) ? GetAxisNum(keyShape_, SFAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(queryShape_, SFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SFALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, SFAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, SFAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == SFALayout::PA_BSND) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    // kv存储模式基准值
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetKvLayout()
{
    const map<string, SFALayout> layoutKVMap = {
        {"BSND",        SFALayout::BSND},
        {"PA_BSND",     SFALayout::PA_BSND},
        {"TND",         SFALayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OPS_LOG_E(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (qLayout_ == SFALayout::TND && kvLayout_ != SFALayout::PA_BSND) {
        OPS_LOG_E(opName_, "When layoutQ is TND, layoutKV only supports PA_BSND, but now is %s.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == SFALayout::PA_BSND && keyDimNum != 4U) {
        OPS_LOG_E(opName_, "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %d.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ != SFALayout::BSND) {
        OPS_LOG_E(opName_, "the layout of key is %s, it is unsupported.", SFALayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    } else if (kvLayout_ == SFALayout::BSND){ // BSND
        s2Size_ = GetAxisNum(keyShape_, SFAAxis::S, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OPS_LOG_E(opName_, "the layout_kv is %u, blockTable must be provided.", SFALayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum <= 1) {
        OPS_LOG_E(opName_, "the dim num of block_table is %u, it should be greater 1.", dimNum);
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetBlockSize()
{
    if (kvLayout_ == SFALayout::PA_BSND) {
        blockSize_ = GetAxisNum(keyShape_, SFAAxis::Bs, kvLayout_);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, SFAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus SFAInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = GetAxisNum(valueShape_, SFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetRopeHeadDim()
{
    ropeHeadDim_ = GetAxisNum(queryRopeShape_, SFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<SFALayout, SFALayout>> layoutMap = {
        {"BSND",        {SFALayout::BSND,    SFALayout::BSND}},
        {"TND",         {SFALayout::TND,     SFALayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQuery);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OPS_LOG_E(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, SFAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, SFAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void SFAInfoParser::SetSFAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    queryRopeShape_ = opParamInfo_.queryRope.tensor->GetStorageShape();
}

ge::graphStatus SFAInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OPS_LOG_E(opName_, "num_key_value_heads or num_heads is incorrect, num_key_value_heads: %u, num_heads: %u",
            n2Size_, n1Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

void SFAInfoParser::GenerateInfo(SFATilingInfo &sfaInfo)
{
    sfaInfo.opName = opName_;
    sfaInfo.platformInfo = platformInfo_;
    sfaInfo.opParamInfo = opParamInfo_;
    sfaInfo.socVersion = socVersion_;

    sfaInfo.bSize = bSize_;
    sfaInfo.n1Size = n1Size_;
    sfaInfo.n2Size = n2Size_;
    sfaInfo.s1Size = s1Size_;
    sfaInfo.s2Size = s2Size_;
    sfaInfo.gSize = gSize_;
    sfaInfo.qkHeadDim = qkHeadDim_;
    sfaInfo.vHeadDim = vHeadDim_;
    sfaInfo.ropeHeadDim = ropeHeadDim_;
    sfaInfo.qTSize = qTSize_;
    sfaInfo.kvTSize = kvTSize_;
    sfaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    sfaInfo.sparseBlockCount = sparseBlockCount_;
    sfaInfo.needInit = needInit_;

    sfaInfo.inputQType = inputQType_;
    sfaInfo.inputKvType = inputKvType_;
    sfaInfo.inputQRopeType = inputQRopeType_;
    sfaInfo.inputKRopeType = inputKRopeType_;
    sfaInfo.outputType = outputType_;

    sfaInfo.kvStorageMode = kvStorageMode_;
    sfaInfo.l2CacheSize = l2CacheSize_;

    sfaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    sfaInfo.scaleValue = *opParamInfo_.scaleValue;
    sfaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    sfaInfo.blockSize = blockSize_;
    sfaInfo.blockTypeSize =  sizeof(float);
    sfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    sfaInfo.actualLenDimsQ = actualLenDimsQ_;
    sfaInfo.actualLenDimsKV = actualLenDimsKV_;
    sfaInfo.maxActualseq = maxActualseq_;
    sfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    sfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    sfaInfo.isSameActualseq = isSameActualseq_;

    sfaInfo.sparseMode = *opParamInfo_.sparseMode;

    sfaInfo.qLayout = qLayout_;
    sfaInfo.topkLayout = topkLayout_;
    sfaInfo.kvLayout = kvLayout_;
    sfaInfo.outLayout = outLayout_;
}

ge::graphStatus SFAInfoParser::Parse(SFATilingInfo &sfaInfo)
{
    if (context_ == nullptr) {
        OPS_LOG_E("SparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_FULL(DLOG_INFO, "SparseFlashAttention", "TilingContext: %s", SFADebugTilingContext(context_).c_str());
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetSFAShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(sfaInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashAttention)
    .Tiling(TilingSparseFlashAttention)
    .TilingParse<SparseFlashAttentionCompileInfo>(TilingPrepareForSparseFlashAttention);
} // namespace optiling

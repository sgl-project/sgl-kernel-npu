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
 * \file sparse_attn_sharedkv_tiling.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_TILING_H
#define SPARSE_ATTN_SHAREDKV_TILING_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include "tiling/tiling_api.h"
#include "ge_helper.h"
#include "sparse_attn_sharedkv_tiling_data.h"

namespace optiling {
namespace ge_helper = sglang::ge_helper;
// ------------------Common definitions--------------------------
struct SASTilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct SASTilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class SASLayout : uint32_t { BSND = 0, TND = 1, PA_ND = 2 };

enum class SASAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,  // sparse_indices K and key D use the same enum value for the same final-dimension position.
    T = 5,
    Bn = 6,  // block number
    Bs = 7   // block size
};

enum class SASTemplateMode : uint32_t { SWA_TEMPLATE_MODE = 0, CFA_TEMPLATE_MODE = 1, SCFA_TEMPLATE_MODE = 2 };

enum class KvStorageMode : uint32_t { BATCH_CONTINUOUS = 0, TENSOR_LIST = 1, PAGE_ATTENTION = 2 };

// ------------------Operator prototype index constants----------------
// Inputs Index
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t CU_SEQLENS_KV_INDEX = 8;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 9;
constexpr uint32_t SEQUSED_Q_INDEX = 10;
constexpr uint32_t SEQUSED_KV_INDEX = 11;
constexpr uint32_t SINKS_INDEX = 12;
constexpr uint32_t METADATA_INDEX = 13;
// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;

// Attributes Index
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 0;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 1;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 2;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 3;
constexpr uint32_t ATTR_ORI_KV_STRIDE_INDEX = 4;
constexpr uint32_t ATTR_CMP_KV_STRIDE_INDEX = 5;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 6;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 7;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 8;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 9;
constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE = 10;

// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_IDX_FOUR = 4;

// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

// Constants.
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;

// Input constraint constants.
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t METADATA_LIMIT = 1024;
constexpr uint32_t DIM_LIMIT = 512;
constexpr uint32_t TOPK_LIMIT = 1024;
constexpr uint32_t BLOCK_SIZE_LIMIT = 1024;

struct SASParaInfo {
    SASTilingRequiredParaInfo q = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriSparseIndices = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    SASTilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    SASTilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
    SASTilingOptionalParaInfo cuSeqLensKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo cuSeqLensCmpKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo sequsedKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo sinks = {nullptr, nullptr};
    SASTilingOptionalParaInfo metadata = {nullptr, nullptr};
    SASTilingRequiredParaInfo attnOut = {nullptr, nullptr};

    const float *softmaxScale = nullptr;
    const uint32_t *cmpRatio = nullptr;
    const uint32_t *oriMaskMode = nullptr;
    const uint32_t *cmpMaskMode = nullptr;
    const uint32_t *oriKvStride = nullptr;
    const uint32_t *cmpKvStride = nullptr;
    const uint32_t *oriWinLeft = nullptr;
    const uint32_t *oriWinRight = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKv = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

static std::string SASDataTypeToSerialString(ge::DataType type);
std::string SASLayoutToSerialString(SASLayout layout);

// -----------Operator tiling input information class---------------
class SASTilingInfo
{
public:
    const char *opName = nullptr;
    SASParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t gSize = 0;
    uint32_t qHeadDim = 0;
    uint32_t oriKvHeadDim = 0;
    uint32_t cmpKvHeadDim = 0;
    uint32_t qTSize = 0;  // Effective only for TND.

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;

    float softmaxScale = 0;
    int64_t cmpRatio = 1;
    uint64_t oriMaskMode = 0;
    uint64_t cmpMaskMode = 0;
    uint64_t oriKvStride = 0;
    uint64_t cmpKvStride = 0;
    int64_t oriWinLeft = 0;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;
    bool hasOriSparseIndices = false;
    uint32_t oriSparseIndexWidth = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;

    bool returnSoftmaxLse = false;
    // PageAttention
    uint32_t blockTypeSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    int32_t blockSize = 0;
    int32_t oriBlockSize = 0;
    int32_t cmpBlockSize = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    // DType
    ge::DataType qType = ge::DT_FLOAT16;
    ge::DataType oriKvType = ge::DT_FLOAT16;
    ge::DataType cmpKvType = ge::DT_FLOAT16;
    ge::DataType oriSparseIndicesType = ge::DT_INT32;
    ge::DataType outputType = ge::DT_FLOAT16;

    // Layout
    SASLayout qLayout = SASLayout::TND;
    SASLayout cmpSparseIndicesLayout = SASLayout::TND;
    SASLayout oriSparseIndicesLayout = SASLayout::TND;
    SASLayout kvLayout = SASLayout::PA_ND;
    SASLayout outLayout = SASLayout::BSND;

    // template mode
    SASTemplateMode perfMode = SASTemplateMode::SWA_TEMPLATE_MODE;
};

// -----------Operator tiling input parser and checker class---------------
class SASTilingCheck
{
public:
    explicit SASTilingCheck(const SASTilingInfo &sasInfo) : sasInfo_(sasInfo) {};
    ~SASTilingCheck() = default;
    virtual ge::graphStatus Process();

private:
    void Init();

    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                              const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const SASLayout &actualLayout, const std::string &name) const;
    template <typename T>
    void LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                               const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                               const std::string subName) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::vector<size_t> &expectDimNumList,
                                       const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<SASLayout> &expectLayoutList, const SASLayout &actualLayout,
                               const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const SASLayout &layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaOriKv() const;
    ge::graphStatus CheckSingleParaCmpKv() const;
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaOriSparseIndices() const;
    ge::graphStatus CheckSingleParaCmpSparseIndices() const;
    ge::graphStatus CheckSingleParaSinks() const;
    ge::graphStatus CheckSingleParaMetadata() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckSingleParaOriMaskMode() const;
    ge::graphStatus CheckSingleParaCmpMaskMode() const;
    ge::graphStatus CheckSingleParaOriKvStride() const;
    ge::graphStatus CheckSingleParaCmpKvStride() const;
    ge::graphStatus CheckSingleParaOriWinLeft() const;
    ge::graphStatus CheckSingleParaOriWinRight() const;
    ge::graphStatus CheckSingleParaOriBlockTable() const;
    ge::graphStatus CheckSingleParaCmpBlockTable() const;

    ge::graphStatus CheckParaExistence() const;
    ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckNotExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckExistenceByMap(std::map<std::string, const void *> &existMap,
                                        std::map<std::string, const void *> &notExistMap) const;

    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckFeatureShape() const;
    ge::graphStatus CheckFeatureLayout() const;
    ge::graphStatus CheckFeatureDtype() const;
    ge::graphStatus CheckFeaturePa() const;

    ge::graphStatus CheckMultiParaConsistency();
    void SetSASShapeCompare();
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype, const ge::DataType &expectDtype,
                                          const std::string &name) const;
    ge::graphStatus CheckOriAndCmpKv() const;
    ge::graphStatus CheckAttenOut() const;
    ge::graphStatus CheckActualSeqLensQ() const;
    ge::graphStatus CheckActualSeqLens() const;
    ge::graphStatus CheckBlockTable() const;

    gert::Shape queryShapeCmp_{};
    gert::Shape oriKvShapeCmp_{};
    gert::Shape cmpKvShapeCmp_{};
    gert::Shape oriKvSparseIndicesCmp_{};
    gert::Shape cmpKvSparseIndicesCmp_{};
    gert::Shape attenOutShapeCmp_{};

private:
    const char *opName_;
    SASParaInfo opParamInfo_;
    const SASTilingInfo &sasInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qHeadDim_ = 0;
    uint32_t oriKvHeadDim_ = 0;
    uint32_t cmpKvHeadDim_ = 0;

    uint32_t qTSize_ = 0;   // Effective only for TND.
    uint32_t kvTSize_ = 0;  // Effective only for TND.
    int64_t cmpRatio_ = 1;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;
    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;
    SASLayout qLayout_ = SASLayout::TND;
    SASLayout cmpSparseIndicesLayout_ = SASLayout::TND;
    SASLayout oriSparseIndicesLayout_ = SASLayout::TND;
    SASLayout outLayout_ = SASLayout::TND;
    SASLayout kvLayout_ = SASLayout::PA_ND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;
    bool hasOriSparseIndices_ = false;
    uint32_t oriSparseIndexWidth_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType oriSparseIndicesType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;
};

template <typename T>
inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

class SASInfoParser
{
public:
    explicit SASInfoParser(ge_helper::TilingContext *context) : context_(context) {}
    ~SASInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckUnrequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, SASLayout &layout,
                                        const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    ge::graphStatus GetSASTemplateMode(SASTilingInfo &sasInfo);
    void SetSASShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetKVTSize();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2SizeForTND();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQHeadDim();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetSinks();
    void GenerateInfo(SASTilingInfo &sasInfo);
    ge::graphStatus Parse(SASTilingInfo &sasInfo);

public:
    ge_helper::TilingContext *context_ = nullptr;
    const char *opName_;
    SASParaInfo opParamInfo_;

    bool HasAxis(const SASAxis &axis, const SASLayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const SASAxis &axis, const SASLayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const SASAxis &axis, const SASLayout &layout) const;
    uint32_t GetSparseIndexWidth(const gert::Shape &shape, const SASLayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t orikvTSize_ = 0;
    uint32_t cmpkvTSize_ = 0;
    uint32_t qHeadDim_ = 0;
    uint32_t oriKvHeadDim_ = 0;
    uint32_t cmpKvHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t sparseBlockCount_ = 0;
    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t actualLenDimsKV_ = 0;
    uint32_t actualLenDimsQ_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    // Layout
    SASLayout qLayout_ = SASLayout::TND;
    SASLayout cmpSparseIndicesLayout_ = SASLayout::TND;
    SASLayout oriSparseIndicesLayout_ = SASLayout::TND;
    SASLayout outLayout_ = SASLayout::BSND;
    SASLayout kvLayout_ = SASLayout::PA_ND;
    // PageAttention
    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;
    bool hasOriSparseIndices_ = false;
    uint32_t oriSparseIndexWidth_ = 0;

    // template mode
    SASTemplateMode perfMode_ = SASTemplateMode::SWA_TEMPLATE_MODE;

    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType oriSparseIndicesType_ = ge::DT_INT32;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape qShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape cmpKvShape_{};
    gert::Shape oriSparseIndicesShape_{};
    gert::Shape cmpSparseIndicesShape_{};
};

// ---------------Operator tiling class---------------
class SparseAttnSharedkvTiling
{
public:
    explicit SparseAttnSharedkvTiling(ge_helper::TilingContext *context) : context_(context) {};
    ge::graphStatus DoOpTiling(SASTilingInfo *tilingInfo);
    const SparseAttnSharedkvTilingData &GetTilingData() const
    {
        return tilingData_;
    }
    uint32_t GetBlockDim() const
    {
        return blockDim_;
    }

private:
    void SplitBalanced(SASTilingInfo *tilingInfo);
    void CalcUbBmm(SASTilingInfo *tilingInfo);
    ge_helper::TilingContext *context_ = nullptr;
    SASTemplateMode perfMode_ = SASTemplateMode::SWA_TEMPLATE_MODE;
    SparseAttnSharedkvTilingData tilingData_{};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    SASTilingInfo *sasInfo_ = nullptr;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 512;  // Fixed S2 tile size of 512.
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 64;
};

}  // namespace optiling
#endif

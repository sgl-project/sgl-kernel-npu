/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe tiling function implementation file
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe tiling function implementation file
 */
#if defined(__DAV_C310__)
#include <cstdint>
#include <string>

#include "../ops_log.h"
#include "../ops_error.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../mc2_tiling_utils.h"
#include "../../op_kernel/fused_deep_moe_a5/fused_deep_moe_a5_tiling.h"
using fp8_e8m0_t = int8_t;
using fp8_e4m3_t = int8_t;


using namespace ge;
using namespace Cam;
namespace {
constexpr const char *OPS_UTILS_LOG_SUB_MOD_NAME = "FUSED_DEEP_MOE";
constexpr const char *OPS_UTILS_LOG_PACKAGE_TYPE = "CAM_OPS";
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t GM_ALIGN_SIZE = 512;
constexpr uint32_t TOKEN_DTYPE_BYTE_SIZE = 2;
constexpr uint32_t L1_TILE_BYTE_SIZE = 32 * 1024;
constexpr uint32_t CUBE_WORKSPACE_STAGE = 4;
constexpr uint32_t RESERVED_WORKSPACE_SIZE = 256 * 1024;

constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_EXPERT_IDS_INDEX = 1;
constexpr uint32_t INPUT_GMM1_WEIGHT_INDEX = 2;
constexpr uint32_t INPUT_GMM1_WEIGHT_SCALE_INDEX = 3;
constexpr uint32_t INPUT_GMM2_WEIGHT_INDEX = 4;
constexpr uint32_t INPUT_GMM2_WEIGHT_SCALE_INDEX = 5;
constexpr uint32_t INPUT_EXPERT_SCALE_INDEX = 6;
constexpr uint32_t INPUT_SHARE_GMM1_WEIGHT_INDEX = 7;
constexpr uint32_t INPUT_SHARE_GMM1_WEIGHT_SCALE_INDEX = 8;
constexpr uint32_t INPUT_SHARE_GMM2_WEIGHT_INDEX = 9;
constexpr uint32_t INPUT_SHARE_GMM2_WEIGHT_SCALE_INDEX = 10;
constexpr uint32_t INPUT_SMOOTH_SCALE_INDEX = 11;
constexpr uint32_t INPUT_SHARE_SMOOTH_SCALE_INDEX = 12;
constexpr uint32_t INPUT_SHARE_X_ACTIVE_MASK_INDEX = 13;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 4;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 5;

constexpr uint32_t MIN_BATCH_SIZE = 0;
constexpr uint32_t MAX_BATCH_SIZE = 256;
constexpr uint32_t MAX_MOE_EXERT_NUM = 512;
constexpr uint32_t SUPPORT_TOP_K = 12;
constexpr uint32_t ONE_DIMS = 1;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t THREE_DIMS = 3;
constexpr uint32_t MIN_TOKEN_LENGTH = 512;
constexpr uint32_t MAX_TOKEN_LENGTH = 7168;
constexpr uint32_t MIN_GMM1_HIDDEN = 1024;
constexpr uint32_t MAX_GMM1_HIDDEN = 6144;
constexpr uint32_t GMM1_HIDDEN_ALIGN = 1024;
constexpr uint32_t TENSOR_HIDDEN_INDEX = 1;
constexpr uint32_t SINGLE_HIDDEN_INDEX = 2;
constexpr uint32_t MAX_TENSOR_COUNT = 256;
constexpr uint32_t MB_SIZE = 1024 * 1024;
constexpr uint32_t DOUBLE_BUFFER = 2;
}  // namespace

namespace optiling {
static size_t Ceil(size_t x, size_t y)
{
    return (x + y - 1) / y;
}
static size_t CeilUp(size_t x, size_t y)
{
    return (x + y - 1) / y * y;
}

static uint32_t CountTensorListLen(const gert::TilingContext &context, int descIndex)
{
    int count = 0;
    for (uint32_t i = 0; i < MAX_TENSOR_COUNT; i++) {
        auto tensorElement = context.GetDynamicInputTensor(descIndex, i);
        if (tensorElement == nullptr) {
            break;
        }
        count++;
    }
    return count;
}

static bool CheckOptionalInputExist(const gert::TilingContext &context, int descIndex)
{
    const gert::StorageShape* tensorStorageShape = context.GetOptionalInputShape(descIndex);
    bool tensorExist = (tensorStorageShape != nullptr);
    return tensorExist;
}

static ge::graphStatus CheckGmm1Shape(gert::TilingContext &context, FusedDeepMoeTilingData &tilingData)
{
    const char *nodeName = context.GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData.fusedDeepMoeInfo.moeExpertNumPerRank;
    uint32_t h = tilingData.fusedDeepMoeInfo.h;
    uint32_t gmm1ListLen = CountTensorListLen(context, INPUT_GMM1_WEIGHT_INDEX);
    auto gmm1FirstTensorElement = context.GetDynamicInputTensor(INPUT_GMM1_WEIGHT_INDEX, 0);
    OPS_ERR_IF(gmm1FirstTensorElement == nullptr,
        OPS_LOG_E(nodeName, "gmm1Weight is null."), return ge::GRAPH_FAILED);
    auto gmm1FirstTensorElementShape = gmm1FirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm1FirstTensorElementShape.GetDimNum();
    uint32_t epRankId = tilingData.fusedDeepMoeInfo.epRankId;
    uint32_t localExpertNum = moeExpertNumPerRank;
    bool isMxFp4 = tilingData.fusedDeepMoeInfo.mxActStorageFp4 == MX_FP4_QUANT_MODE;

    OPS_ERR_IF(elementDims != TWO_DIMS && elementDims != THREE_DIMS,
                    OPS_LOG_E(nodeName, "gmm1Weight shape is invalid."),
                    return ge::GRAPH_FAILED);
    if (gmm1ListLen > 1) { // List
        OPS_ERR_IF(gmm1ListLen != localExpertNum,
                OPS_LOG_E(nodeName, "gmm1 listlen does not equals to localExpertNum."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(h != gmm1FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Weight input length does not equals to token hidden size."),
                return ge::GRAPH_FAILED);
        tilingData.fusedDeepMoeInfo.gmm1HLen =
                                    static_cast<uint64_t>(gmm1FirstTensorElementShape.GetDim(TENSOR_HIDDEN_INDEX));
        tilingData.fusedDeepMoeInfo.isTensorList = true;
    } else { // Single
        if (elementDims == TWO_DIMS) {  // one localExpert perRank
            OPS_ERR_IF(h != gmm1FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Weight input length does not equals to token hidden size."),
                return ge::GRAPH_FAILED);
            tilingData.fusedDeepMoeInfo.gmm1HLen =
                                    static_cast<uint64_t>(gmm1FirstTensorElementShape.GetDim(SINGLE_HIDDEN_INDEX - 1));
        } else {    // multi localExperts perRank
            OPS_ERR_IF(localExpertNum != gmm1FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Weight does not match local expert number per rank."),
                return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm1FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm1Weight input length does not equals to token hidden size."),
                return ge::GRAPH_FAILED);
            tilingData.fusedDeepMoeInfo.gmm1HLen =
                                    static_cast<uint64_t>(gmm1FirstTensorElementShape.GetDim(SINGLE_HIDDEN_INDEX));
        }
        tilingData.fusedDeepMoeInfo.isTensorList = false;
    }
    if (isMxFp4) {
        tilingData.fusedDeepMoeInfo.gmm1HLen = tilingData.fusedDeepMoeInfo.gmm1HLen * 2;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShareExpertShapes(gert::TilingContext &context, FusedDeepMoeTilingData &tilingData)
{
    const char *nodeName = context.GetNodeName();
    uint32_t h = tilingData.fusedDeepMoeInfo.h;

    // Check share_gmm1_weight: [h, shareGmm1HLen] (2D) or [1, h, shareGmm1HLen] (3D)
    const gert::StorageShape* shareGmm1WeightStorageShape = context.GetOptionalInputShape(INPUT_SHARE_GMM1_WEIGHT_INDEX);
    auto shareGmm1OriginShape = shareGmm1WeightStorageShape->GetOriginShape();
    uint32_t gmm1WeightDims = shareGmm1OriginShape.GetDimNum();
    OPS_ERR_IF(gmm1WeightDims != TWO_DIMS && gmm1WeightDims != THREE_DIMS,
                    OPS_LOG_E(nodeName, "shareGmm1Weight shape is invalid."),
                    return ge::GRAPH_FAILED);

    uint64_t shareGmm1HLen = 0;
    if (gmm1WeightDims == TWO_DIMS) {  // [h, shareGmm1HLen] format
        OPS_ERR_IF(h != shareGmm1OriginShape.GetDim(0),
            OPS_LOG_E(nodeName, "shareGmm1Weight dim0 should be h(%u), but got %ld.", h, shareGmm1OriginShape.GetDim(0)),
            return ge::GRAPH_FAILED);
        shareGmm1HLen = static_cast<uint64_t>(shareGmm1OriginShape.GetDim(1));
    } else {    // [1, h, shareGmm1HLen] format (three dims)
        OPS_ERR_IF(1 != shareGmm1OriginShape.GetDim(0),
            OPS_LOG_E(nodeName, "shareGmm1Weight dim0 should be 1 for shared expert, but got %ld.", shareGmm1OriginShape.GetDim(0)),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(h != shareGmm1OriginShape.GetDim(1),
            OPS_LOG_E(nodeName, "shareGmm1Weight dim1 should be h(%u), but got %ld.", h, shareGmm1OriginShape.GetDim(1)),
            return ge::GRAPH_FAILED);
        shareGmm1HLen = static_cast<uint64_t>(shareGmm1OriginShape.GetDim(2));
    }
    tilingData.fusedDeepMoeInfo.shareGmm1HLen = shareGmm1HLen;
    OPS_ERR_IF(
        shareGmm1HLen < MIN_GMM1_HIDDEN || shareGmm1HLen > MAX_GMM1_HIDDEN,
        OPS_LOG_E(nodeName, "shareGmm1 hidden size is invalid. Only support [%u, %u].",
            MIN_GMM1_HIDDEN, MAX_GMM1_HIDDEN),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(shareGmm1HLen % GMM1_HIDDEN_ALIGN != 0,
        OPS_LOG_E(nodeName, "shareGmm1 hidden size must be divisible by %u, but got %lu.",
            GMM1_HIDDEN_ALIGN, shareGmm1HLen),
        return ge::GRAPH_FAILED);
    bool isMxFp4 = tilingData.fusedDeepMoeInfo.mxActStorageFp4 == MX_FP4_QUANT_MODE;
    if (isMxFp4) {
        tilingData.fusedDeepMoeInfo.shareGmm1HLen = tilingData.fusedDeepMoeInfo.shareGmm1HLen * 2;
    }

    // Check share_gmm1_weight_scale: [shareGmm1HLen] (1D) or [1, shareGmm1HLen] (2D)
    // const gert::StorageShape* shareGmm1ScaleStorageShape = context.GetOptionalInputShape(INPUT_SHARE_GMM1_WEIGHT_SCALE_INDEX);
    // OPS_ERR_IF(shareGmm1ScaleStorageShape == nullptr,
    //     OPS_LOG_E(nodeName, "shareGmm1Scale is null."), return ge::GRAPH_FAILED);
    // auto shareGmm1ScaleOriginShape = shareGmm1ScaleStorageShape->GetOriginShape();
    // uint32_t gmm1ScaleDims = shareGmm1ScaleOriginShape.GetDimNum();
    // OPS_ERR_IF(gmm1ScaleDims != ONE_DIMS && gmm1ScaleDims != TWO_DIMS,
    //                 OPS_LOG_E(nodeName, "shareGmm1Scale shape dims must be 1 or 2, but current dim num is %u.",
    //                         gmm1ScaleDims),
    //                 return ge::GRAPH_FAILED);
    // if (gmm1ScaleDims == ONE_DIMS) {  // [shareGmm1HLen] format
    //     OPS_ERR_IF(static_cast<uint64_t>(shareGmm1ScaleOriginShape.GetDim(0)) != shareGmm1HLen,
    //                     OPS_LOG_E(nodeName, "shareGmm1Scale length should be shareGmm1HLen(%lu), but got %ld.",
    //                             shareGmm1HLen, shareGmm1ScaleOriginShape.GetDim(0)),
    //                     return ge::GRAPH_FAILED);
    // } else {    // [1, shareGmm1HLen] format (2D)
    //     OPS_ERR_IF(1 != shareGmm1ScaleOriginShape.GetDim(0),
    //                     OPS_LOG_E(nodeName, "shareGmm1Scale dim0 should be 1, but got %ld.",
    //                             shareGmm1ScaleOriginShape.GetDim(0)),
    //                     return ge::GRAPH_FAILED);
    //     OPS_ERR_IF(static_cast<uint64_t>(shareGmm1ScaleOriginShape.GetDim(1)) != shareGmm1HLen,
    //                     OPS_LOG_E(nodeName, "shareGmm1Scale dim1 should be shareGmm1HLen(%lu), but got %ld.",
    //                             shareGmm1HLen, shareGmm1ScaleOriginShape.GetDim(1)),
    //                     return ge::GRAPH_FAILED);
    // }

    // // Check share_gmm2_weight: [shareGmm1HLen/2, h] (2D) or [1, shareGmm1HLen/2, h] (3D)
    // const gert::StorageShape* shareGmm2WeightStorageShape = context.GetOptionalInputShape(INPUT_SHARE_GMM2_WEIGHT_INDEX);
    // OPS_ERR_IF(shareGmm2WeightStorageShape == nullptr,
    //     OPS_LOG_E(nodeName, "shareGmm2Weight is null."), return ge::GRAPH_FAILED);
    // auto shareGmm2OriginShape = shareGmm2WeightStorageShape->GetOriginShape();
    // uint32_t gmm2WeightDims = shareGmm2OriginShape.GetDimNum();
    // OPS_ERR_IF(gmm2WeightDims != TWO_DIMS && gmm2WeightDims != THREE_DIMS,
    //                 OPS_LOG_E(nodeName, "shareGmm2Weight shape is invalid."),
    //                 return ge::GRAPH_FAILED);

    // uint64_t shareGmm2InputDim = shareGmm1HLen / 2;
    // if (gmm2WeightDims == TWO_DIMS) {  // [shareGmm1HLen/2, h] format
    //     OPS_ERR_IF(static_cast<uint64_t>(shareGmm2OriginShape.GetDim(0)) != shareGmm2InputDim,
    //         OPS_LOG_E(nodeName, "shareGmm2Weight dim0 should be shareGmm1HLen/2(%lu), but got %ld.",
    //                 shareGmm2InputDim, shareGmm2OriginShape.GetDim(0)),
    //         return ge::GRAPH_FAILED);
    //     OPS_ERR_IF(h != shareGmm2OriginShape.GetDim(1),
    //         OPS_LOG_E(nodeName, "shareGmm2Weight dim1 should be h(%u), but got %ld.", h, shareGmm2OriginShape.GetDim(1)),
    //         return ge::GRAPH_FAILED);
    // } else {    // [1, shareGmm1HLen/2, h] format (three dims)
    //     OPS_ERR_IF(1 != shareGmm2OriginShape.GetDim(0),
    //         OPS_LOG_E(nodeName, "shareGmm2Weight dim0 should be 1 for shared expert, but got %ld.", shareGmm2OriginShape.GetDim(0)),
    //         return ge::GRAPH_FAILED);
    //     OPS_ERR_IF(static_cast<uint64_t>(shareGmm2OriginShape.GetDim(1)) != shareGmm2InputDim,
    //         OPS_LOG_E(nodeName, "shareGmm2Weight dim1 should be shareGmm1HLen/2(%lu), but got %ld.",
    //                 shareGmm2InputDim, shareGmm2OriginShape.GetDim(1)),
    //         return ge::GRAPH_FAILED);
    //     OPS_ERR_IF(h != shareGmm2OriginShape.GetDim(2),
    //         OPS_LOG_E(nodeName, "shareGmm2Weight dim2 should be h(%u), but got %ld.", h, shareGmm2OriginShape.GetDim(2)),
    //         return ge::GRAPH_FAILED);
    // }

    // // Check share_gmm2_weight_scale: [h] (1D) or [1, h] (2D)
    // const gert::StorageShape* shareGmm2ScaleStorageShape = context.GetOptionalInputShape(INPUT_SHARE_GMM2_WEIGHT_SCALE_INDEX);
    // OPS_ERR_IF(shareGmm2ScaleStorageShape == nullptr,
    //     OPS_LOG_E(nodeName, "shareGmm2Scale is null."), return ge::GRAPH_FAILED);
    // auto shareGmm2ScaleOriginShape = shareGmm2ScaleStorageShape->GetOriginShape();
    // uint32_t gmm2ScaleDims = shareGmm2ScaleOriginShape.GetDimNum();
    // OPS_ERR_IF(gmm2ScaleDims != ONE_DIMS && gmm2ScaleDims != TWO_DIMS,
    //                 OPS_LOG_E(nodeName, "shareGmm2Scale shape dims must be 1 or 2, but current dim num is %u.",
    //                         gmm2ScaleDims),
    //                 return ge::GRAPH_FAILED);
    // if (gmm2ScaleDims == ONE_DIMS) {  // [h] format
    //     OPS_ERR_IF(h != shareGmm2ScaleOriginShape.GetDim(0),
    //                     OPS_LOG_E(nodeName, "shareGmm2Scale length should be h(%u), but got %ld.",
    //                             h, shareGmm2ScaleOriginShape.GetDim(0)),
    //                     return ge::GRAPH_FAILED);
    // } else {    // [1, h] format (2D)
    //     OPS_ERR_IF(1 != shareGmm2ScaleOriginShape.GetDim(0),
    //                     OPS_LOG_E(nodeName, "shareGmm2Scale dim0 should be 1, but got %ld.",
    //                             shareGmm2ScaleOriginShape.GetDim(0)),
    //                     return ge::GRAPH_FAILED);
    //     OPS_ERR_IF(h != shareGmm2ScaleOriginShape.GetDim(1),
    //                     OPS_LOG_E(nodeName, "shareGmm2Scale dim1 should be h(%u), but got %ld.",
    //                             h, shareGmm2ScaleOriginShape.GetDim(1)),
    //                     return ge::GRAPH_FAILED);
    // }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShareExpertDtypes(gert::TilingContext &context, FusedDeepMoeTilingData &tilingData)
{
    const char *nodeName = context.GetNodeName();
    auto gmm1ScaleTensor = context.GetDynamicInputTensor(INPUT_GMM1_WEIGHT_SCALE_INDEX, 0);
    auto gmm2ScaleTensor = context.GetDynamicInputTensor(INPUT_GMM2_WEIGHT_SCALE_INDEX, 0);
    auto shareGmm1ScaleTensor = context.GetOptionalInputTensor(INPUT_SHARE_GMM1_WEIGHT_SCALE_INDEX);
    auto shareGmm2ScaleTensor = context.GetOptionalInputTensor(INPUT_SHARE_GMM2_WEIGHT_SCALE_INDEX);
    OPS_ERR_IF((shareGmm1ScaleTensor->GetDataType() != gmm1ScaleTensor->GetDataType()),
            OPS_LOG_E(nodeName, "share expert weight1 scale datatype (%d) must be same with routed experts'(%d).",
                    static_cast<ge::DataType>(shareGmm1ScaleTensor->GetDataType()),
                    static_cast<ge::DataType>(gmm1ScaleTensor->GetDataType())
                ),
            return ge::GRAPH_FAILED);
    OPS_ERR_IF((shareGmm2ScaleTensor->GetDataType() != gmm2ScaleTensor->GetDataType()),
            OPS_LOG_E(nodeName, "share expert weight2 scale datatype (%d) must be same with routed experts'(%d).",
                    static_cast<ge::DataType>(shareGmm2ScaleTensor->GetDataType()),
                    static_cast<ge::DataType>(gmm2ScaleTensor->GetDataType())
                ),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGmm1ScaleShape(gert::TilingContext &context,
                                           const FusedDeepMoeTilingData &tilingData)
{
    const char *nodeName = context.GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData.fusedDeepMoeInfo.moeExpertNumPerRank;
    uint32_t n = tilingData.fusedDeepMoeInfo.gmm1HLen;
    uint32_t epRankId = tilingData.fusedDeepMoeInfo.epRankId;
    uint32_t localExpertNum = moeExpertNumPerRank;
    bool listFlag = false;

    uint32_t gmm1ScaleListLen = CountTensorListLen(context, INPUT_GMM1_WEIGHT_SCALE_INDEX);
    auto gmm1ScaleFirstTensorElement = context.GetDynamicInputTensor(INPUT_GMM1_WEIGHT_SCALE_INDEX, 0);
    OPS_ERR_IF(gmm1ScaleFirstTensorElement == nullptr,
        OPS_LOG_E(nodeName, "gmm1Scale is null."), return ge::GRAPH_FAILED);
    auto gmm1ScaleFirstTensorElementShape = gmm1ScaleFirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm1ScaleFirstTensorElementShape.GetDimNum();
    OPS_ERR_IF(elementDims != 1 && elementDims != 2, OPS_LOG_E(nodeName, "gmm1WeightScale shape is invalid."),
            return ge::GRAPH_FAILED);
    if (gmm1ScaleListLen > 1) { // List
        OPS_ERR_IF(gmm1ScaleListLen != localExpertNum,
                OPS_LOG_E(nodeName, "gmm1scale listlen does not equals to localExpertNum."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(n != gmm1ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Scale length does not equals to gmm1 hidden size."), return ge::GRAPH_FAILED);
        listFlag = true;
    } else { // Single
        if (elementDims == 1) { // one localExpert perRank
            OPS_ERR_IF(n != gmm1ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Scale length does not equals to gmm1 hidden size."), return ge::GRAPH_FAILED);
        } else { // multi localExperts perRank
            OPS_ERR_IF(localExpertNum != gmm1ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Scale does not match local expert number perRank."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(n != gmm1ScaleFirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm1Scale length does not equals to gmm1 hidden size."), return ge::GRAPH_FAILED);
        }
    }
    OPS_ERR_IF(listFlag != tilingData.fusedDeepMoeInfo.isTensorList,
        OPS_LOG_E(nodeName, "gmm1Scale listFlag does not match gmm1Weight listFlag."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGmm2Shape(const gert::TilingContext &context, const FusedDeepMoeTilingData &tilingData)
{
    const char *nodeName = context.GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData.fusedDeepMoeInfo.moeExpertNumPerRank;
    uint32_t h = tilingData.fusedDeepMoeInfo.h;
    uint32_t n = tilingData.fusedDeepMoeInfo.gmm1HLen;
    uint32_t epRankId = tilingData.fusedDeepMoeInfo.epRankId;
    uint32_t localExpertNum = moeExpertNumPerRank;
    bool listFlag = false;

    uint32_t gmm2ListLen = CountTensorListLen(context, INPUT_GMM2_WEIGHT_INDEX);
    auto gmm2FirstTensorElement = context.GetDynamicInputTensor(INPUT_GMM2_WEIGHT_INDEX, 0);
    OPS_ERR_IF(gmm2FirstTensorElement == nullptr,
        OPS_LOG_E(nodeName, "gmm2Weight is null."), return ge::GRAPH_FAILED);
    auto gmm2FirstTensorElementShape = gmm2FirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm2FirstTensorElementShape.GetDimNum();
    OPS_ERR_IF(elementDims != 2 && elementDims != 3, OPS_LOG_E(nodeName, "gmm2Weight shape is invalid."),
            return ge::GRAPH_FAILED);
    if (gmm2ListLen > 1) { // List
        OPS_ERR_IF(gmm2ListLen != localExpertNum,
                OPS_LOG_E(nodeName, "gmm2 does not match local expert number perRank."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(n / 2 != gmm2FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2 does not match half of gmm1 hidden size."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(h != gmm2FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2 does not match token hidden size."), return ge::GRAPH_FAILED);
        listFlag = true;
    } else { // Single
        if (elementDims == TWO_DIMS) { // one localExpert perRank
            OPS_ERR_IF(n / 2 != gmm2FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2 does not match half of gmm1 hidden size."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm2FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2 does not match token hidden size."), return ge::GRAPH_FAILED);
        } else { // multi localExperts perRank
            OPS_ERR_IF(localExpertNum != gmm2FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2 does not match local expert num perRank."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(n / 2 != gmm2FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2 does not match half of gmm1 hidden size."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm2FirstTensorElementShape.GetDim(2),
                OPS_LOG_E(nodeName, "gmm2 does not match token hidden size."), return ge::GRAPH_FAILED);
        }
    }
    OPS_ERR_IF(listFlag != tilingData.fusedDeepMoeInfo.isTensorList,
        OPS_LOG_E(nodeName, "gmm2 listFlag does not match gmm1Weight listFlag."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGmm2ScaleShape(gert::TilingContext &context,
                                           const FusedDeepMoeTilingData &tilingData)
{
    const char *nodeName = context.GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData.fusedDeepMoeInfo.moeExpertNumPerRank;
    uint32_t h = tilingData.fusedDeepMoeInfo.h;
    uint32_t epRankId = tilingData.fusedDeepMoeInfo.epRankId;
    uint32_t localExpertNum = moeExpertNumPerRank;
    bool listFlag = false;

    uint32_t gmm2ScaleListLen = CountTensorListLen(context, INPUT_GMM2_WEIGHT_SCALE_INDEX);
    auto gmm2ScaleFirstTensorElement = context.GetDynamicInputTensor(INPUT_GMM2_WEIGHT_SCALE_INDEX, 0);
    OPS_ERR_IF(gmm2ScaleFirstTensorElement == nullptr,
        OPS_LOG_E(nodeName, "gmm2Scale is null."), return ge::GRAPH_FAILED);
    auto gmm2ScaleFirstTensorElementShape = gmm2ScaleFirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm2ScaleFirstTensorElementShape.GetDimNum();
    OPS_ERR_IF(elementDims != 1 && elementDims != 2, OPS_LOG_E(nodeName, "gmm2WeightScale shape is invalid."),
            return ge::GRAPH_FAILED);
    if (gmm2ScaleListLen > 1) { // List
        OPS_ERR_IF(gmm2ScaleListLen != localExpertNum,
                OPS_LOG_E(nodeName, "gmm2scale listlen does not equals to localExpertNum."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(h != gmm2ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Scale does not match token hidden size."), return ge::GRAPH_FAILED);
        listFlag = true;
    } else { // Single
        if (elementDims == 1) { // one localExpert perRank
            OPS_ERR_IF(h != gmm2ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Scale does not match token hidden size."), return ge::GRAPH_FAILED);
        } else { // multi localExperts perRank
            OPS_ERR_IF(localExpertNum != gmm2ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Scale does not match local expert number perRank."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm2ScaleFirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2Scale does not match token hidden size."), return ge::GRAPH_FAILED);
        }
    }
    OPS_ERR_IF(listFlag != tilingData.fusedDeepMoeInfo.isTensorList,
        OPS_LOG_E(nodeName, "gmm2Scale listFlag does not match gmm1Weight listFlag."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorList(gert::TilingContext &context,
                                             FusedDeepMoeTilingData &tilingData)
{
    if (CheckGmm1Shape(context, tilingData) == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    // if (CheckGmm1Shape(context, tilingData) == ge::GRAPH_SUCCESS &&
    //     CheckGmm1ScaleShape(context, tilingData) == ge::GRAPH_SUCCESS &&
    //     CheckGmm2Shape(context, tilingData) == ge::GRAPH_SUCCESS &&
    //     CheckGmm2ScaleShape(context, tilingData) == ge::GRAPH_SUCCESS) {
    //     return ge::GRAPH_SUCCESS;
    // }
    return ge::GRAPH_FAILED;
}

ge::graphStatus CheckXActiveMaskShape(const gert::TilingContext &context, const char *nodeName,
                                      const FusedDeepMoeTilingData &tilingData)
{
    uint32_t batchSize = tilingData.fusedDeepMoeInfo.bs;
    const gert::StorageShape* xActiveMaskStorageShape = context.GetOptionalInputShape(
    INPUT_SHARE_X_ACTIVE_MASK_INDEX);
    OPS_ERR_IF(xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                OPS_LOG_E(nodeName, " xActiveMask scale shape dims must be 1, but current dim num is %lu.",
                        xActiveMaskStorageShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    const int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
    OPS_ERR_IF(xActiveMaskDim0 != batchSize, OPS_LOG_E(nodeName,
                "xActiveMask Dim0 must be batchSize(%u), but current dim is %ld.", batchSize, xActiveMaskDim0),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSmoothScales(const gert::TilingContext &context, const char *nodeName,
                                      const FusedDeepMoeTilingData &tilingData, bool calShareExpert)
{
    uint32_t moeExpertNum = tilingData.fusedDeepMoeInfo.moeExpertNum;
    uint32_t h = tilingData.fusedDeepMoeInfo.h;
    auto expertSmoothScalesTensor = context.GetOptionalInputTensor(INPUT_SMOOTH_SCALE_INDEX);
    const gert::StorageShape* expertSmoothScalesStorageShape = context.GetOptionalInputShape(
        INPUT_SMOOTH_SCALE_INDEX);
    auto expertSmoothScalesOriginShape = expertSmoothScalesStorageShape->GetOriginShape();
    OPS_ERR_IF((expertSmoothScalesTensor->GetDataType() != ge::DT_FLOAT),
                    OPS_LOG_E(nodeName, "expertSmoothScales datatype is invalid, datatype should be float32, but is %d.",
                            static_cast<ge::DataType>(expertSmoothScalesTensor->GetDataType())),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertSmoothScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "expertSmoothScales shape dims must be 2, but current dim num is %lu.",
                            expertSmoothScalesStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertSmoothScalesOriginShape.GetDim(0) != moeExpertNum,
        OPS_LOG_E(nodeName, "expertSmoothScales dim-0 must be equal moeExpertNum."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertSmoothScalesOriginShape.GetDim(1) != h,
        OPS_LOG_E(nodeName, "expertSmoothScales dim-1 must be equal to token hidden size."),
        return ge::GRAPH_FAILED);

    if (calShareExpert) {
        bool shareSmoothScalesExist = CheckOptionalInputExist(context, INPUT_SHARE_SMOOTH_SCALE_INDEX);
        OPS_ERR_IF(!shareSmoothScalesExist, OPS_LOG_E(nodeName,
                    "When routed expert smooth scales exist, shared expert smooth scales must exist, too."),
                    return ge::GRAPH_FAILED);
        auto shareSmoothScalesTensor = context.GetOptionalInputTensor(INPUT_SHARE_SMOOTH_SCALE_INDEX);
        const gert::StorageShape* shareSmoothScalesStorageShape = context.GetOptionalInputShape(
            INPUT_SHARE_SMOOTH_SCALE_INDEX);
        auto shareSmoothScalesOriginShape = shareSmoothScalesStorageShape->GetOriginShape();
        OPS_ERR_IF((shareSmoothScalesTensor->GetDataType() != ge::DT_FLOAT),
                OPS_LOG_E(nodeName, "shareSmoothScales datatype is invalid, datatype should be float32, but is %d.",
                        static_cast<ge::DataType>(shareSmoothScalesTensor->GetDataType())),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF(shareSmoothScalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                        OPS_LOG_E(nodeName, "shareSmoothScales shape dims must be 1, but current dim num is %lu.",
                                shareSmoothScalesStorageShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF(shareSmoothScalesOriginShape.GetDim(0) != h,
            OPS_LOG_E(nodeName, "shareSmoothScales dim-0 must be equal to token hidden size."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static bool IsMxFp4GmmWeight(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2;
}

static ge::graphStatus SetMxActStorageFromGmmWeight(const gert::TilingContext &context, const char *nodeName,
                                                    FusedDeepMoeTilingData &tilingData)
{
    auto gmm1Tensor = context.GetDynamicInputTensor(INPUT_GMM1_WEIGHT_INDEX, 0);
    OPS_ERR_IF(gmm1Tensor == nullptr, OPS_LOG_E(nodeName, "gmm1Weight is null."), return ge::GRAPH_FAILED);
    ge::DataType gmm1WeightDtype = gmm1Tensor->GetDataType();
    tilingData.fusedDeepMoeInfo.mxActStorageFp4 =
        IsMxFp4GmmWeight(gmm1WeightDtype) ? MX_FP4_QUANT_MODE : 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckData(const char *nodeName, FusedDeepMoeTilingData &tilingData)
{
    uint32_t batchSize = tilingData.fusedDeepMoeInfo.bs;
    OPS_ERR_IF(batchSize < MIN_BATCH_SIZE, OPS_LOG_E(nodeName, "batchSize(bs) must >= %u.", MIN_BATCH_SIZE),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(batchSize > MAX_BATCH_SIZE, OPS_LOG_E(nodeName, "batchSize(bs) must <= %u.", MAX_BATCH_SIZE),
                    return ge::GRAPH_FAILED);
    uint32_t tokenLength = tilingData.fusedDeepMoeInfo.h;
    OPS_ERR_IF(
        tokenLength < MIN_TOKEN_LENGTH || tokenLength > MAX_TOKEN_LENGTH,
        OPS_LOG_E(nodeName, "tokenLength(h) is invalid. Only support [%u, %u].", MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH),
        return ge::GRAPH_FAILED);
    uint32_t gmm1HLen = tilingData.fusedDeepMoeInfo.gmm1HLen;
    OPS_ERR_IF(
        gmm1HLen < MIN_GMM1_HIDDEN || gmm1HLen > MAX_GMM1_HIDDEN,
        OPS_LOG_E(nodeName, "gmm1 hidden size is invalid. Only support [%u, %u].", MIN_GMM1_HIDDEN, MAX_GMM1_HIDDEN),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmm1HLen % GMM1_HIDDEN_ALIGN != 0,
        OPS_LOG_E(nodeName, "gmm1 hidden size must be divisible by %u, but got %u.",
            GMM1_HIDDEN_ALIGN, gmm1HLen),
        return ge::GRAPH_FAILED);
    if (tilingData.fusedDeepMoeInfo.mxActStorageFp4 == MX_FP4_QUANT_MODE) {
        OPS_ERR_IF(tokenLength % 2 != 0,
            OPS_LOG_E(nodeName, "tokenLength(h) must be even for MX FP4 gmm weight, but got %u.",
                tokenLength),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(gmm1HLen % 2 != 0,
            OPS_LOG_E(nodeName, "gmm1 hidden size must be even for MX FP4 gmm weight, but got %u.",
                gmm1HLen),
            return ge::GRAPH_FAILED);
    }
    uint32_t topK = tilingData.fusedDeepMoeInfo.k;
    OPS_ERR_IF(topK > SUPPORT_TOP_K, OPS_LOG_E(nodeName, "topK(k) must <= %u.", SUPPORT_TOP_K),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(topK > tilingData.fusedDeepMoeInfo.moeExpertNum,
        OPS_LOG_E(nodeName, "topK(k) must <= moeExpertNum(%u).",
                tilingData.fusedDeepMoeInfo.moeExpertNum),
        return ge::GRAPH_FAILED);
    uint32_t globalBatchSize = tilingData.fusedDeepMoeInfo.globalBs;
    uint32_t epRankSize = tilingData.fusedDeepMoeInfo.epRankSize;
    if (globalBatchSize == 0) {
        globalBatchSize = epRankSize * batchSize;
        tilingData.fusedDeepMoeInfo.globalBs = globalBatchSize;
    } else {
        OPS_ERR_IF(globalBatchSize < 0, OPS_LOG_E(nodeName, "globalBatchSize must >= 0."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(globalBatchSize % epRankSize > 0,
                        OPS_LOG_E(nodeName, "globalBatchSize must be divisible by epRankSize."),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext &context, const char *nodeName,
                                               FusedDeepMoeTilingData &tilingData, std::string &groupEp)
{
    auto attrs = context.GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);

    uint32_t epRankSize = static_cast<uint32_t>(*epRankSizePtr);
    uint32_t epRankId = static_cast<uint32_t>(*epRankIdPtr);
    uint32_t moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    uint32_t moeExpertNumPerRank = moeExpertNum / epRankSize;

    OPS_ERR_IF(epRankSize <= 0, OPS_LOG_E(nodeName, "epRankSize must > 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epRankId < 0, OPS_LOG_E(nodeName, "epRankId must >= 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epRankId >= epRankSize, OPS_LOG_E(nodeName, "epRankId must < epRankSize."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNum > MAX_MOE_EXERT_NUM, OPS_LOG_E(nodeName, "moeExpertNum must <= %u.", MAX_MOE_EXERT_NUM),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNumPerRank * epRankSize > MAX_MOE_EXERT_NUM,
                    OPS_LOG_E(nodeName, "moeExpertNumPerRank * epRankSize must <= %u.", MAX_MOE_EXERT_NUM),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNum <= 0, OPS_LOG_E(nodeName, "moeExpertNum must > 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF((moeExpertNum % epRankSize) != 0,
                    OPS_LOG_E(nodeName, "moeExpertNum must be divisible by epRankSize."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(quantModePtr == nullptr, OPS_LOG_E(nodeName, "quantModePtr is nullptr."), return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.fusedDeepMoeInfo.epRankSize = epRankSize;
    tilingData.fusedDeepMoeInfo.epRankId = epRankId;
    tilingData.fusedDeepMoeInfo.moeExpertNum = moeExpertNum;
    tilingData.fusedDeepMoeInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.fusedDeepMoeInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    tilingData.fusedDeepMoeInfo.moeExpertNumPerRank = moeExpertNumPerRank;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext &context, FusedDeepMoeTilingData &tiling, const std::string groupEp)
{
    const char *nodeName = context.GetNodeName();
    OPS_LOG_D(nodeName, "FusedDeepMoe groupEp = %s", groupEp.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    // std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(3);
    mc2CcTilingConfig.GetTiling(tiling.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling.mc2CcTiling);
}

static ge::graphStatus CheckHcclBufferSize(const char *nodeName, const FusedDeepMoeTilingData &tilingData)
{
    uint32_t moeExpertNumPerRank = tilingData.fusedDeepMoeInfo.moeExpertNumPerRank;
    uint32_t globalBatchSize = tilingData.fusedDeepMoeInfo.globalBs;
    uint32_t h = tilingData.fusedDeepMoeInfo.h;
    uint64_t bufferDemand = moeExpertNumPerRank * globalBatchSize * h * TOKEN_DTYPE_BYTE_SIZE * DOUBLE_BUFFER;
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    OPS_ERR_IF(bufferDemand > maxWindowSize,
                    OPS_LOG_E(nodeName,
                            "HCCL_BUFFSIZE is too SMALL, globalBatchSize = %u, h = %u, moeExpertNumPerRank = %u,"
                            " NEEDED_HCCL_BUFFSIZE(moeExpertNumPerRank * globalBatchSize * h *"
                            " TOKEN_DTYPE_BYTE_SIZE * doubleBuffer) B=%luMB, HCCL_BUFFSIZE=%luMB.",
                            globalBatchSize, h, moeExpertNumPerRank, (bufferDemand + MB_SIZE - 1) / MB_SIZE,
                            maxWindowSize / MB_SIZE),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static size_t MxActStorageBytes(size_t logicalElems, bool isMxFp4)
{
    if (isMxFp4) {
        return (logicalElems + 1U) / 2U;
    }
    return logicalElems * sizeof(fp8_e4m3_t);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext &context, const char *nodeName,
                                    FusedDeepMoeTilingData &tilingData, bool calShareExpert)
{
    size_t *workSpaces = context.GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    size_t maxTokenNum;
    uint32_t epRankSize = tilingData.fusedDeepMoeInfo.epRankSize;
    uint32_t batchSize = tilingData.fusedDeepMoeInfo.bs;
    uint32_t globalBs = tilingData.fusedDeepMoeInfo.globalBs;
    uint32_t topK = tilingData.fusedDeepMoeInfo.k;
    uint32_t moeExpertNumPerRank = tilingData.fusedDeepMoeInfo.moeExpertNumPerRank;
    uint32_t h = tilingData.fusedDeepMoeInfo.h;
    uint32_t aicNum = tilingData.fusedDeepMoeInfo.aicNum;
    uint32_t shareExpertTokenNum = calShareExpert ? batchSize : 0;
    uint64_t shareGmm1HLen = tilingData.fusedDeepMoeInfo.shareGmm1HLen;
    uint64_t shareGmm2HLen = shareGmm1HLen / 2;
    uint64_t gmm1HLen = tilingData.fusedDeepMoeInfo.gmm1HLen;
    uint64_t gmm2HLen = gmm1HLen / 2;
    uint64_t x1MxScaleNum = CeilUp(Ceil(h, 32), 2);
    uint64_t shareX1MxScaleNum = x1MxScaleNum;
    uint64_t x2MxScaleNum = CeilUp(Ceil(gmm2HLen, 32), 2);
    uint64_t shareX2MxScaleNum = CeilUp(Ceil(shareGmm2HLen, 32), 2);;
    maxTokenNum = globalBs * std::min(topK, moeExpertNumPerRank);
    bool isMxFp4 = tilingData.fusedDeepMoeInfo.mxActStorageFp4 == MX_FP4_QUANT_MODE;

    size_t x1TokenSize = MxActStorageBytes(shareExpertTokenNum * h + maxTokenNum * h, isMxFp4);
    size_t x2TokenSize = MxActStorageBytes(shareExpertTokenNum * shareGmm2HLen + maxTokenNum * gmm2HLen, isMxFp4);
    size_t maxTokenSize = CeilUp(x1TokenSize < x2TokenSize ? x2TokenSize : x1TokenSize, GM_ALIGN_SIZE);
    // size_t tokenScaleSize = CeilUp((shareExpertTokenNum + maxTokenNum) * sizeof(float), GM_ALIGN_SIZE);
    size_t x1MxScaleSize = (shareExpertTokenNum * shareX1MxScaleNum + maxTokenNum * x1MxScaleNum) * sizeof(fp8_e8m0_t);
    size_t x2MxScaleSize = (shareExpertTokenNum * shareX2MxScaleNum + maxTokenNum * x2MxScaleNum) * sizeof(fp8_e8m0_t);
    size_t maxMxScaleSize = CeilUp(x1MxScaleSize < x2MxScaleSize ? x2MxScaleSize : x1MxScaleSize, GM_ALIGN_SIZE);
    size_t gmm1SwapSize = (maxTokenNum * gmm1HLen + shareExpertTokenNum * shareGmm1HLen) * sizeof(float);
    size_t swigluOutSize = (maxTokenNum * gmm1HLen + shareExpertTokenNum * shareGmm1HLen) * sizeof(float);
    size_t gmm2SwapSize = (maxTokenNum * h + shareExpertTokenNum * h) * sizeof(float);
    size_t maxSwapSwigluSize = CeilUp(gmm1SwapSize < gmm2SwapSize ? gmm2SwapSize : gmm1SwapSize, GM_ALIGN_SIZE);
    size_t gmm2DepOutSize = CeilUp(moeExpertNumPerRank > 1 ?
        0 : maxTokenNum * h * TOKEN_DTYPE_BYTE_SIZE, GM_ALIGN_SIZE);
    size_t groupListSize = CeilUp(moeExpertNumPerRank * sizeof(int64_t), GM_ALIGN_SIZE);
    size_t expandIdxSize = CeilUp(batchSize * topK * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t epSendCountSize = CeilUp(epRankSize * moeExpertNumPerRank * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t reservedSize = CeilUp(RESERVED_WORKSPACE_SIZE, GM_ALIGN_SIZE);
    size_t offset = 0;
#ifdef ENABLE_REUSE_MEMORY
    tilingData.workSpaceOffset.shareX1TokenOffset = offset;
    tilingData.workSpaceOffset.shareX2TokenOffset = offset;
    tilingData.workSpaceOffset.x1TokenOffset = offset + MxActStorageBytes(shareExpertTokenNum * h, isMxFp4);
    tilingData.workSpaceOffset.x2TokenOffset = offset + MxActStorageBytes(shareExpertTokenNum * shareGmm2HLen, isMxFp4);
    offset += maxTokenSize;
    tilingData.workSpaceOffset.shareX1ScaleOffset = offset;
    tilingData.workSpaceOffset.shareX2ScaleOffset = offset;
    tilingData.workSpaceOffset.x1ScaleOffset = offset + shareExpertTokenNum * shareX1MxScaleNum * sizeof(fp8_e8m0_t);
    tilingData.workSpaceOffset.x2ScaleOffset = offset + shareExpertTokenNum * shareX2MxScaleNum * sizeof(fp8_e8m0_t);
    offset += maxMxScaleSize;
    tilingData.workSpaceOffset.shareMm1SwapSpaceOffset = offset;
    tilingData.workSpaceOffset.gmm1SwapSpaceOffset = offset + shareExpertTokenNum * shareGmm1HLen * sizeof(float);
    tilingData.workSpaceOffset.shareSwigluOffset = offset;
    tilingData.workSpaceOffset.swigluOffset = offset + shareExpertTokenNum * shareGmm1HLen * sizeof(float);
    tilingData.workSpaceOffset.shareMm2SwapSpaceOffset = offset;
    tilingData.workSpaceOffset.gmm2SwapSpaceOffset = offset + shareExpertTokenNum * h * sizeof(float);
    offset += maxSwapSwigluSize;
    tilingData.workSpaceOffset.y2TokenOffset = offset;
#else
    tilingData.workSpaceOffset.shareX1TokenOffset = offset;
    offset += CeilUp(MxActStorageBytes(shareExpertTokenNum * h, isMxFp4), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.shareX2TokenOffset = offset;
    offset += CeilUp(MxActStorageBytes(shareExpertTokenNum * shareGmm2HLen, isMxFp4), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.x1TokenOffset = offset;
    offset += CeilUp(MxActStorageBytes(maxTokenNum * h, isMxFp4), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.x2TokenOffset = offset;
    offset += CeilUp(MxActStorageBytes(maxTokenNum * gmm2HLen, isMxFp4), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.shareX1ScaleOffset = offset;
    offset += CeilUp(shareExpertTokenNum * x1MxScaleNum * sizeof(fp8_e8m0_t), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.shareX2ScaleOffset = offset;
    offset += CeilUp(shareExpertTokenNum * x2MxScaleNum * sizeof(fp8_e8m0_t), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.x1ScaleOffset = offset;
    offset += CeilUp(maxTokenNum * x1MxScaleNum * sizeof(fp8_e8m0_t), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.x2ScaleOffset = offset;
    offset += CeilUp(maxTokenNum * x2MxScaleNum * sizeof(fp8_e8m0_t), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.shareMm1SwapSpaceOffset = offset;
    offset += CeilUp(shareExpertTokenNum * shareGmm1HLen * sizeof(float), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.gmm1SwapSpaceOffset = offset;
    offset += CeilUp(maxTokenNum * gmm1HLen * sizeof(float), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.shareSwigluOffset = offset;
    offset += CeilUp(shareExpertTokenNum * shareGmm1HLen * sizeof(float), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.swigluOffset = offset;
    offset += CeilUp(swigluOutSize, GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.shareMm2SwapSpaceOffset = offset;
    offset += CeilUp(shareExpertTokenNum * h * sizeof(float), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.gmm2SwapSpaceOffset = offset;
    offset += CeilUp(maxTokenNum * h * sizeof(float), GM_ALIGN_SIZE);
    tilingData.workSpaceOffset.y2TokenOffset = offset;
#endif
    tilingData.workSpaceOffset.groupListOffset = tilingData.workSpaceOffset.y2TokenOffset + gmm2DepOutSize;
    tilingData.workSpaceOffset.expandIdxOffset = tilingData.workSpaceOffset.groupListOffset + groupListSize;
    tilingData.workSpaceOffset.epSendCountOffset = tilingData.workSpaceOffset.expandIdxOffset + expandIdxSize;
    tilingData.workSpaceOffset.reservedOffset = tilingData.workSpaceOffset.epSendCountOffset + epSendCountSize;
    size_t usrSize = tilingData.workSpaceOffset.reservedOffset + reservedSize;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + usrSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedDeepMoeTilingFuncImpl(gert::TilingContext &context)
{
    const char *nodeName = context.GetNodeName();
    OPS_ERR_IF(nodeName == nullptr, OPS_LOG_E("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);
    FusedDeepMoeTilingData *tilingData = context.GetTilingData<FusedDeepMoeTilingData>();
    OPS_ERR_IF(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";

    const gert::StorageShape *xStorageShape = context.GetInputShape(INPUT_X_INDEX);
    OPS_ERR_IF(xStorageShape == nullptr, OPS_LOG_E(nodeName, "x shape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "x shape dims must be 2, but current dim num is %lu.",
                            xStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t batchSize = xStorageShape->GetStorageShape().GetDim(0);
    tilingData->fusedDeepMoeInfo.bs = batchSize;
    const int64_t hiddenSize = xStorageShape->GetStorageShape().GetDim(1);
    tilingData->fusedDeepMoeInfo.h = hiddenSize;

    const gert::StorageShape *expertIdsStorageShape = context.GetInputShape(INPUT_EXPERT_IDS_INDEX);
    OPS_ERR_IF(expertIdsStorageShape == nullptr, OPS_LOG_E(nodeName, "expertIds shape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "expertIds shape dims must be 2, but current dim num is %lu.",
                            expertIdsStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertIdsStorageShape->GetStorageShape().GetDim(0) != batchSize,
                    OPS_LOG_E(nodeName, "expertIds dim 0 must be batchSize(%ld), but get %ld.",
                            batchSize, expertIdsStorageShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);
    const int64_t topK = expertIdsStorageShape->GetStorageShape().GetDim(1);
    tilingData->fusedDeepMoeInfo.k = topK;
    OPS_ERR_IF(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(SetMxActStorageFromGmmWeight(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Set mx act storage from gmm weight failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckWeightTensorList(context, *tilingData) != ge::GRAPH_SUCCESS,
           OPS_LOG_E(nodeName, "CheckWeightTensorList failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckHcclBufferSize(nodeName, *tilingData) != ge::GRAPH_SUCCESS,
           OPS_LOG_E(nodeName, "CheckHcclBuffSize failed."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    tilingData->fusedDeepMoeInfo.aicNum = aicNum;
    tilingData->fusedDeepMoeInfo.aivNum = aivNum;
    OPS_ERR_IF(CheckData(nodeName, *tilingData) != ge::GRAPH_SUCCESS, OPS_LOG_E(nodeName, "CheckData failed."),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape* xActiveMaskStorageShape = context.GetOptionalInputShape(
        INPUT_SHARE_X_ACTIVE_MASK_INDEX);
    bool xActiveMaskEnable = (xActiveMaskStorageShape != nullptr);
    if (xActiveMaskEnable) {
        OPS_ERR_IF(CheckXActiveMaskShape(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(nodeName, "CheckXActiveMaskShape failed."), return ge::GRAPH_FAILED);
    }
    const gert::StorageShape* shareGmm1WeightStorageShape = context.GetOptionalInputShape(INPUT_SHARE_GMM1_WEIGHT_INDEX);
    bool calShareExpert = (shareGmm1WeightStorageShape != nullptr);
    if (calShareExpert) {
        OPS_ERR_IF(CheckShareExpertDtypes(context, *tilingData) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(nodeName, "CheckShareExpertDtypes failed."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(CheckShareExpertShapes(context, *tilingData) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(nodeName, "CheckShareExpertShapes failed."), return ge::GRAPH_FAILED);
    }
    bool expertSmoothScalesExist = CheckOptionalInputExist(context, INPUT_SMOOTH_SCALE_INDEX);
    if (expertSmoothScalesExist) {
        OPS_ERR_IF(CheckSmoothScales(context, nodeName, *tilingData, calShareExpert) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(nodeName, "CheckSmoothScales failed."), return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(SetWorkSpace(context, nodeName, *tilingData, calShareExpert) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, *tilingData, groupEp);
    uint64_t tilingKey = 0;
    if (xActiveMaskEnable) {
        tilingKey |= EXEC_FLAG_X_ACTIVE_MASK;
    }
    if (calShareExpert) {
        tilingKey |= EXEC_FLAG_SHARED_EXPERT;
    }
    if (tilingData->fusedDeepMoeInfo.moeExpertNumPerRank != 1) {
        tilingKey |= EXEC_FLAG_DEEP_FUSE;
    }
    if (tilingData->fusedDeepMoeInfo.isTensorList) {
        tilingKey |= EXEC_FLAG_TENSOR_LIST;
    }
    if (expertSmoothScalesExist) {
        tilingKey |= EXEC_FLAG_SMOOTH_QUANT;
    }
    context.SetTilingKey(tilingKey);
    context.SetBlockDim(aicNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedDeepMoeTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = FusedDeepMoeTilingFuncImpl(*context);
    return ret;
}

struct FusedDeepMoeCompileInfo {};
ge::graphStatus TilingParseForFusedDeepMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedDeepMoe)
    .Tiling(FusedDeepMoeTilingFunc)
    .TilingParse<FusedDeepMoeCompileInfo>(TilingParseForFusedDeepMoe);
}  // namespace optiling
#endif  // defined(__DAV_C310__)


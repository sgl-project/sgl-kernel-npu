/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe tiling function implementation file
 * Author: WANG Qiankun
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe tiling function implementation file
 */
#include <cstdio>
#include <cstdint>
#include <string>

#include "error_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/fused_deep_moe_tiling.h"

constexpr uint32_t GM_ALIGN_SIZE = 512;

using namespace ge;
namespace {
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint32_t TOKEN_LENGTH = 7168;
    constexpr uint32_t TOKEN_DTYPE_BYTE_SIZE = 2;
    constexpr uint32_t GMM1_HIDDEN_SIZE = 4096;
    constexpr uint32_t GMM2_HIDDEN_SIZE = GMM1_HIDDEN_SIZE / 2;
    constexpr uint32_t USE_CORE_NUM = 24;
    constexpr uint32_t L1_TILE_BYTE_SIZE = 32 * 1024;
    constexpr uint32_t CUBE_WORKSPACE_STAGE = 4;
    constexpr uint32_t RESERVED_WORKSPACE_SIZE = 256 * 1024;

    constexpr uint32_t INPUT_X_INDEX = 0;
    constexpr uint32_t INPUT_EXPERT_IDS_INDEX = 1;

    constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
    constexpr uint32_t ATTR_EP_RANK_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
    constexpr uint32_t ATTR_SHARE_EXPERT_NUM_INDEX = 4;
    constexpr uint32_t ATTR_SHARE_EXPERT_RANK_NUM_INDEX = 5;
    constexpr uint32_t ATTR_QUANT_MODE_INDEX = 6;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 7;
}

namespace optiling {
static size_t CeilUp(size_t x, size_t y)
{
    return (x + y - 1) / y * y;
}

    static ge::graphStatus CheckData(const char *nodeName, FusedDeepMoeTilingData &tilingData)
    {
        uint32_t batchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.bs;
        OP_TILING_CHECK(batchSize < 8, OP_LOGE(nodeName, "batchSize must >= 8."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(batchSize > 256, OP_LOGE(nodeName, "batchSize must <= 256."), return ge::GRAPH_FAILED);
        uint32_t tokenLength = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.h;
        OP_TILING_CHECK(tokenLength != TOKEN_LENGTH, OP_LOGE(nodeName,
                                                             "tokenLength must be 7168."), return ge::GRAPH_FAILED);
        uint32_t topK = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.k;
        OP_TILING_CHECK(topK != 8, OP_LOGE(nodeName,
                                           "topK must be 8."), return ge::GRAPH_FAILED);
        uint32_t epRankSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
        OP_TILING_CHECK(epRankSize % 8 != 0, OP_LOGE(nodeName,
                                                     "epRankSize must be divisible by 8."), return ge::GRAPH_FAILED);
        uint32_t sharedExpertRankNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
        OP_TILING_CHECK(sharedExpertRankNum != 0, OP_LOGE(nodeName,
                                                          "sharedExpertRankNum must be 0."), return ge::GRAPH_FAILED);
        uint32_t globalBatchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.globalBs;
        OP_TILING_CHECK(globalBatchSize != batchSize * epRankSize, OP_LOGE(nodeName,
                                                                           "globalBatchSize must be batchSize * epRankSize."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
    FusedDeepMoeTilingData &tilingData, std::string &groupEp)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARE_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARE_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);

    int32_t moeExpertNum = static_cast<int32_t>(*moeExpertNumPtr);
    OP_TILING_CHECK(moeExpertNum <= 0, OP_LOGE(nodeName, "moeExpertNum must > 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNum > 64, OP_LOGE(nodeName,
                                               "moeExpertNum must <= 64."), return ge::GRAPH_FAILED);
    uint32_t epRankSize = static_cast<uint32_t>(*epRankSizePtr);
    uint32_t sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    OP_TILING_CHECK(moeExpertNum % (epRankSize - sharedExpertRankNum) != 0, OP_LOGE(nodeName,
                                                                                    "moeExpertNum must be divisible by (epRankSize - sharedExpertRankNum)."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNum < (epRankSize - sharedExpertRankNum), OP_LOGE(nodeName,
                                                                               "moeExpertNum must >= (epRankSize - sharedExpertRankNum)."), return ge::GRAPH_FAILED);
    uint32_t sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    OP_TILING_CHECK(sharedExpertNum != 1, OP_LOGE(nodeName,
                                                  "sharedExpertNum must be 1."), return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize = static_cast<uint32_t>(*epRankSizePtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank = static_cast<uint32_t>(*moeExpertNumPtr) / (
        static_cast<uint32_t>(*epRankSizePtr) - static_cast<uint32_t>(*sharedExpertRankNumPtr));
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext *context, FusedDeepMoeTilingData *tiling,
    const std::string groupEp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "FusedDeepMoe groupEp = %s", groupEp.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName,
    FusedDeepMoeTilingData &tilingData)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    size_t maxTokenNum;
    uint32_t epRankSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
    uint32_t epRankId = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    uint32_t sharedExpertRankNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    uint32_t batchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    uint32_t topK = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.k;
    uint32_t moeExpertNumPerRank = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    if (epRankId < sharedExpertRankNum) {
        maxTokenNum = batchSize * epRankSize / sharedExpertRankNum;
    } else {
        maxTokenNum = batchSize * epRankSize * std::min(topK, moeExpertNumPerRank);
    }

    size_t x2TokenSize = CeilUp(maxTokenNum * GMM2_HIDDEN_SIZE * sizeof(int8_t), GM_ALIGN_SIZE);
    size_t x2ScaleSize = CeilUp(maxTokenNum * sizeof(float), GM_ALIGN_SIZE);
    size_t CVSwapBufferSize = CeilUp(USE_CORE_NUM * L1_TILE_BYTE_SIZE * CUBE_WORKSPACE_STAGE * sizeof(int32_t),
                                     GM_ALIGN_SIZE);
    size_t swigluOutSize = CeilUp(maxTokenNum * GMM2_HIDDEN_SIZE * sizeof(float), GM_ALIGN_SIZE);
    size_t groupListSize = CeilUp(moeExpertNumPerRank * sizeof(int64_t), GM_ALIGN_SIZE);
    size_t expandIdxSize = CeilUp(batchSize * topK * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t epSendCountSize = CeilUp(epRankSize * moeExpertNumPerRank * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t x1TokenSize = CeilUp(maxTokenNum * TOKEN_LENGTH * sizeof(int8_t), GM_ALIGN_SIZE);
    size_t x1ScaleSize = CeilUp(maxTokenNum * sizeof(float), GM_ALIGN_SIZE);
    size_t gmm2DepOutSize = CeilUp(maxTokenNum * TOKEN_LENGTH * TOKEN_DTYPE_BYTE_SIZE, GM_ALIGN_SIZE);
    size_t resveredSize = CeilUp(RESERVED_WORKSPACE_SIZE, GM_ALIGN_SIZE);
    size_t usrSize = x2TokenSize + x2ScaleSize + CVSwapBufferSize + swigluOutSize + groupListSize +
                    expandIdxSize + epSendCountSize + x1TokenSize + x1ScaleSize + gmm2DepOutSize + resveredSize;

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + usrSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedDeepMoeTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    FusedDeepMoeTilingData *tilingData = context->GetTilingData<FusedDeepMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";

    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.bs = xDim0;

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const int64_t topK = expertIdsStorageShape->GetStorageShape().GetDim(1);
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.k = topK;
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h = TOKEN_LENGTH;

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkSpace(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, groupEp);
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank == 1 ||
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankId <
            tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum) {
        context->SetTilingKey(0);
    } else {
        context->SetTilingKey(EXEC_FLAG_DEEP_FUSE);
    }
    context->SetBlockDim(USE_CORE_NUM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedDeepMoeTilingFunc(gert::TilingContext* context)
{
    ge::graphStatus ret;
    printf("Enter FusedDeepMoe tiling check func. compile data %s %s\n", __DATE__, __TIME__);
    ret = FusedDeepMoeTilingFuncImpl(context);
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
} // namespace optiling

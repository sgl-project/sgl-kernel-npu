/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling_a5.cc
 * \brief
 */

#include "moe_distribute_combine_v2_ccu_tiling.h"

#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <sys/types.h>
#include <queue>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "register/tilingdata_base.h"
#include "mc2_tiling_utils.h"
#include "../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_tiling_helper.h"
namespace {
constexpr uint32_t OP_VERSION_2 = 2;
constexpr uint32_t ATTRS_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTRS_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTRS_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTRS_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTRS_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTRS_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTRS_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTRS_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTRS_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTRS_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTRS_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTRS_COMM_QUANT_MODE_INDEX = 12;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 512;
const uint64_t COMM_ALIGN = 512U;
const uint64_t STATUS_SIZE = 512U;

constexpr int64_t MOE_EXPERT_MAX_NUM_V1 = 512;
constexpr int64_t MOE_EXPERT_MAX_NUM_V2 = 1024;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

constexpr uint32_t DAVID_EP_WORLD_SIZE_FOUR = 4;
constexpr uint32_t DAVID_EP_WORLD_SIZE_TWO = 2;
constexpr uint32_t ENABLE = 1;
constexpr uint32_t NOT_ENABLE = 0;

const int64_t H_V1 = 7168;
const int64_t H_UPPER_BOUND_V2 = 8192;
const int64_t H_LOWER_BOUND_V2 = 1024;
const int64_t MAX_EP_WORLD_SIZE_V2 = 768;
const int64_t MIN_EP_WORLD_SIZE_V2 = 2;
const int64_t MAX_SHARED_EXPERT_NUM_V2 = 4;
const int64_t MIN_SHARED_EXPERT_NUM_V2 = 0;
const int64_t K_UPPER_BOUND_V1 = 8;
const int64_t K_UPPER_BOUND_V2 = 16;
const int64_t BUFFER_NUM = 2;

const std::string OP_NAME = "MoeDistributeCombineA5";

constexpr uint64_t TILING_KEY_CCU_TYPE = 60000;
constexpr uint32_t TILINGKEY_SCALES = 10;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineV2Info.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %d.", tilingData.moeDistributeCombineV2Info.bs);
    OP_LOGD(nodeName, "k is %d.", tilingData.moeDistributeCombineV2Info.k);
    OP_LOGD(nodeName, "h is %d.", tilingData.moeDistributeCombineV2Info.h);
    OP_LOGD(nodeName, "aivNum is %d.", tilingData.moeDistributeCombineV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %ld.", tilingData.moeDistributeCombineV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %ld.", tilingData.moeDistributeCombineV2Info.totalWinSize);
}

inline ge::graphStatus CheckEpAndTpWorldSize(const gert::TilingContext *context, std::string &groupEp,
                                             std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_WORLD_SIZE_INDEX);
    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTRS_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTRS_GROUP_TP_INDEX));
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(nodeName, "The groupEpPtr is null."), return ge::GRAPH_FAILED);
    uint64_t len = strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH);
    OP_TILING_CHECK(
        (len == 0) || (len == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName,
                "Valid length of groupEp must be in the range (0, %lu), but actually got strnlen(groupEp)=%lu.",
                MAX_GROUP_NAME_LENGTH, len),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epWorldSizePtr < MIN_EP_WORLD_SIZE_V2) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE_V2),
                    OP_LOGE(nodeName, "The valid range of epWorldSize is [%ld, %ld], but actually got epWorldSize=%ld.",
                            MIN_EP_WORLD_SIZE_V2, MAX_EP_WORLD_SIZE_V2, *epWorldSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
                    OP_LOGE(nodeName, "The valid range of tpWorldSize is [0, %ld], but actually got tpWorldSize=%ld.",
                            MAX_TP_WORLD_SIZE, *tpWorldSizePtr),
                    return ge::GRAPH_FAILED);
    groupEp = std::string(groupEpPtr);
    groupTp = std::string(groupTpPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpRankId(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_RANK_ID_INDEX);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE(nodeName, "The valid range of epRankId is [0, %ld), but actually got epRankId=%ld.",
                            *epWorldSizePtr, *epRankIdPtr),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckTpRankId(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_RANK_ID_INDEX);
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTRS_GROUP_TP_INDEX));
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
                        OP_LOGE(nodeName, "The valid range of tpRankId is [0, %ld), but actually got tpRankId=%ld.",
                                *tpWorldSizePtr, *tpRankIdPtr),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr), OP_LOGE(nodeName, "The groupTpPtr is null."), return ge::GRAPH_FAILED);
        uint64_t len = strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH);
        OP_TILING_CHECK(
            (len == 0) || (len == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "Valid length of groupTp should be in the range (0, %lu), but got strnlen(groupTp)=%lu.",
                    MAX_GROUP_NAME_LENGTH, len),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            *tpRankIdPtr != 0,
            OP_LOGE(nodeName, "The expected value of tpRankId is 0 in NoTp mode, but the actual value is %ld.",
                    *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckSharedExpertAttrs(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTRS_SHARED_EXPERT_RANK_NUM_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_WORLD_SIZE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTRS_SHARED_EXPERT_NUM_INDEX));

    OP_TILING_CHECK(
        (*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName,
                "The valid range of sharedExpertRankNum is [0, %ld), but actually got sharedExpertRankNum=%ld.",
                *epWorldSizePtr, *sharedExpertRankNumPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*sharedExpertNumPtr < MIN_SHARED_EXPERT_NUM_V2) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM_V2),
        OP_LOGE(nodeName, "The valid range of sharedExpertNum is [%ld, %ld], but the actual value is %ld.",
                MIN_SHARED_EXPERT_NUM_V2, MAX_SHARED_EXPERT_NUM_V2, *sharedExpertNumPtr),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

inline void SetSharedExpertNum(MoeDistributeCombineV2TilingData &tilingData, const uint32_t sharedExpertRankNum,
                               const uint32_t sharedExpertNum)
{
    if ((sharedExpertRankNum == 0) && (sharedExpertNum == 1)) {
        tilingData.moeDistributeCombineV2Info.sharedExpertNum = static_cast<uint32_t>(0);
    } else {
        tilingData.moeDistributeCombineV2Info.sharedExpertNum = static_cast<uint32_t>(sharedExpertNum);
    }
}

inline void SetEpTpInfo(MoeDistributeCombineV2TilingData &tilingData, const uint32_t epWorldSize,
                        const uint32_t tpWorldSize, const uint32_t epRankId, const uint32_t tpRankId)
{
    tilingData.moeDistributeCombineV2Info.epWorldSize = epWorldSize;
    tilingData.moeDistributeCombineV2Info.tpWorldSize = tpWorldSize;
    tilingData.moeDistributeCombineV2Info.epRankId = epRankId;
    tilingData.moeDistributeCombineV2Info.tpRankId = tpRankId;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context,
                                               MoeDistributeCombineV2TilingData &tilingData, std::string &groupEp,
                                               std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "The context attrs is null."), return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_RANK_ID_INDEX);
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTRS_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTRS_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTRS_SHARED_EXPERT_NUM_INDEX));
    // 判空
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "The epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "The tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "The epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "The tpRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardTypePtr == nullptr, OP_LOGE(nodeName, "The expertShardType is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertRankNum is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "The moeExpertNum is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertNum is null."),
                    return ge::GRAPH_FAILED);
    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(CheckEpAndTpWorldSize(context, groupEp, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpAndTpWorldSize failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckEpRankId(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "CheckEpRankId failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTpRankId(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "CheckEpRankId failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*expertShardTypePtr != 0,
                    OP_LOGE(nodeName, "The expected value of expertShardType is 0, but the actual value is %ld.",
                            *expertShardTypePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedExpertAttrs(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckSharedExpertAttrs failed."), return ge::GRAPH_FAILED);
    const int64_t MOE_EXPERT_MAX = MOE_EXPERT_MAX_NUM_V2;
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX),
                    OP_LOGE(nodeName, "The valid range of moeExpertNum is (0, %ld], but actually got moeExpertNum=%ld.",
                            MOE_EXPERT_MAX, *moeExpertNumPtr),
                    return ge::GRAPH_FAILED);
    SetEpTpInfo(tilingData, *epWorldSizePtr, *tpWorldSizePtr, *epRankIdPtr, *tpRankIdPtr);
    tilingData.moeDistributeCombineV2Info.expertShardType = static_cast<uint32_t>(*expertShardTypePtr);
    tilingData.moeDistributeCombineV2Info.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeCombineV2Info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    SetSharedExpertNum(tilingData, *sharedExpertRankNumPtr, *sharedExpertNumPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpWorldSize(const char *nodeName, uint32_t epWorldSize)
{
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE_V2) || (epWorldSize > MAX_EP_WORLD_SIZE_V2),
                    OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%u.",
                            MIN_EP_WORLD_SIZE_V2, MAX_EP_WORLD_SIZE_V2, epWorldSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const gert::TilingContext *context, const MoeDistributeCombineV2TilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U)) ||
                        ((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U)),
                    OP_LOGE(nodeName,
                            "sharedExpertRankNum and sharedExpertNum must both be zero or both non-zero, "
                            "but got sharedExpertRankNum=%u, sharedExpertNum=%u.",
                            sharedExpertRankNum, sharedExpertNum),
                    return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(
            ((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName,
                    "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
                    "sharedExpertNum=%u.",
                    sharedExpertRankNum, sharedExpertNum),
            return false);
    }

    return true;
}

static bool CheckAndSetGlobalBSAttrs(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData)
{
    bool isTokenMask = tilingData.moeDistributeCombineV2Info.isTokenMask;
    uint32_t epWorldSize = tilingData.moeDistributeCombineV2Info.epWorldSize;
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "The context attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTRS_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);

    OP_LOGD(nodeName, "MoeDistributeCombineA5 *globalBsPtr=%ld, bs=%ld, epWorldSize=%u\n", *globalBsPtr, expertIdsDim0,
            epWorldSize);
    OP_TILING_CHECK(
        (*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
                                ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
        OP_LOGE(nodeName,
                "globalBS is invalid, only "
                "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
                "bs=%ld, epWorldSize=%u.",
                *globalBsPtr, expertIdsDim0, epWorldSize),
        return false);
    OP_TILING_CHECK(((*globalBsPtr > (expertIdsDim0 * static_cast<int64_t>(epWorldSize))) && isTokenMask),
                    OP_LOGE(nodeName,
                            "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
                            "bs=%ld, epWorldSize=%u.",
                            *globalBsPtr, expertIdsDim0, epWorldSize),
                    return false);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return true;
}

static bool CheckExpertIdsAndSetBs(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIdshape is null."), return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > BS_UPPER_BOUND),
                    OP_LOGE(nodeName, "Invalid expertIds dims0(BS) %ld. Should be between [1, %ld].", expertIdsDim0,
                            BS_UPPER_BOUND),
                    return false);
    tilingData.moeDistributeCombineV2Info.bs = static_cast<uint32_t>(expertIdsDim0);

    return true;
}

static bool CheckAndSetAttrs(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                             uint32_t &localMoeExpertNum)
{
    const char *nodeName = context->GetNodeName();
    uint32_t epWorldSize = tilingData.moeDistributeCombineV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeCombineV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;

    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %u", localMoeExpertNum),
                    return false);
    // tpWorldSize 当前仅支持1
    OP_TILING_CHECK(
        tpWorldSize != 1,
        OP_LOGE(nodeName, "The tpWorldSize must be 1 in current version, but got tpWorldSize=%u.", tpWorldSize),
        return false);
    tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum = localMoeExpertNum;
    OP_TILING_CHECK(CheckEpWorldSize(nodeName, epWorldSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpWorldSize failed."), return false);

    OP_TILING_CHECK(!CheckSharedAttrs(context, tilingData), OP_LOGE(nodeName, "CheckSharedAttrs failed."),
                    return false);

    // 校验输入expertIds的维度0并设bs
    OP_TILING_CHECK(!CheckExpertIdsAndSetBs(context, tilingData), OP_LOGE(nodeName, "CheckExpertIdsAndSetBs failed."),
                    return false);
    // 校验globalBS
    OP_TILING_CHECK(!CheckAndSetGlobalBSAttrs(context, tilingData),
                    OP_LOGE(nodeName, "CheckAndSetGlobalBSAttrs failed."), return false);
    return true;
}

inline ge::graphStatus CheckSharedExpertXShape(const gert::TilingContext *context,
                                               MoeDistributeCombineV2TilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *sharedExpertXShape = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    uint32_t isSharedExpertX = (sharedExpertXShape != nullptr) ? ENABLE : NOT_ENABLE;
    tilingData.moeDistributeCombineV2Info.hasSharedExpertX = isSharedExpertX;
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    if (sharedExpertXShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t sharedExpertXDim0 = sharedExpertXShape->GetStorageShape().GetDim(0);
    int64_t sharedExpertXDim1 = sharedExpertXShape->GetStorageShape().GetDim(1);
    if (sharedExpertXShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
        OP_TILING_CHECK(sharedExpertXDim0 != expertIdsDim0,
                        OP_LOGE(nodeName, "sharedExpertX's dim0 not equal to bs, sharedExpertX's dim0 = %ld, bs = %ld",
                                sharedExpertXDim0, expertIdsDim0),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(sharedExpertXDim1 != expandXDim1,
                        OP_LOGE(nodeName, "sharedExpertX's dim1 not equal to h, sharedExpertX's dim1 = %ld, h = %ld",
                                sharedExpertXDim1, expandXDim1),
                        return ge::GRAPH_FAILED);
    } else {
        int64_t sharedExpertXDim2 = sharedExpertXShape->GetStorageShape().GetDim(TWO_DIMS);
        OP_TILING_CHECK(sharedExpertXDim0 * sharedExpertXDim1 != expertIdsDim0,
                        OP_LOGE(nodeName,
                                "sharedExpertX's dim0 * sharedExpertX's dim1 not equal to bs, "
                                "sharedExpertX's dim0 * sharedExpertX's dim1 = (%ld * %ld), bs = %ld.",
                                sharedExpertXDim0, sharedExpertXDim1, expertIdsDim0),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(sharedExpertXDim2 != expandXDim1,
                        OP_LOGE(nodeName, "sharedExpertX's dim2 not equal to h, sharedExpertX's dim2 = %ld, h = %ld",
                                sharedExpertXDim2, expandXDim1),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpSendCountAndTpSendCountShape(const gert::TilingContext *context,
                                                           const MoeDistributeCombineV2TilingData &tilingData,
                                                           int64_t tpWorldSize)

{
    bool isShared = true;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;
    uint32_t epRankId = tilingData.moeDistributeCombineV2Info.epRankId;
    if (epRankId >= sharedExpertRankNum) {  // 本卡为moe专家
        isShared = false;
    }
    const char *nodeName = context->GetNodeName();
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."),
                    return ge::GRAPH_FAILED);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epSendCount = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;
    OP_TILING_CHECK(epSendCountDim0 < epSendCount * tpWorldSize,
                    OP_LOGE(nodeName,
                            "The epSendCountDim0 not greater than or equal to epSendCount * tpWorldSize, "
                            "epSendCountDim0=%ld, epSendCount=%ld, tpWorldSize=%ld.",

                            epSendCountDim0, epSendCount, tpWorldSize),
                    return ge::GRAPH_FAILED);
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        const gert::StorageShape *tpSendCountStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
        OP_TILING_CHECK(tpSendCountStorageShape == nullptr, OP_LOGE(nodeName, "tpSendCounts is null."),
                        return ge::GRAPH_FAILED);
        const int64_t tpSendCountDim0 = tpSendCountStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            tpSendCountDim0 != tpWorldSize,
            OP_LOGE(nodeName, "tpSendCountDim0 not equal to tpWorldSize, tpSendCountDim0=%ld, tpWorldSize=%ld.",
                    tpSendCountDim0, tpWorldSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context,
                                             MoeDistributeCombineV2TilingData &tilingData)
{
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.tpWorldSize);
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    const char *nodeName = context->GetNodeName();
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE(nodeName, "expandIdx is null."), return ge::GRAPH_FAILED);
    int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandIdxDim0 < expertIdsDim0 * expertIdsDim1,
                    OP_LOGE(nodeName, "The expandIdxDim0 < bs * k, expandIdxDim0=%ld, (bs * k)=%ld.", expandIdxDim0,
                            expertIdsDim0 * expertIdsDim1),
                    return ge::GRAPH_FAILED);
    // 校验xActiveMask的维度
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    if (xActiveMaskStorageShape != nullptr) {
        const int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != expertIdsDim0,
                        OP_LOGE(nodeName, "The dim0 of xActiveMask should be equal to Bs=%ld, but actually got %ld.",
                                expertIdsDim0, xActiveMaskDim0),
                        return ge::GRAPH_FAILED);
    }
    // 校验epSendCount和tpSendCount的维度
    OP_TILING_CHECK(CheckEpSendCountAndTpSendCountShape(context, tilingData, tpWorldSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpSendCountAndTpSendCountShape failed."), return ge::GRAPH_FAILED);

    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."),
                    return ge::GRAPH_FAILED);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != expertIdsDim0,
                    OP_LOGE(nodeName, "expertScales' dim0 not equal to bs, expertScalesDim0=%ld, bs=%ld",
                            expertScalesDim0, expertIdsDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertScalesDim1 != expertIdsDim1,
                    OP_LOGE(nodeName, "expertScales' dim1 not equal to k, expertScalesDim1=%ld, k=%ld",
                            expertScalesDim1, expertIdsDim1),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedExpertXShape(context, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckSharedExpertXShape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline void CalculateAValue(const MoeDistributeCombineV2TilingData &tilingData, uint32_t &A, uint32_t localExpertNum,
                            int64_t expertIdsDim1)
{
    bool isShared = true;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;
    uint32_t epRankId = tilingData.moeDistributeCombineV2Info.epRankId;
    if (epRankId >= sharedExpertRankNum) {  // 本卡为moe专家
        isShared = false;
    }
    uint32_t globalBs = tilingData.moeDistributeCombineV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;

    if (isShared) {  // 本卡为共享专家
        uint32_t epWorldSizeU32 = tilingData.moeDistributeCombineV2Info.epWorldSize;
        uint32_t maxBs = globalBs / epWorldSizeU32;
        uint32_t maxSharedGroupNum = 0;
        uint32_t rankNumPerSharedExpert = 0;
        if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) {  // 除零保护
            rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
            maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
        }
        A = maxBs * maxSharedGroupNum;
    } else {  // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localExpertNum), expertIdsDim1);
    }
}

static bool CheckTensorShape(gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                             uint32_t localExpertNum)
{
    const char *nodeName = context->GetNodeName();
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    uint32_t A = 0;
    CalculateAValue(tilingData, A, localExpertNum, expertIdsDim1);
    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
                    OP_LOGE(nodeName,
                            "expandX's dim0 should be greater than or equal to A * tpWorldSize, "
                            "expandXDim0 = %ld, A = %ld, tpWorldSize = %ld",
                            expandXDim0, static_cast<int64_t>(A), tpWorldSize),
                    return false);
    OP_TILING_CHECK((expandXDim1 < H_LOWER_BOUND_V2) || (expandXDim1 > H_UPPER_BOUND_V2),
                    OP_LOGE(nodeName, "expandX dims1(H) should be in the range [%ld, %ld], but got %ld.",
                            H_LOWER_BOUND_V2, H_UPPER_BOUND_V2, expandXDim1),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.h = static_cast<uint32_t>(expandXDim1);

    const int64_t maxK = K_UPPER_BOUND_V2;
    OP_TILING_CHECK(
        (expertIdsDim1 <= 0) || (expertIdsDim1 > maxK),
        OP_LOGE(nodeName, "expertIdShape's dim1(k) should be in (0, %ld], but got expertIdShape's dim1=%ld.", maxK,
                expertIdsDim1),
        return false);
    tilingData.moeDistributeCombineV2Info.k = static_cast<uint32_t>(expertIdsDim1);
    // 校验expandIdx、epSendCount和tpSendCount、expertScales的维度
    OP_TILING_CHECK(CheckInputTensorShape(context, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckInputTensorShape failed."), return false);
    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName, "xDim0 not equal to bs, bs=%ld, xDim0=%ld", expertIdsDim0, xDim0), return false);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE(nodeName, "xDim1 not equal to h, xDim1=%ld, h=%ld", xDim1, expandXDim1), return false);

    return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const MoeDistributeCombineV2TilingData &tilingData,
                                    uint32_t localMoeExpertNum)
{
    const char *nodeName = context->GetNodeName();
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "get workspace failed"), return ge::GRAPH_FAILED);
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.globalBs) / epWorldSize;
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    workspace[0] =
        SYSTEM_NEED_WORKSPACE + epWorldSize * sizeof(uint64_t) * BUFFER_NUM * BUFFER_NUM +
        epWorldSize *
            (maxBs * Mc2TilingUtils::CeilAlign(h * ge::GetSizeByDataType(expandXDesc->GetDataType()), COMM_ALIGN) *
             localMoeExpertNum);
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

static void SetHcclTiling(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                          const std::string groupEp, const std::string groupTp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineV2 groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    // Only HalfAllToAllV is set, as A5 does not support TP communication.
    // Setting op types other than HalfAllToAllV will result in an error.
    uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV);
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling2);
}

inline ge::graphStatus CheckCommAttrs(const char *nodeName, const MoeDistributeCombineV2TilingData &tilingData,
                                      uint32_t localMoeExpertNum)
{
    // A5 实际物理分配为 HCCL_BUFFSIZE 的 2 倍
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize() * 2UL;
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.h);
    uint64_t aivNum = tilingData.moeDistributeCombineV2Info.aivNum;
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.globalBs) / epWorldSize;
    uint64_t actualSize = aivNum * STATUS_SIZE + epWorldSize * maxBs * Mc2TilingUtils::CeilAlign(h * 2UL, COMM_ALIGN) *
                                                     2UL * static_cast<uint64_t>(localMoeExpertNum);
    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
                "ep_worldsize * maxBs * Align512(h * 2) * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.",
                maxBs, h, epWorldSize, localMoeExpertNum, actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline void SetPlatformInfo(gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.moeDistributeCombineV2Info.aivNum = aivNum;
    tilingData.moeDistributeCombineV2Info.totalUbSize = ubSize;
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %lu, ubsize = %lu", numBlocks, aivNum, ubSize);
}

ge::graphStatus MoeDistributeCombineTilingImpl(gert::TilingContext *context)
{
    // Tiling implementation
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(nodeName, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "Start MoeDistributeCombineA5 tiling.");
    MoeDistributeCombineV2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    uint32_t localMoeExpertNum = 1;
    std::string groupEp = "";
    std::string groupTp = "";
    // Attrs
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, groupEp, groupTp) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    // X active mask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isTokenMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeCombineV2Info.isTokenMask = isTokenMask;
    // Check and Set Attrs
    OP_TILING_CHECK(!CheckAndSetAttrs(context, *tilingData, localMoeExpertNum),
                    OP_LOGE(nodeName, "attr check and set failed."), return ge::GRAPH_FAILED);
    // Shape
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, localMoeExpertNum),
                    OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);
    // Comm
    OP_TILING_CHECK(CheckCommAttrs(nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckCommAttrs failed."), return ge::GRAPH_FAILED);
    // A5 实际物理分配为 HCCL_BUFFSIZE 的 2 倍
    tilingData->moeDistributeCombineV2Info.totalWinSize = Mc2TilingUtils::GetMaxWindowSize() * 2UL;
    OP_TILING_CHECK(SetWorkSpace(context, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);
    SetHcclTiling(context, *tilingData, groupEp, groupTp);
    uint64_t tilingKey = TILING_KEY_CCU_TYPE;
    context->SetTilingKey(tilingKey);
    // Platform
    SetPlatformInfo(context, *tilingData);
    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD(nodeName, "Finish MoeDistributeCombine tiling.");
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

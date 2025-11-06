/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_tiling.cc
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "error_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "mc2_tiling_utils.h"
#include "experiment/platform/platform/platform_infos_def.h"
#include "../op_kernel/moe_distribute_dispatch_v2_tiling.h"

using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;
constexpr uint32_t OUTPUT_EXPAND_SCALES_INDEX = 6U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 13;
constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 14;
constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 15;
constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 16;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t DYN_SCALE_DIMS = 1;
constexpr uint32_t ASSIST_INFO_DIMS = 1;
constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint32_t ARR_LENGTH = 128;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t STATIC_SCALES = 1;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t STATIC_QUANT_MODE = 1;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768L;  // 384 * 2
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t EP_RESTRICT_8 = 8;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 512;

constexpr uint64_t NUM_10 = 10ULL;
constexpr uint32_t TILINGKEY_SCALES = 10;
constexpr uint32_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint32_t TP_WORLD_SIZE_TWO = 2;
constexpr uint32_t TILINGKEY_IS_SHARE_EXPERT = 1000;
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t K_MAX = 16;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL;  // scale32B + 3*4expandIdx
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;

// A2定义
const char *K_INNER_DEBUG = "MoeDistributeDispatchV2 Tiling Debug";
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 512;
constexpr size_t USER_WORKSPACE_A2 = 1UL * 1024UL * 1024UL;  // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
constexpr uint64_t INIT_TILINGKEY_A2 = 1000;
}  // namespace

namespace optiling {
// a2函数
static ge::graphStatus MoeDistributeDispatchA2CheckAttrAndSetTiling(gert::TilingContext *context,
                                                                    MoeDistributeDispatchV2Info &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."),
                    return GRAPH_FAILED);
    int32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(K_INNER_DEBUG, "groupEp is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > MAX_EP_WORLD_SIZE_A2 ||
                        *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
                    OP_LOGE(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
                    OP_LOGE(K_INNER_DEBUG, "epRankId is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr % *epWorldSizePtr != 0 || *moeExpertNumPtr <= 0 ||
                        *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2,
                    OP_LOGE(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "expertSharedType is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "sharedExpertRankNum is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr || (*quantModePtr != UNQUANT_MODE && *quantModePtr != DYNAMIC_QUANT_MODE),
                    OP_LOGE(K_INNER_DEBUG, "quantMode is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr || *expertTokenNumsTypePtr < 0 || *expertTokenNumsTypePtr > 1,
                    OP_LOGE(K_INNER_DEBUG, "expertTokenNumsType is invalid. Must be 0 or 1. "), return GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "zeroExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "copyExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "constExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;

    OP_LOGD(K_INNER_DEBUG, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
            constExpertNum);
    OP_TILING_CHECK(
        zeroComputeExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(K_INNER_DEBUG,
                "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
                zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum),
        return ge::GRAPH_FAILED);

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    info.quantMode = *quantModePtr;
    if (*globalBsPtr == 0) {
        info.globalBs = *epWorldSizePtr * bs;
    } else {
        info.globalBs = *globalBsPtr;
    }
    info.expertTokenNumsType = *expertTokenNumsTypePtr;
    info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "quantMode=%d", info.quantMode);
    OP_LOGD(K_INNER_DEBUG, "globalBs=%d", info.globalBs);
    OP_LOGD(K_INNER_DEBUG, "expertTokenNumsType=%d", info.expertTokenNumsType);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType=%d", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum=%d", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%d", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%d", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "tpWorldSize=%d", info.tpWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId=%d", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "tpRankId=%d", info.tpRankId);
    OP_LOGD(K_INNER_DEBUG, "zeroComputeExpertNum=%d", info.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2CheckShapeAndSetTiling(gert::TilingContext *context,
                                                                     MoeDistributeDispatchV2Info &info, bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
    const gert::StorageShape *expandScalesStorageShape = context->GetOutputShape(OUTPUT_EXPAND_SCALES_INDEX);

    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "xShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expertScalesStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertScales is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expandScalesStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expandScales is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(K_INNER_DEBUG, "x dims is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(K_INNER_DEBUG, "expertId dims is invalid."), return GRAPH_FAILED);
    OP_LOGD(nodeName, "X dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "X dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    uint32_t h = xStorageShape->GetStorageShape().GetDim(1);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    bool isScales = (scalesStorageShape != nullptr);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    OP_TILING_CHECK(h % BLOCK_SIZE_A2 != 0 || h <= 0 || h > MAX_HIDDEN_SIZE_A2,
                    OP_LOGE(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(bs <= 0 || bs > MAX_BATCH_SIZE_A2, OP_LOGE(K_INNER_DEBUG, "batchsize is invalid."),
                    return GRAPH_FAILED);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = static_cast<int32_t>(*constExpertNumPtr);
    int32_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    OP_TILING_CHECK(k <= 0 || k > MAX_K_VALUE_A2 || k > static_cast<uint32_t>(moeExpertNum + zeroComputeExpertNum),
                    OP_LOGE(K_INNER_DEBUG, "k is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(*quantModePtr == UNQUANT_MODE && isScales,
                    OP_LOGE(K_INNER_DEBUG, "scales should be null when quantMode is unQuant."), return GRAPH_FAILED);

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            ((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask must be 1-dimension or 2-dimension, but got %ld dim", xActiveMaskDimNums),
            return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != static_cast<int64_t>(bs),
                        OP_LOGE(nodeName,
                                "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
                                "expertIds's dim0 is %d",
                                xActiveMaskDim0, bs),
                        return GRAPH_FAILED);

        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
                         (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))),
                        OP_LOGE(nodeName,
                                "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %lu, "
                                "expertIds's dim1 is %d",
                                xActiveMaskStorageShape->GetStorageShape().GetDim(1), k),
                        return GRAPH_FAILED);
    }

    info.isTokenMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));
    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", info.bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", info.k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", info.h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(gert::TilingContext *context,
                                                                          MoeDistributeDispatchV2Info &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

// 为了兼容老版本，在未配置commAlg参数时，读取环境变量；
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余使用默认fullmesh不分层方案。
static ge::graphStatus MoeDistributeDispatchA2CheckCommAlg(gert::TilingContext *context, bool &isLayered)
{
    isLayered = false;
    auto attrs = context->GetAttrs();
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));

    const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
    const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
    if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr && strcmp(hcclIntraPcieEnable, "1") == 0 &&
        strcmp(hcclIntraRoceEnable, "0") == 0) {
        OP_LOGD(K_INNER_DEBUG,
                "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use hierarchy algorithm.");
        isLayered = true;
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGD(K_INNER_DEBUG,
                "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default fullmesh algorithm.");
    }

    if (commAlg == nullptr || strlen(commAlg) == 0) {
        OP_LOGE(K_INNER_DEBUG, "Attr commAlg is invalid, please configure fullmesh or hierarchy.");
        return GRAPH_FAILED;
    }

    OP_LOGI(K_INNER_DEBUG, "commAlg is %s", commAlg);
    if (strcmp(commAlg, "fullmesh") == 0) {
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(commAlg, "hierarchy") == 0) {
        isLayered = true;
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(K_INNER_DEBUG, "commAlg is not support");
        return GRAPH_FAILED;
    }
}

static uint64_t MoeDistributeDispatchA2CalcTilingKey(gert::TilingContext *context, const bool isLayered)
{
    uint64_t tilingKey = TILING_KEY_BASE_A2 + INIT_TILINGKEY_A2;
    if (isLayered) {
        tilingKey += TILING_KEY_LAYERED_COMM_A2;
    }

    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    tilingKey += static_cast<uint64_t>(*quantModePtr);

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    if (isScales) {
        tilingKey += NUM_10;
    }

    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);

    return tilingKey;
}

static ge::graphStatus MoeDistributeDispatchA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling func.");

    // 1. tilingData
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    MoeDistributeDispatchV2Info &info = tilingData->moeDistributeDispatchV2Info;

    bool isLayered = false;
    OP_TILING_CHECK(
        MoeDistributeDispatchA2CheckCommAlg(context, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MoeDistributeDispatchV2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "MoeDistributeDispatchV2 CheckShapeAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeDispatchA2CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MoeDistributeDispatchV2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "MoeDistributeDispatchV2 GetPlatformInfoAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetAicpuBlockDim(mc2tiling::AICPU_BLOCK_DIM_A2);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    std::string algConfig = isLayered ? "BatchWrite=level1:hierarchy" : "BatchWrite=level1:fullmesh";
    uint32_t opType = 18;  // BatchWrite

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OP_LOGI(nodeName, "Leave MoeDistributeDispatchV2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeDispatchA2TilingFuncImpl(context);
    } else {
        // ret = MoeDistributeDispatchA3TilingFuncImpl(context);
    }
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
}  // namespace optiling

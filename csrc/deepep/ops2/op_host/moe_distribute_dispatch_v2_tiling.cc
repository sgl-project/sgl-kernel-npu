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
#ifdef USE_CANN83_PATH
#include "platform/platform_infos_def.h"
#elif defined(USE_CANN82_PATH)
#include "experiment/platform/platform/platform_infos_def.h"
#else
#error "CANN version not supported or platform_infos_def.h not found. Check CANN_VERSION_MACRO definition."
#endif
#include "../op_kernel/moe_distribute_dispatch_v2_tiling.h"

using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;

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
constexpr uint64_t TILING_KEY_ONE_SERVER_A2 = 10000;
}  // namespace

namespace optiling {
static ge::graphStatus MoeDistributeDispatchA2GetIsSingle(gert::TilingContext *context, bool &isSingleServer)
{
    isSingleServer = false;
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "epWorldSizePtr is null."),
                    return ge::GRAPH_FAILED);

    int64_t epWorldSize = *epWorldSizePtr;
    OP_TILING_CHECK((epWorldSize <= 0) || (epWorldSize > MAX_EP_WORLD_SIZE_A2),
                    OP_LOGE(K_INNER_DEBUG, "epWorldSize is invalid, only support (%d, %d), but got epWorldSize=%ld.", 0,
                            MAX_EP_WORLD_SIZE_A2, epWorldSize),
                    return ge::GRAPH_FAILED);

    if (epWorldSize <= RANK_NUM_PER_NODE_A2) {
        isSingleServer = true;
    }

    return ge::GRAPH_SUCCESS;
}

// a2 single server
static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp,
                                               std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));

    // 判空
    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName, "groupEpPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedPtr == nullptr, OP_LOGE(nodeName, "expertSharedPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "expertTokenNumsTypePtr is null."),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
                    OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
                            MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
                    OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
                            MAX_TP_WORLD_SIZE, *tpWorldSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
                    OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.", epWorldSize,
                            *epRankIdPtr),
                    return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
                        OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
                                *tpWorldSizePtr, *tpRankIdPtr),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(
            *tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertSharedPtr != 0,
                    OP_LOGE(nodeName, "expertSharedType is invalid, only support 0, but got expertSharedType=%ld.",
                            *expertSharedPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
                    OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
                            MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
                epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
                    OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
                            MOE_EXPERT_MAX_NUM, moeExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %u], but got quantMode=%ld.", DYNAMIC_SCALES,
                *quantModePtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
                    OP_LOGE(nodeName, "expertTokenNumsType only support 0 or 1, but got expertTokenNumsType=%ld.",
                            *expertTokenNumsTypePtr),
                    return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.expertSharedType = static_cast<uint32_t>(*expertSharedPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeDispatchV2Info.sharedExpertNum == 1U) {
            tilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0U;
        }
    }
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);

    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isScales,
                           const uint32_t quantMode, const bool isActiveMask)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
                            xStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expertIdShape dims must be 2, but current dim num is %lu.",
                            expertIdStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);

    // 如果scales不为空进行shape维度检查
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return false);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(nodeName, "scalesShape dims must be 2, but current dim num is %lu.",
                                scalesStorageShape->GetStorageShape().GetDimNum()),
                        return false);
        OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
        OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
    }

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask shape is null."),
                        return false);
        const int64_t xActiveMaskDimNum = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            ((xActiveMaskDimNum != ONE_DIM) && (xActiveMaskDimNum != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask shape dim must be 1 or 2, but current dim num is %lu.", xActiveMaskDimNum),
            return false);
        OP_TILING_CHECK((xActiveMaskStorageShape->GetStorageShape().GetDim(0) != xDim0),
                        OP_LOGE(nodeName, "The input of xActiveMask dim0 = %ld is not equal to x dim0 = %ld.",
                                xActiveMaskStorageShape->GetStorageShape().GetDim(0), xDim0),
                        return false);
        OP_TILING_CHECK(
            ((xActiveMaskDimNum == TWO_DIMS) && (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdDim1)),
            OP_LOGE(nodeName, "The input of xActiveMask dim1 = %ld is not equal to expertId dim1 = %ld.",
                    xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdDim1),
            return false);
    }

    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandXShape is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expandXShape dims must be 2, but current dim num is %lu.",
                            expandXStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if (quantMode == DYNAMIC_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr, OP_LOGE(nodeName, "dynamicScalesShape is null."),
                        return false);
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_DIM_NUM,
                        OP_LOGE(nodeName, "dynamicScalesShape dims must be %u, but current dim num is %lu.",
                                DYNAMIC_SCALE_DIM_NUM, dynamicScalesStorageShape->GetStorageShape().GetDimNum()),
                        return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoShape is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "assistInfoShape dims must be 1, but current dim num is %lu.",
                            assistInfoStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr, OP_LOGE(nodeName, "expertTokenNumsShape is null."),
                    return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "expertTokenNumsShape dims must be 1, but current dim num is %lu.",
                            expertTokenNumsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCountShape is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "epRecvCountShape dims must be 1, but current dim num is %lu.",
                            epRecvCountStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "tpRecvCountShape is null."), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "tpRecvCountShape dims must be 1, but current dim num is %lu.",
                            tpRecvCountStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const uint32_t quantMode, const bool isActiveMask)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
                            std::to_string(xDesc->GetDataType()).c_str()),
                    return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "expertId dataType is invalid, dataType should be int32, but is %s.",
                            std::to_string(expertIdDesc->GetDataType()).c_str()),
                    return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "scales dataType is invalid, dataType should be float, but is %s.",
                                std::to_string(scalesDesc->GetDataType()).c_str()),
                        return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                        OP_LOGE(nodeName, "xActiveMask dataType is invalid, dataType should be bool, but is %s.",
                                std::to_string(xActiveMaskDesc->GetDataType()).c_str()),
                        return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (quantMode != NO_SCALES) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
                        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is %s.",
                                std::to_string(expandXDesc->GetDataType()).c_str()),
                        return false);
    } else {
        OP_TILING_CHECK(
            expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %s, but is %s.",
                    std::to_string(xDesc->GetDataType()).c_str(), std::to_string(expandXDesc->GetDataType()).c_str()),
            return false);
    }

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."), return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is %s.",
                                std::to_string(dynamicScalesDesc->GetDataType()).c_str()),
                        return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is %s.",
                            std::to_string(assistInfoDesc->GetDataType()).c_str()),
                    return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be int64, but is %s.",
                            std::to_string(expertTokenNumsDesc->GetDataType()).c_str()),
                    return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "epRecvCounts dataType is invalid, dataType should be int32, but is %s.",
                            std::to_string(epRecvCountsDesc->GetDataType()).c_str()),
                    return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is %s.",
                            std::to_string(tpRecvCountsDesc->GetDataType()).c_str()),
                    return false);

    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName, const bool isScales,
                              const uint32_t quantMode, const bool isActiveMask)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertId format is invalid."), return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "scales format is invalid."), return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "xActiveMask format is invalid."), return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expandX format is invalid."), return false);

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "dynamicScales format is invalid."), return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "assistInfoForCombine format is invalid."), return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertTokenNums format is invalid."), return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "epRecvCounts format is invalid."), return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "tpRecvCounts format is invalid."), return false);

    return true;
}

static ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
                                                        const bool isActiveMask, const bool isScales,
                                                        const uint32_t quantMode)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask),
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask),
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask),
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(gert::TilingContext *context, const char *nodeName,
                             MoeDistributeDispatchV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(
        (sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support 0 when sharedExpertNum is 0, but got %u.",
                sharedExpertRankNum),
        return false);
    OP_TILING_CHECK(
        (sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 when sharedExpertRankNum is 0, but got %u.",
                sharedExpertNum),
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

static ge::graphStatus CheckAttrs(gert::TilingContext *context, const char *nodeName,
                                  MoeDistributeDispatchV2TilingData &tilingData, uint32_t &localMoeExpertNum,
                                  bool isActiveMask)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(context, nodeName, tilingData),
                    OP_LOGE(nodeName, "Check shared expert related attributes failed."), return ge::GRAPH_FAILED);

    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE(nodeName,
                            "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
                            "but moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.",
                            moeExpertNum, epWorldSize, sharedExpertRankNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %u", localMoeExpertNum),
                    return ge::GRAPH_FAILED);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((tpWorldSize > 1) && (localMoeExpertNum > 1),
                    OP_LOGE(nodeName,
                            "Cannot support multi-moeExpert %u "
                            "in a rank when tpWorldSize = %u > 1",
                            localMoeExpertNum, tpWorldSize),
                    return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
                    OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.",
                            BS_UPPER_BOUND, xDim0),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0,
            epWorldSize);
    OP_TILING_CHECK(
        (*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
                                ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
        OP_LOGE(nodeName,
                "globalBS is invalid, only "
                "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
                "bs=%ld, epWorldSize=%u.",
                *globalBsPtr, xDim0, epWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
                    OP_LOGE(nodeName,
                            "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
                            "bs=%ld, epWorldSize=%u.",
                            *globalBsPtr, xDim0, epWorldSize),
                    return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
                                        MoeDistributeDispatchV2TilingData &tilingData, const uint32_t quantMode,
                                        const bool isScales, const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX),
                    OP_LOGE(nodeName, "xShape dims1(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, xDim1),
                    return ge::GRAPH_FAILED);  // 32字节对齐
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName,
                            "xShape's dim0 not equal to expertIdShape's dim0, "
                            "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.",
                            xDim0, expertIdsDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) || (expertIdsDim1 > moeExpertNum),
                    OP_LOGE(nodeName,
                            "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum=%ld)], "
                            "but got expertIdShape's dim1=%ld.",
                            K_MAX, moeExpertNum, expertIdsDim1),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验scales的维度
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(scalesDim0 != (static_cast<int64_t>(sharedExpertNum) + moeExpertNum),
                        OP_LOGE(nodeName,
                                "scales's dim0 not equal to sharedExpertNum + moeExpertNum, "
                                "scales's dim0 is %ld, sharedExpertNum is %ld, moeExpertNum is %ld.",
                                scalesDim0, static_cast<int64_t>(sharedExpertNum), moeExpertNum),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(xDim1 != scalesDim1,
                        OP_LOGE(nodeName,
                                "scales's dim1 not equal to xShape's dim1, "
                                "xShape's dim1 is %ld, scales's dim1 is %ld.",
                                xDim1, scalesDim1),
                        return ge::GRAPH_FAILED);
    }

    if (isSharedExpert) {  // 本卡为共享专家
        uint32_t rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
        uint32_t maxBs = globalBs / epWorldSizeU32;
        uint32_t maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
        A = maxBs * maxSharedGroupNum;
    } else {  // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }

    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
                    OP_LOGE(nodeName,
                            "expandX's dim0 not greater than or equal to A*tpWorldSize, "
                            "expandX's dim0 is %ld, A*tpWorldSize is %ld.",
                            expandXDim0, tpWorldSize * A),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE(nodeName,
                            "expandX's dim1 not equal to xShape's dim1, "
                            "xShape's dim1 is %ld, expandX's dim1 is %ld.",
                            xDim1, expandXDim1),
                    return ge::GRAPH_FAILED);

    // 校验dynamicScales的维度
    if (quantMode != NO_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize,
            OP_LOGE(
                nodeName,
                "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, dynamicScales's dim0 is %ld, "
                "A*tpWorldSize is %ld.",
                dynamicScalesDim0, A * tpWorldSize),
            return ge::GRAPH_FAILED);
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(assistInfoDim0 < static_cast<int64_t>(A * TRIPLE),
                    OP_LOGE(nodeName,
                            "assistInfoDim0 < A * 3,"
                            " assistInfoDim0 is %ld, A * 3 is %ld.",
                            assistInfoDim0, static_cast<int64_t>(A * TRIPLE)),
                    return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1,
                        OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.", expertTokenNumsDim0),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            expertTokenNumsDim0 != localMoeExpertNum,
            OP_LOGE(nodeName,
                    "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
                    "localMoeExpertNum is %ld.",
                    expertTokenNumsDim0, localMoeExpertNum),
            return ge::GRAPH_FAILED);
    }

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(
        epRecvCountDim0 < epRecvCount,
        OP_LOGE(
            nodeName,
            "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum * "
            "tpWorldSize, "
            "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, tpWorldSize is %ld.",
            epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tpRecvCountDim0 != tpWorldSize,
        OP_LOGE(nodeName,
                "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount is %ld, "
                "tpWorldSize is %ld.",
                tpRecvCountDim0, tpWorldSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

static void CalTilingKey(uint64_t &tilingKey, const bool isScales, const uint32_t quantMode, const uint32_t tpWorldSize,
                         const bool isShared)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    if (isScales) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES);
    }
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_TP_WORLD_SIZE);
    }
    if (isShared) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_IS_SHARE_EXPERT);
    }

    return;
}

static void SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tiling,
                        const std::string groupEp, const std::string groupTp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_ALL_GATHER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    // mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);

    // mc2CcTilingConfig.SetGroupName(groupTp);
    // mc2CcTilingConfig.SetOpType(opType2);
    // mc2CcTilingConfig.SetAlgConfig(algConfigAllGatherStr);
    // mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2);
}

static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertSharedType is %u.", tilingData.moeDistributeDispatchV2Info.expertSharedType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchV2Info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSize);
}

static ge::graphStatus MoeDistributeDispatchA2SingleTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = NO_SCALES;
    bool isScales = false;
    bool isActiveMask = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;

    // 检查quantMode和scales是否匹配
    OP_TILING_CHECK(quantMode == STATIC_SCALES, OP_LOGE(nodeName, "cannot support static quant now."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
                    OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
                            static_cast<int32_t>(isScales), quantMode),
                    return ge::GRAPH_FAILED);

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        TilingCheckMoeDistributeDispatch(context, nodeName, isActiveMask, isScales, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertNum = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales, isSharedExpert,
                                     static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.h);
    uint64_t k = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.k);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen =
        ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize = ((maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum)) +
                           (maxBs * tokenNeedSizeCombine * (k + static_cast<uint64_t>(sharedExpertNum)))) *
                          DOUBLE_DATA_BUFFER;
    OP_TILING_CHECK(
        (actualSize > maxWindowSize),
        OP_LOGE(
            nodeName,
            "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, sharedExpertNum = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum) +"
            " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB,"
            " HCCL_BUFFSIZE=%luMB.",
            maxBs, h, epWorldSize, localMoeExpertNum, sharedExpertNum, tokenNeedSizeDispatch, tokenNeedSizeCombine, k,
            actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    tilingData->moeDistributeDispatchV2Info.totalWinSize = maxWindowSize;
    OP_LOGD(nodeName, "windowSize = %lu", maxWindowSize);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, groupEp, groupTp);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    // uint64_t tilingKey = INIT_TILINGKEY;
    uint64_t tilingKey = TILING_KEY_BASE_A2 + INIT_TILINGKEY_A2 + TILING_KEY_ONE_SERVER_A2;  // a2 single server
    CalTilingKey(tilingKey, isScales, quantMode, tpWorldSize, isSharedExpert);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);  // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

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

    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "xShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."),
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

    bool isSingleServer = false;
    OP_TILING_CHECK(
        MoeDistributeDispatchA2GetIsSingle(context, isSingleServer) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MoeDistributeDispatchV2 GetIsSingle Failed"),
        return ge::GRAPH_FAILED);

    if (isSingleServer) {
        // 单机的tiling校验
        return MoeDistributeDispatchA2SingleTilingFuncImpl(context);
    }

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

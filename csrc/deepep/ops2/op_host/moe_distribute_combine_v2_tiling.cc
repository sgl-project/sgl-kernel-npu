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
 * \file moe_distribute_combine_v2_tiling.cc
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
#include <type_traits>

#include "mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "register/op_def_registry.h"
#include "experiment/platform/platform/platform_infos_def.h"
#include "../op_kernel/moe_distribute_combine_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t ASSIST_INFO_INDEX = 2;
constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
constexpr uint32_t EXPERT_SCALES_INDEX = 4;
constexpr uint32_t TP_SEND_COUNTS_INDEX = 5;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 6;
constexpr uint32_t ACTIVATION_SCALE_INDEX = 7;
constexpr uint32_t WEIGHT_SCALE_INDEX = 8;
constexpr uint32_t GROUP_LIST_INDEX = 9;
constexpr uint32_t SHARED_EXPERT_X_INDEX = 11;
constexpr uint32_t ELASTIC_INFO_INDEX = 12;
constexpr uint32_t ORI_X_INDEX = 13;
constexpr uint32_t CONST_EXPERT_ALPHA_1_INDEX = 14;
constexpr uint32_t CONST_EXPERT_ALPHA_2_INDEX = 15;
constexpr uint32_t CONST_EXPERT_V_INDEX = 16;
constexpr uint32_t OUTPUT_X_INDEX = 0;

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
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTR_OUT_DTYPE_INDEX = 11;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;
constexpr uint32_t ATTR_GROUP_LIST_TYPE_INDEX = 13;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 14;
constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 15;
constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 16;
constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 17;

constexpr uint32_t INT8_COMM_QUANT = 2U;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint64_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
constexpr uint64_t TILINGKEY_IS_SHARE_EXPERT = 1000;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t ASSIST_INFO_DIMS = 1U;
constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
constexpr uint64_t TILING_KEY_INT8_COMM_QUANT_A2 = 100UL;
constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;      // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U;  // numeric representation of AlltoAll

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768L;  // 384 * 2
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t EP_RESTRICT_8 = 8;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 512;

constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t K_MAX = 16;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t ASSIST_NUM_PER_A = 128UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL;  // scale32B + 3*4expandIdx
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;

// A2
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 512;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
const char *K_INNER_DEBUG = "MoeDistributeCombineV2 Tiling Debug";

enum class CommQuantMode : int32_t { NON_QUANT = 0, INT12_QUANT = 1, INT8_QUANT = 2 };
using CommQuantModeType = std::underlying_type<CommQuantMode>::type;
}  // namespace

namespace optiling {
// a2专有
static void PrintA2TilingDataInfo(MoeDistributeCombineV2Info &info)
{
    OP_LOGD(K_INNER_DEBUG, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "tpWorldSize is %u.", info.tpWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId is %u.", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "tpRankId is %u.", info.tpRankId);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType is %u.", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "globalBs is %u.", info.globalBs);
}

static ge::graphStatus MoeDistributeCombineA2CheckAttrAndSetTiling(gert::TilingContext *context,
                                                                   MoeDistributeCombineV2Info &info,
                                                                   int32_t &commQuantMode, const bool isLayered)
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
    auto globalBsPtr = attrs->GetAttrPointer<int>(ATTR_GLOBAL_BS_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int>(ATTR_COMM_QUANT_MODE_INDEX);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "zeroExpertNum is invalid."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "copyExpertNum is invalid."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "constExpertNum is invalid."),
                    return GRAPH_FAILED);

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(K_INNER_DEBUG, "groupEp is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > MAX_EP_WORLD_SIZE_A2 ||
                        *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
                    OP_LOGE(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
                    OP_LOGE(K_INNER_DEBUG, "epRankId is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2 ||
                        *moeExpertNumPtr % *epWorldSizePtr != 0,
                    OP_LOGE(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "expertSharedType is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "sharedExpertRankNum is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "commQuantMode is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(!isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT),
                    OP_LOGE(K_INNER_DEBUG, "commQuantMode is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT) &&
                        *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT),
                    OP_LOGE(K_INNER_DEBUG, "commQuantMode is invalid."), return GRAPH_FAILED);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "xShape is null."), return false);
    int32_t globalBs = *epWorldSizePtr * expertIdStorageShape->GetStorageShape().GetDim(0);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    OP_TILING_CHECK(
        (moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) > INT32_MAX,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum exceeds MAX_INT32."),
        return ge::GRAPH_FAILED);
    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;

    info.zeroExpertNum = static_cast<uint32_t>(zeroExpertNum);
    info.copyExpertNum = static_cast<uint32_t>(copyExpertNum);
    info.constExpertNum = static_cast<uint32_t>(constExpertNum);

    if (*globalBsPtr == 0) {
        info.globalBs = static_cast<uint32_t>(globalBs);
    } else {
        info.globalBs = *globalBsPtr;
    }
    commQuantMode = *commQuantModePtr;
    PrintA2TilingDataInfo(info);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2CheckShapeAndSetTiling(gert::TilingContext *context,
                                                                    MoeDistributeCombineV2Info &info)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);

    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expandXShape is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."),
                    return GRAPH_FAILED);

    // copy expert and const expert
    const gert::StorageShape *oriXStorageShape = context->GetOptionalInputShape(ORI_X_INDEX);
    const gert::StorageShape *constExpertAlpha1StorageShape =
        context->GetOptionalInputShape(CONST_EXPERT_ALPHA_1_INDEX);
    const gert::StorageShape *constExpertAlpha2StorageShape =
        context->GetOptionalInputShape(CONST_EXPERT_ALPHA_2_INDEX);
    const gert::StorageShape *constExpertVStorageShape = context->GetOptionalInputShape(CONST_EXPERT_V_INDEX);

    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(K_INNER_DEBUG, "expandXshape is invalid"), return GRAPH_FAILED);
    uint32_t h = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(h <= 0 || h > MAX_HIDDEN_SIZE_A2 || h % BLOCK_SIZE_A2 != 0,
                    OP_LOGE(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(K_INNER_DEBUG, "expertIdshape is invalid"), return GRAPH_FAILED);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(bs <= 0 || bs > MAX_BATCH_SIZE_A2, OP_LOGE(K_INNER_DEBUG, "batchsize is invalid."),
                    return GRAPH_FAILED);

    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(k <= 0 || k > MAX_K_VALUE_A2, OP_LOGE(K_INNER_DEBUG, "k is invalid."), return GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = static_cast<int32_t>(*constExpertNumPtr);
    uint32_t totalExpertNum = static_cast<uint32_t>(moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum);
    OP_TILING_CHECK(k <= 0 || k > MAX_K_VALUE_A2 || k > totalExpertNum, OP_LOGE(K_INNER_DEBUG, "k is invalid."),
                    return GRAPH_FAILED);

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
                        OP_LOGE(K_INNER_DEBUG, "xActiveMask must be 1-dimension or 2-dimension, but got %ld dim",
                                xActiveMaskDimNums),
                        return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != static_cast<int64_t>(bs),
                        OP_LOGE(K_INNER_DEBUG,
                                "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
                                "expertIds's dim0 is %d",
                                xActiveMaskDim0, bs),
                        return GRAPH_FAILED);

        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
                         (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))),
                        OP_LOGE(K_INNER_DEBUG,
                                "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %lu, "
                                "expertIds's dim1 is %d",
                                xActiveMaskStorageShape->GetStorageShape().GetDim(1), k),
                        return GRAPH_FAILED);
    }

    // copy expert and const expert
    OP_TILING_CHECK(copyExpertNum > 0 && oriXStorageShape == nullptr,
                    OP_LOGE(K_INNER_DEBUG, "oriX must be exist when copyExpertNum > 0"), return GRAPH_FAILED);
    OP_TILING_CHECK(
        constExpertNum > 0 && (oriXStorageShape == nullptr || constExpertAlpha1StorageShape == nullptr ||
                               constExpertAlpha2StorageShape == nullptr || constExpertVStorageShape == nullptr),
        OP_LOGE(K_INNER_DEBUG, "oriX、alpha1、alpha2、V must be exist when constExpertNum > 0"), return GRAPH_FAILED);

    if (oriXStorageShape != nullptr) {
        // 必须是2维
        OP_TILING_CHECK(oriXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(K_INNER_DEBUG, "ori_x must be 2-dimension, but got %lu dim",
                                oriXStorageShape->GetStorageShape().GetDimNum()),
                        return GRAPH_FAILED);

        // shape为(bs, h)
        int64_t oriXDim0 = oriXStorageShape->GetStorageShape().GetDim(0);
        int64_t oriXDim1 = oriXStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(oriXDim0 != static_cast<int64_t>(bs),
                        OP_LOGE(K_INNER_DEBUG, "ori_x's dim0 not equal to bs, ori_x's dim0 = %ld, bs = %ld", oriXDim0,
                                static_cast<int64_t>(bs)),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(oriXDim1 != static_cast<int64_t>(h),
                        OP_LOGE(K_INNER_DEBUG, "ori_x's dim1 not equal to h, ori_x's dim1 = %ld, h = %ld", oriXDim1,
                                static_cast<int64_t>(h)),
                        return GRAPH_FAILED);
    }

    if (constExpertAlpha1StorageShape != nullptr) {
        // 必须是1维
        OP_TILING_CHECK(constExpertAlpha1StorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(K_INNER_DEBUG, "const_expert_alpha_1 must be 1-dimension, but got %lu dim",
                                constExpertAlpha1StorageShape->GetStorageShape().GetDimNum()),
                        return GRAPH_FAILED);

        // shape为(constExpertNum)
        int64_t constExpertAlpha1Dim0 = constExpertAlpha1StorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            constExpertAlpha1Dim0 != *constExpertNumPtr,
            OP_LOGE(K_INNER_DEBUG,
                    "const_expert_alpha_1's dim0 not equal to const_expert_num, const_expert_alpha_1's dim0 = %ld, "
                    "const_expert_num = %ld",
                    constExpertAlpha1Dim0, *constExpertNumPtr),
            return GRAPH_FAILED);
    }

    if (constExpertAlpha2StorageShape != nullptr) {
        // 必须是1维
        OP_TILING_CHECK(constExpertAlpha2StorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(K_INNER_DEBUG, "const_expert_alpha_2 must be 1-dimension, but got %lu dim",
                                constExpertAlpha2StorageShape->GetStorageShape().GetDimNum()),
                        return GRAPH_FAILED);

        // shape为(constExpertNum)
        int64_t constExpertAlpha2Dim0 = constExpertAlpha2StorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            constExpertAlpha2Dim0 != *constExpertNumPtr,
            OP_LOGE(K_INNER_DEBUG,
                    "const_expert_alpha_2's dim0 not equal to const_expert_num, const_expert_alpha_2's dim0 = %ld, "
                    "const_expert_num = %ld",
                    constExpertAlpha2Dim0, *constExpertNumPtr),
            return GRAPH_FAILED);
    }

    if (constExpertVStorageShape != nullptr) {
        // 必须是2维
        OP_TILING_CHECK(constExpertVStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(K_INNER_DEBUG, "const_expert_v must be 2-dimension, but got %lu dim",
                                constExpertVStorageShape->GetStorageShape().GetDimNum()),
                        return GRAPH_FAILED);
        // 必须是2维(constExpertNum, H)
        int64_t constExpertVDim0 = constExpertVStorageShape->GetStorageShape().GetDim(0);
        int64_t constExpertVDim1 = constExpertVStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(constExpertVDim0 != *constExpertNumPtr,
                        OP_LOGE(K_INNER_DEBUG,
                                "const_expert_v's dim0 not equal to const_expert_num, const_expert_v's dim0 = %ld, "
                                "const_expert_num = %ld",
                                constExpertVDim0, *constExpertNumPtr),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(
            constExpertVDim1 != static_cast<int64_t>(h),
            OP_LOGE(K_INNER_DEBUG, "const_expert_v's dim1 not equal to h, const_expert_v's dim1 = %ld, h = %ld",
                    constExpertVDim1, static_cast<int64_t>(h)),
            return GRAPH_FAILED);
    }

    info.isTokenMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", h);
    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2GetPlatformInfoAndSetTiling(gert::TilingContext *context,
                                                                         MoeDistributeCombineV2Info &info)
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
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余报错。
static ge::graphStatus MoeDistributeCombineCheckCommAlg(gert::TilingContext *context, bool &isLayered)
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

static uint64_t MoeDistributeCombineA2CalcTilingKey(gert::TilingContext *context, const bool isLayered,
                                                    const int32_t commQuantMode)
{
    uint64_t tilingKey = TILING_KEY_BASE_A2;
    if (isLayered) {
        tilingKey = TILING_KEY_LAYERED_COMM_A2;
        if (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT)) {
            tilingKey += TILING_KEY_INT8_COMM_QUANT_A2;
        }
    }
    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);
    return tilingKey;
}

static ge::graphStatus MoeDistributeCombineA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeCombineV2 tiling func.");

    // tilingData
    MoeDistributeCombineV2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    MoeDistributeCombineV2Info &info = tilingData->moeDistributeCombineV2Info;

    bool isLayered = false;
    OP_TILING_CHECK(
        MoeDistributeCombineCheckCommAlg(context, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MoeDistributeCombineV2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    int32_t commQuantMode = 0;
    OP_TILING_CHECK(
        MoeDistributeCombineA2CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MoeDistributeCombineV2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeCombineA2CheckAttrAndSetTiling(context, info, commQuantMode, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MoeDistributeCombineV2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "MoeDistributeCombineV2 GetPlatformInfoAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetAicpuBlockDim(mc2tiling::AICPU_BLOCK_DIM_A2);

    uint64_t tilingKey = MoeDistributeCombineA2CalcTilingKey(context, isLayered, commQuantMode);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);
    size_t userWorkspaceSize = info.moeExpertNum * sizeof(uint32_t) * 2U;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkspaceSize;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    std::string algConfig = isLayered ? "BatchWrite=level1:hierarchy" : "BatchWrite=level1:fullmesh";
    uint32_t opType = 18;  // DispatchCombine

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineV2TilingFunc(gert::TilingContext *context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %d",
                            static_cast<ge::DataType>(expandXDesc->GetDataType())),
                    return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeCombineA2TilingFuncImpl(context);
    } else {
        // ret = MoeDistributeCombineA3TilingFuncImpl(context);
    }

    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineV2)
    .Tiling(MoeDistributeCombineV2TilingFunc)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2);
}  // namespace optiling

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ShmemCommMoeDistributeDispatch tiling function implementation file
 * History: 2025-08-20 create ShmemCommMoeDistributeDispatch tiling function implementation file
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "error_log.h"
#include "../op_kernel/shmem_moe_distribute_dispatch_tiling.h"
// #include "shmem_api.h"

using namespace ge;

namespace {
    constexpr uint32_t X_INDEX = 0U;
    constexpr uint32_t EXPERT_IDS_INDEX = 1U;
    constexpr uint32_t SCALES_INDEX = 2U;
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
    constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
    constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
    constexpr uint32_t OUTPUT_EXPAND_IDX_INDEX = 2U;
    constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
    constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
    constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;

    // constexpr uint32_t ATTR_GROUP_EP_INDEX = -1;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 2;
    // constexpr uint32_t ATTR_GROUP_TP_INDEX = -1;
    constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
    constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 4;
    constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 5;
    constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 6;
    constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 7;
    constexpr uint32_t ATTR_QUANT_MODE_INDEX = 8;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 9;
    constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 10;
    constexpr uint32_t ATTR_EXT_INFO_INDEX = 11;

    constexpr uint32_t TWO_DIMS = 2;
    constexpr uint32_t ONE_DIM = 1;
    constexpr uint32_t DYN_SCALE_DIMS = 1;
    constexpr uint32_t EXPAND_IDX_DIMS = 1;
    constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
    constexpr uint64_t INIT_TILINGKEY = 1000;
    constexpr uint32_t ARR_LENGTH = 128;
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
    constexpr uint32_t NO_SCALES = 0;
    constexpr uint32_t STATIC_SCALES = 1;
    constexpr uint32_t DYNAMIC_SCALES = 2;
    constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

    constexpr uint32_t UNQUANT_MODE = 0;
    constexpr uint32_t STATIC_QUANT_MODE = 1;
    constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
    constexpr uint32_t BLOCK_SIZE_A2 = 32;
    constexpr uint32_t MAX_K_VALUE_A2 = 8;
    constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
    constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
    const char *K_INNER_DEBUG = "ShmemMoeDistributeDispatch Tiling Debug";
    const size_t MAX_GROUP_NAME_LENGTH = 128UL;
    const int64_t MAX_EP_WORLD_SIZE = 288;
    const int64_t MAX_TP_WORLD_SIZE = 2;
    const int64_t BS_UPPER_BOUND = 512;

    constexpr uint32_t NUM_10 = 10;
    constexpr uint32_t NUM_100 = 100;
    constexpr uint32_t VERSION_2 = 2;
    constexpr uint32_t HCOMMCNT_2 = 2;
    constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
    constexpr int64_t K_MAX = 8;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

    constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
    constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
}

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, const ShmemMoeDistributeDispatchTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchInfo.tpRankId);
    OP_LOGD(nodeName, "expertSharedType is %u.", tilingData.moeDistributeDispatchInfo.expertSharedType);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchInfo.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchInfo.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchInfo.globalBs);
    OP_LOGD(nodeName, "isQuant is %d.", tilingData.moeDistributeDispatchInfo.isQuant);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.moeDistributeDispatchInfo.totalWinSize);
}

static bool CheckTensorDim(const gert::TilingContext &context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    const gert::StorageShape *xStorageShape = context.GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
        xStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdStorageShape = context.GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIdShape dims must be 2, but current dim num is %lu.",
        expertIdStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));
    // 如果scales不为空进行shape维度检查
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context.GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return false);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "scalesShape dims must be 2, but current dim num is %lu.",
            scalesStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
        OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
    }

    // x_active_mask当前不支持传入
    // const gert::StorageShape *xActiveMaskStorageShape = context.GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    // OP_TILING_CHECK(xActiveMaskStorageShape != nullptr, OP_LOGE(nodeName, "x_active_mask only support input None."),
    //     return false);

    const gert::StorageShape *expandXStorageShape = context.GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandXShape is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandXShape dims must be 2, but current dim num is %lu.",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if (quantMode == DYNAMIC_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context.GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr,
            OP_LOGE(nodeName, "dynamicScalesShape is null."), return false);
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_DIM_NUM,
            OP_LOGE(nodeName, "dynamicScalesShape dims must be %u, but current dim num is %lu.",
            DYNAMIC_SCALE_DIM_NUM, dynamicScalesStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *expandIdxStorageShape = context.GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE(nodeName, "expandIdxShape is null."), return false);
    OP_TILING_CHECK(expandIdxStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "expandIdxShape dims must be 1, but current dim num is %lu.",
        expandIdxStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandIdx dim0 = %ld", expandIdxStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context.GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
        OP_LOGE(nodeName, "expertTokenNumsShape is null."), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "expertTokenNumsShape dims must be 1, but current dim num is %lu.",
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context.GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCountShape is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epRecvCountShape dims must be 1, but current dim num is %lu.",
        epRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context.GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr,
        OP_LOGE(nodeName, "tpRecvCountShape is null."), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "tpRecvCountShape dims must be 1, but current dim num is %lu.",
        tpRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDataType(const gert::TilingContext &context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    auto xDesc = context.GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x datatype is invalid, datatype should be bf16 or float16, but is %d.",
        static_cast<ge::DataType>(xDesc->GetDataType())), return false);

    auto expertIdDesc = context.GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expertId datatype is invalid, datatype should be int32, but is %d.",
        static_cast<ge::DataType>(expertIdDesc->GetDataType())), return false);

    if (isScales) {
        auto scalesDesc = context.GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "scales datatype is invalid, datatype should be float, but is %d.",
            static_cast<ge::DataType>(scalesDesc->GetDataType())), return false);
    }

    auto expandXDesc = context.GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (quantMode != NO_SCALES) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
            OP_LOGE(nodeName, "expandX datatype is invalid, datatype should be int8, but is %d.",
            static_cast<ge::DataType>(expandXDesc->GetDataType())), return false);
    } else {
        OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %d, but is %d.",
            static_cast<ge::DataType>(xDesc->GetDataType()), static_cast<ge::DataType>(expandXDesc->GetDataType())),
            return false);
    }

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context.GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "dynamicScales datatype is invalid, datatype should be float, but is %d.",
            static_cast<ge::DataType>(dynamicScalesDesc->GetDataType())), return false);
    }

    auto expandIdxDesc = context.GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE(nodeName, "expandIdxDesc is null."), return false);
    OP_TILING_CHECK(expandIdxDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expandIdx datatype is invalid, datatype should be int32, but is %d.",
        static_cast<ge::DataType>(expandIdxDesc->GetDataType())), return false);

    auto expertTokenNumsDesc = context.GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."),
        return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(nodeName, "expertTokenNums datatype is invalid, datatype should be int64, but is %d.",
        static_cast<ge::DataType>(expertTokenNumsDesc->GetDataType())), return false);

    auto epRecvCountsDesc = context.GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "epRecvCounts datatype is invalid, datatype should be int32, but is %d.",
        static_cast<ge::DataType>(epRecvCountsDesc->GetDataType())), return false);

    auto tpRecvCountsDesc = context.GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "tpRecvCounts datatype is invalid, datatype should be int32, but is %d.",
        static_cast<ge::DataType>(tpRecvCountsDesc->GetDataType())), return false);
    return true;
}

static bool CheckTensorFormat(const gert::TilingContext &context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    auto xDesc = context.GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "x format is invalid."), return false);

    auto expertIdDesc = context.GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertId format is invalid."), return false);

    if (isScales) {
        auto scalesDesc = context.GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "scales format is invalid."), return false);
    }

    auto expandXDesc = context.GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandX format is invalid."), return false);

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context.GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "dynamicScales format is invalid."), return false);
    }

    auto expandIdxDesc = context.GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE(nodeName, "expandIdxDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandIdxDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandIdx format is invalid."), return false);

    auto expertTokenNumsDesc = context.GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."),
        return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertTokenNums format is invalid."), return false);

    auto epRecvCountsDesc = context.GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epRecvCounts format is invalid."), return false);

    auto tpRecvCountsDesc = context.GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpRecvCounts format is invalid."), return false);
    return true;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext &context, const char *nodeName,
    ShmemMoeDistributeDispatchTilingData &tilingData, std::string &groupEp, std::string &groupTp)
{
    auto attrs = context.GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));
    auto shmemPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXT_INFO_INDEX);

    OP_LOGI(nodeName, "shmemPtr inside tiling is %ld", shmemPtr);

    // 判空
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "expertTokenNumsType is null."),
        return ge::GRAPH_FAILED);
    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support (0, %ld], but got epWorldSize=%ld.",
        MAX_EP_WORLD_SIZE, *epWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        *epWorldSizePtr, *epRankIdPtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertSharedType is invalid, only support 0, but got expertSharedType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
        *epWorldSizePtr, *sharedExpertRankNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX_NUM),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, *moeExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(NO_SCALES)) ||
        (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %u], but got quantMode=%ld.",
        DYNAMIC_SCALES, *quantModePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*sharedExpertNumPtr != 1, OP_LOGE(nodeName,
        "sharedExpertNum only support 1, but got sharedExpertNum=%ld.", *sharedExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
        OP_LOGE(nodeName, "expertTokenNumsType only support 0 or 1, but got expertTokenNumsType=%ld.",
        *expertTokenNumsTypePtr), return ge::GRAPH_FAILED);

    tilingData.moeDistributeDispatchInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeDispatchInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchInfo.expertSharedType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeDispatchInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.moeDistributeDispatchInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchInfo.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    tilingData.moeDistributeDispatchInfo.shmemptr = static_cast<uint64_t>(*shmemPtr);

    OP_LOGI(nodeName, "shmemPtr send to tilingInfo is %ld", tilingData.moeDistributeDispatchInfo.shmemptr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext &context, const char *nodeName,
    ShmemMoeDistributeDispatchTilingData &tilingData, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;
    // 校验ep能否均分共享专家
    OP_TILING_CHECK((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0),
        OP_LOGE(nodeName, "epWorldSize should be divisible by sharedExpertRankNum, but epWorldSize=%u, "
        "sharedExpertRankNum=%u.", epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0, OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %u",
        localMoeExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tpWorldSize > 1) && (localMoeExpertNum > 1), OP_LOGE(nodeName, "Cannot support multi-moeExpert %u "
        "in a rank when tpWorldSize = %u > 1", localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    // 检验epWorldSize是否是8的倍数
    OP_TILING_CHECK(epWorldSize % 8 != 0, OP_LOGE(nodeName,
        "epWorldSize should be divisible by 8, but got epWorldSize = %u.",
        epWorldSize), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0), OP_LOGE(nodeName,
        "epWorldSize should be in the list[8, 16, 32, 64, 128, 144, 256, 288], but got epWorldSize = %u.",
        epWorldSize), return ge::GRAPH_FAILED);
    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context.GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xStorageShape is nullptr."), return ge::GRAPH_FAILED);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
    OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.", BS_UPPER_BOUND,
        xDim0), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.bs = static_cast<uint32_t>(xDim0);
    // 校验globalBS
    auto attrs = context.GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "ShmemMoeDistributeDispatch *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, xDim0, epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext &context, const char *nodeName,
    ShmemMoeDistributeDispatchTilingData &tilingData, const uint32_t quantMode, const bool isScales,
    const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    uint32_t A = 0;
    uint32_t globalBs = tilingData.moeDistributeDispatchInfo.globalBs;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;
    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context.GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xStorageShape is nullptr."), return ge::GRAPH_FAILED);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 != 7168), OP_LOGE(nodeName, "xShape dims1(H) only supports 7168, but got %ld.", xDim1),
        return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.h = static_cast<uint32_t>(xDim1);
    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context.GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr,
        OP_LOGE(nodeName, "expertIdStorageShape is nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to expertIdShape's dim0, "
        "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.", xDim0, expertIdsDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX),
        OP_LOGE(nodeName, "expertIdShape's dim1(k) should be in (0, %ld], but got expertIdShape's dim1=%ld.",
        K_MAX, expertIdsDim1), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.k = static_cast<uint32_t>(expertIdsDim1);
    // 校验scales的维度
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context.GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr,
            OP_LOGE(nodeName, "scalesStorageShape is nullptr."),
            return ge::GRAPH_FAILED);
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
        if (sharedExpertRankNum == 0U) {
            OP_TILING_CHECK(scalesDim0 != moeExpertNum, OP_LOGE(nodeName,
                "scales's dim0 not equal to moeExpertNum, scales's dim0 is %ld, moeExpertNum is %ld.",
                scalesDim0, moeExpertNum), return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK(scalesDim0 != (moeExpertNum + 1), OP_LOGE(nodeName,
                "scales's dim0 not equal to moeExpertNum + 1, scales's dim0 is %ld, moeExpertNum + 1 is %ld.",
                scalesDim0, moeExpertNum + 1), return ge::GRAPH_FAILED);
        }
        OP_TILING_CHECK(xDim1 != scalesDim1, OP_LOGE(nodeName, "scales's dim1 not equal to xShape's dim1, "
            "xShape's dim1 is %ld, scales's dim1 is %ld.", xDim1, scalesDim1), return ge::GRAPH_FAILED);
    }

    if (isSharedExpert && sharedExpertRankNum != 0) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else {     // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }
    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context.GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr,
        OP_LOGE(nodeName, "expandXStorageShape is nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A), OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A*tpWorldSize, "
        "expandX's dim0 is %ld, A*tpWorldSize is %ld.", expandXDim0, tpWorldSize * A), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName, "expandX's dim1 not equal to xShape's dim1, "
        "xShape's dim1 is %ld, expandX's dim1 is %ld.", xDim1, expandXDim1), return ge::GRAPH_FAILED);
    // 校验dynamicScales的维度
    if (quantMode != NO_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context.GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr,
            OP_LOGE(nodeName, "dynamicScalesStorageShape is nullptr."),
            return ge::GRAPH_FAILED);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
            "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, "
            "dynamicScales's dim0 is %ld, A*tpWorldSize is %ld.",
            dynamicScalesDim0, A * tpWorldSize), return ge::GRAPH_FAILED);
    }
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context.GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr,
        OP_LOGE(nodeName, "expandIdxStorageShape is nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandIdxDim0 != expertIdsDim1 * xDim0, OP_LOGE(nodeName,
        "expandIdxDim0 != bs * k, expandIdxDim0 is %ld, bs * k is %ld.", expandIdxDim0, xDim0 * expertIdsDim1),
        return ge::GRAPH_FAILED);
    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context.GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
        OP_LOGE(nodeName, "expertTokenNumsStorageShape is nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1, OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.",
            expertTokenNumsDim0), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(expertTokenNumsDim0 != localMoeExpertNum, OP_LOGE(nodeName,
            "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, localMoeExpertNum), return ge::GRAPH_FAILED);
    }
    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context.GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context.GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr,
        OP_LOGE(nodeName, "epRecvCountStorageShape is nullptr."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr,
        OP_LOGE(nodeName, "tpRecvCountStorageShape is nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = (isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum;
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum * tpWorldSize, "
        "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, tpWorldSize is %ld.",
        epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount is %ld, "
        "tpWorldSize is %ld.", tpRecvCountDim0, tpWorldSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext &context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode),
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void CalTilingKey(uint64_t &tilingKey, const bool isScales, const uint32_t quantMode, const uint32_t tpWorldSize)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    tilingKey += static_cast<uint64_t>((isScales ? NUM_10 : 0));
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        tilingKey += static_cast<uint64_t>(NUM_100);
    }
    return;
}

static void SetHcommCfg(const gert::TilingContext &context, ShmemMoeDistributeDispatchTilingData &tiling,
    const std::string groupEp, const std::string groupTp)
{
    const char *nodeName = context.GetNodeName();
    OP_LOGD(nodeName, "ShmemMoeDistributeDispatch groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_ALL_GATHER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling.mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfigAllGatherStr);
    mc2CcTilingConfig.GetTiling(tiling.mc2CcTiling2);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext &context, const char *nodeName)
{
    size_t *workSpaces = context.GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA3TilingFuncImpl(gert::TilingContext &context)
{
    const char *nodeName = context.GetNodeName();
    ShmemMoeDistributeDispatchTilingData *tilingData = context.GetTilingData<ShmemMoeDistributeDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = NO_SCALES;
    bool isScales = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter ShmemMoeDistributeDispatch tiling check func.");
    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    // 获取scales
    const gert::StorageShape *scalesStorageShape = context.GetOptionalInputShape(SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);
    tilingData->moeDistributeDispatchInfo.isQuant = isScales;
    quantMode = tilingData->moeDistributeDispatchInfo.quantMode;
    // 检查quantMode和scales是否匹配
    OP_TILING_CHECK(quantMode == STATIC_SCALES, OP_LOGE(nodeName, "cannot support static quant now."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
        OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
        static_cast<int32_t>(isScales), quantMode), return ge::GRAPH_FAILED);
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckMoeDistributeDispatch(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);
    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    bool isSharedExpert = true;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchInfo.sharedExpertRankNum;

    uint32_t epRankId = tilingData->moeDistributeDispatchInfo.epRankId;
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isSharedExpert = false;
    }
    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales,
        isSharedExpert, static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);
    // 校验win区大小
    uint16_t defaultWindowSize = 200;
    const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
    uint64_t bs = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.bs);
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.globalBs) / epWorldSize;
    uint64_t actualSize = epWorldSize * maxBs * h * 2UL * 2UL * static_cast<uint64_t>(localMoeExpertNum);
    // if (actualSize > maxWindowSize) {
    //     OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
    //         "ep_worldsize * maxBs * h * 2 * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.", maxBs, h, epWorldSize,
    //         localMoeExpertNum, actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE);
    //     return ge::GRAPH_FAILED;
    // }
    tilingData->moeDistributeDispatchInfo.totalWinSize = maxWindowSize;

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, *tilingData, groupEp, groupTp);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchInfo.tpWorldSize;
    uint64_t tilingKey = INIT_TILINGKEY;
    CalTilingKey(tilingKey, isScales, quantMode, tpWorldSize);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context.SetTilingKey(tilingKey);
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context.SetBlockDim(blockDim);
    tilingData->moeDistributeDispatchInfo.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = MoeDistributeDispatchA3TilingFuncImpl(*context);
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ShmemMoeDistributeDispatch)
    .Tiling(MoeDistributeDispatchTilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatch);
} // namespace optiling

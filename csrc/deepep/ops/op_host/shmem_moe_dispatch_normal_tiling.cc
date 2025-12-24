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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/shmem_moe_dispatch_normal_tiling.h"
#include "tiling_args.h"

using namespace AscendC;
using namespace ge;
using namespace Moe;

namespace {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SEND_TOKENIDX_INDEX = 2U;
constexpr uint32_t PUT_OFFSET_INDEX = 3U;

constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
constexpr uint32_t OUTPUT_WAIT_RECV_COST_INDEX = 3U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 2;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 3;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 4;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 5;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 6;
constexpr uint32_t ATTR_SHMEM_PTR_INDEX = 7;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 384;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;

constexpr uint32_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint32_t TP_WORLD_SIZE_TWO = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 16;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 7168;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr uint64_t TRIPLE = 3;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL;  // scale32B + 3*4expandIdx
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, ShmemMoeDispatchNormalTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDispatchNormalInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDispatchNormalInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDispatchNormalInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDispatchNormalInfo.tpRankId);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDispatchNormalInfo.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDispatchNormalInfo.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDispatchNormalInfo.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDispatchNormalInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDispatchNormalInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDispatchNormalInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDispatchNormalInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDispatchNormalInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.moeDispatchNormalInfo.totalWinSize);
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const uint32_t quantMode,
                           const bool isEnableDiagnose)
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
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

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

    if (isEnableDiagnose) {
        const gert::StorageShape *waitRecvcostStatsStorageShape = context->GetOutputShape(OUTPUT_WAIT_RECV_COST_INDEX);
        OP_TILING_CHECK(waitRecvcostStatsStorageShape == nullptr,
                        OP_LOGE(nodeName, "dispatch waitRecvCostStatsShape is null."), return false);
        OP_TILING_CHECK(waitRecvcostStatsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(nodeName, "dispatch waitRecvCostStatsShape dim must be 1, but current dim num is %lu.",
                                waitRecvcostStatsStorageShape->GetStorageShape().GetDimNum()),
                        return false);
    }

    return true;
}

static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName, const uint32_t quantMode,
                                const bool isEnableDiagnose)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is ."),
                    return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "expertId dataType is invalid, dataType should be int32, but is ."),
                    return false);

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (quantMode != NO_SCALES) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
                        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is."),
                        return false);
    } else {
        OP_TILING_CHECK(
            expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType , but is."),
            return false);
    }

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."), return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is ."),
                        return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is ."),
                    return false);

    if (isEnableDiagnose) {
        auto waitRecvCostStatsDesc = context->GetOutputDesc(OUTPUT_WAIT_RECV_COST_INDEX);
        OP_TILING_CHECK(waitRecvCostStatsDesc == nullptr, OP_LOGE(nodeName, "dispatch waitRecvCostStatsDesc is null."),
                        return false);
        OP_TILING_CHECK(
            waitRecvCostStatsDesc->GetDataType() != ge::DT_INT32,
            OP_LOGE(nodeName, "dispatch waitRecvCostStatsDesc dataType is invalid, dataType should be int32, but is ."),
            return false);
    }

    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName, const uint32_t quantMode,
                              const bool isEnableDiagnose)
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

    if (isEnableDiagnose) {
        auto waitRecvCostStatsDesc = context->GetOutputDesc(OUTPUT_WAIT_RECV_COST_INDEX);
        OP_TILING_CHECK(waitRecvCostStatsDesc == nullptr, OP_LOGE(nodeName, "dispatch waitRecvCostStatsDesc is null."),
                        return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(waitRecvCostStatsDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "dispatch waitRecvCostStatsDesc format is invalid"), return false);
    }

    return true;
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               ShmemMoeDispatchNormalTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);

    // 判空
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
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
    } else {
        OP_TILING_CHECK(
            *tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
                    OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
                            MOE_EXPERT_MAX_NUM, moeExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %u], but got quantMode=%ld.", DYNAMIC_SCALES,
                *quantModePtr),
        return ge::GRAPH_FAILED);

    int64_t moePerRankNum = moeExpertNum / epWorldSize;
    int64_t curDispatchStatusNum = moePerRankNum * epWorldSize;
    OP_TILING_CHECK((curDispatchStatusNum > DISPATCH_STATUS_MAX_SUPPORT_NUM),
                    OP_LOGE(nodeName,
                            "The moe experts num must meet the conditions,"
                            " (moeExpertNum / epWorldSize * epWorldSize <= 1280, but cur is %ld.",
                            curDispatchStatusNum),
                    return ge::GRAPH_FAILED);

    tilingData.moeDispatchNormalInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDispatchNormalInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDispatchNormalInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDispatchNormalInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDispatchNormalInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDispatchNormalInfo.quantMode = static_cast<uint32_t>(*quantModePtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(gert::TilingContext *context, const char *nodeName,
                                  ShmemMoeDispatchNormalTilingData &tilingData, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeDispatchNormalInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDispatchNormalInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDispatchNormalInfo.moeExpertNum;

    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(moeExpertNum % epWorldSize != 0,
                    OP_LOGE(nodeName,
                            "moeExpertNum should be divisible by epWorldSize, "
                            "but moeExpertNum=%u, epWorldSize=%u.",
                            moeExpertNum, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %d", localMoeExpertNum),
                    return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(xDim0 <= 0,
                    OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between >= 1, but got xDim0=%ld.", xDim0),
                    return ge::GRAPH_FAILED);
    tilingData.moeDispatchNormalInfo.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*globalBsPtr <= 0,
                    OP_LOGE(nodeName, "globalBS is invalid, should be positive, but got globalBS=%ld.", *globalBsPtr),
                    return ge::GRAPH_FAILED);
    tilingData.moeDispatchNormalInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    OP_LOGD(nodeName, "globalBs = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0, epWorldSize);

    // 校验shmemPtr
    auto shmemPtrPtr = attrs->GetAttrPointer<uint64_t>(ATTR_SHMEM_PTR_INDEX);
    OP_TILING_CHECK(shmemPtrPtr == nullptr, OP_LOGE(nodeName, "shmemPtrPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "*shmemPtrPtr = %ld\n", *shmemPtrPtr);
    OP_TILING_CHECK(
        *shmemPtrPtr <= 0,
        OP_LOGE(nodeName, "shmemPtr is invalid, should be shmem virtual address, but got shmemPtr=%ld.", *shmemPtrPtr),
        return ge::GRAPH_FAILED);
    tilingData.shmemPtr = static_cast<uint64_t>(*shmemPtrPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
                                        ShmemMoeDispatchNormalTilingData &tilingData, const uint32_t quantMode,
                                        const int64_t localMoeExpertNum)
{
    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDispatchNormalInfo.globalBs;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX),
                    OP_LOGE(nodeName, "xShape dims1(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, xDim1),
                    return ge::GRAPH_FAILED);  // 32字节对齐
    tilingData.moeDispatchNormalInfo.h = static_cast<uint32_t>(xDim1);

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDispatchNormalInfo.moeExpertNum);
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
    tilingData.moeDispatchNormalInfo.k = static_cast<uint32_t>(expertIdsDim1);

    A = globalBs;

    // 校验expandX的维度
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);

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
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckShmemMoeDispatchNormal(gert::TilingContext *context, const char *nodeName,
                                                         const uint32_t quantMode, const bool isEnableDiagnose)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, quantMode, isEnableDiagnose),
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, quantMode, isEnableDiagnose),
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, quantMode, isEnableDiagnose),
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void CalTilingKey(uint64_t &tilingKey, const uint32_t quantMode, const uint32_t tpWorldSize)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_TP_WORLD_SIZE);
    }
    return;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = static_cast<uint64_t>(SYSTEM_NEED_WORKSPACE + WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShmemMoeDispatchNormalA3TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    ShmemMoeDispatchNormalTilingData *tilingData = context->GetTilingData<ShmemMoeDispatchNormalTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    uint32_t quantMode = NO_SCALES;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter ShmemMoeDispatchNormal tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    quantMode = tilingData->moeDispatchNormalInfo.quantMode;

    auto waitRecvcostStatsStorageShape = context->GetOutputShape(OUTPUT_WAIT_RECV_COST_INDEX);
    bool isEnableDiagnose = (waitRecvcostStatsStorageShape != nullptr);
    tilingData->moeDispatchNormalInfo.isEnableDiagnose = isEnableDiagnose;

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        TilingCheckShmemMoeDispatchNormal(context, nodeName, quantMode, isEnableDiagnose) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode,
                                     static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    uint64_t h = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.h);
    uint64_t k = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.k);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDispatchNormalInfo.globalBs) / epWorldSize;

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    uint32_t tpWorldSize = tilingData->moeDispatchNormalInfo.tpWorldSize;
    uint64_t tilingKey = INIT_TILINGKEY;
    CalTilingKey(tilingKey, quantMode, tpWorldSize);
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
    tilingData->moeDispatchNormalInfo.totalUbSize = ubSize;
    tilingData->moeDispatchNormalInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShmemMoeDispatchNormalTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = ShmemMoeDispatchNormalA3TilingFuncImpl(context);
    return ret;
}

struct ShmemMoeDispatchNormalCompileInfo {};
ge::graphStatus TilingParseForShmemMoeDispatchNormal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ShmemMoeDispatchNormal)
    .Tiling(ShmemMoeDispatchNormalTilingFunc)
    .TilingParse<ShmemMoeDispatchNormalCompileInfo>(TilingParseForShmemMoeDispatchNormal);
}  // namespace optiling

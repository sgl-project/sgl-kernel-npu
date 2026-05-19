#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "mc2_tiling_utils.h"
#include "moe_distribute_dispatch_v2_ccu_tiling.h"
#include "../op_kernel/moe_distribute_dispatch_v2_tiling.h"
namespace {
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
constexpr uint32_t ATTR_Y_DATATYPE_INDEX = 13;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 512;
const int64_t COUNT_OFFSET = 512;
const uint64_t BUFFER_NUM = 2;
const uint64_t EVEN_ALIGN = 2;
const uint64_t COMM_ALIGN = 512U;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

constexpr uint32_t NUM_0 = 0;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_100 = 100;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

constexpr uint32_t EP_WORLD_SIZE_FOUR = 4;
constexpr uint32_t EP_WORLD_SIZE_TWO = 2;

constexpr uint64_t MX_BLOCK_SIZE = 32U;
constexpr uint64_t UB_ALIGN = 32U;
constexpr uint64_t MX_PAD_ALIGN = 256U;
constexpr uint64_t PERGROUP_BLOCK_SIZE = 128U;
constexpr uint64_t PERGROUP_PAD_ALIGN = 128U;
constexpr uint64_t STATUS_SIZE = 512U;

constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
constexpr uint64_t HIF8_SCALE_DIM_0 = 1;
constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;

constexpr uint32_t MAX_UINT32 = 4294967295;

const std::string OP_NAME = "MoeDistributeDispatchA5";

constexpr uint64_t TILING_KEY_CCU_TYPE = 60000;
constexpr uint32_t TILINGKEY_SCALES = 10;

// V1、V2对外接口约束不一致，按版本区分范围校验的变量名
constexpr int64_t MAX_MOE_EXPERT_NUM_V2 = 1024;
constexpr int64_t MAX_MOE_EXPERT_NUM_V1 = 512;
const int64_t H_V1 = 7168;
const int64_t H_UPPER_BOUND_V2 = 8192;
const int64_t H_LOWER_BOUND_V2 = 1024;
const int64_t MAX_EP_WORLD_SIZE_V1 = 288;
const int64_t MAX_EP_WORLD_SIZE_V2 = 768;
const int64_t MIN_EP_WORLD_SIZE_V2 = 2;
const int64_t MAX_SHARED_EXPERT_NUM_V2 = 4;
const int64_t MIN_SHARED_EXPERT_NUM_V2 = 0;
const int64_t K_UPPER_BOUND_V2 = 8;
}  // namespace

namespace optiling {
template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchV2Info.expertShardType);
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
    OP_LOGD(nodeName, "expertTokenNumsType is %u.", tilingData.moeDistributeDispatchV2Info.expertTokenNumsType);

    OP_LOGD(nodeName, "scalesCol is %lu.", tilingData.moeDistributeDispatchV2Info.scalesCol);
    OP_LOGD(nodeName, "scalesRow is %lu.", tilingData.moeDistributeDispatchV2Info.scalesRow);
    OP_LOGD(nodeName, "scalesTypeSize is %u.", tilingData.moeDistributeDispatchV2Info.scalesTypeSize);
    OP_LOGD(nodeName, "scalesCount is %lu.", tilingData.moeDistributeDispatchV2Info.scalesCount);
}

inline ge::graphStatus CheckTpAttrs(const char *nodeName, const int64_t tpWorldSize, const int64_t tpRankId,
                                    const char *groupTpPtr, std::string &groupTp)
{
    OP_TILING_CHECK((tpWorldSize < 0) || (tpWorldSize > MAX_TP_WORLD_SIZE),
                    OP_LOGE(nodeName, "The valid range of tpWorldSize is [0, %ld], but actually got tpWorldSize=%ld.",
                            MAX_TP_WORLD_SIZE, tpWorldSize),
                    return ge::GRAPH_FAILED);
    if (tpWorldSize > 1) {
        OP_TILING_CHECK((tpRankId < 0) || (tpRankId >= tpWorldSize),
                        OP_LOGE(nodeName, "The valid range of tpRankId is [0, %ld), but actually got tpRankId=%ld.",
                                tpWorldSize, tpRankId),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr), OP_LOGE(nodeName, "The groupTpPtr is null."), return ge::GRAPH_FAILED);
        uint64_t len = strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH);
        OP_TILING_CHECK(
            (len == 0) || (len == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "Valid length of groupTp must be in the range (0, %lu), but got strnlen(groupTp)=%lu.",
                    MAX_GROUP_NAME_LENGTH, len),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            tpRankId != 0,
            OP_LOGE(nodeName, "The expected value of tpRankId is 0 in NoTp mode, but the actual value is %ld.",
                    tpRankId),
            return ge::GRAPH_FAILED);
    }
    groupTp = std::string(groupTpPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpAttrs(const char *nodeName, const int64_t epWorldSize, const int64_t epRankId,
                                    const char *groupEpPtr, std::string &groupEp)
{
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(nodeName, "The groupEpPtr is null."), return ge::GRAPH_FAILED);
    uint64_t len = strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH);
    OP_TILING_CHECK(
        (len == 0) || (len == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName,
                "Valid length of groupEp must be in the range (0, %lu), but actually got strnlen(groupEp)=%lu.",
                MAX_GROUP_NAME_LENGTH, len),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE_V2) || (epWorldSize > MAX_EP_WORLD_SIZE_V2),
                    OP_LOGE(nodeName, "The valid range of epWorldSize is [%ld, %ld], but actually got epWorldSize=%ld.",
                            MIN_EP_WORLD_SIZE_V2, MAX_EP_WORLD_SIZE_V2, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((epRankId < 0) || (epRankId >= epWorldSize),
                    OP_LOGE(nodeName, "The valid range of epRankId is [0, %ld), but actually got epRankId=%ld.",
                            epWorldSize, epRankId),
                    return ge::GRAPH_FAILED);
    groupEp = std::string(groupEpPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckQuantAndExpertAttrs(const char *nodeName, const int64_t sharedExpertNum,
                                                const int64_t quantMode, const int64_t expertTokenNumsType)
{
    OP_TILING_CHECK((quantMode < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
                        (quantMode >= static_cast<int64_t>(QuantModeA5::BUTT)),
                    OP_LOGE(nodeName, "The valid range of quantMode is [0, %ld), but actually got quantMode=%ld.",
                            static_cast<int64_t>(QuantModeA5::BUTT), quantMode),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertNum < MIN_SHARED_EXPERT_NUM_V2) || (sharedExpertNum > MAX_SHARED_EXPERT_NUM_V2),
                    OP_LOGE(nodeName, "The valid range of sharedExpertNum is [%ld, %ld], but the actual value is %ld.",
                            MIN_SHARED_EXPERT_NUM_V2, MAX_SHARED_EXPERT_NUM_V2, sharedExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (expertTokenNumsType != 0) && (expertTokenNumsType != 1),
        OP_LOGE(nodeName, "The expected value of expertTokenNumsType is 0 or 1, but the actual value is %ld.",
                expertTokenNumsType),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckSharedExpertAttrs(const char *nodeName, const int64_t expertShard,
                                              const int64_t sharedExpertRankNum, const int64_t epWorldSize)
{
    OP_TILING_CHECK(
        expertShard != 0,
        OP_LOGE(nodeName, "The expected value of expertShardType is 0, but the actual value is %ld.", expertShard),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
        OP_LOGE(nodeName,
                "The valid range of sharedExpertRankNum is [0, %ld), but actually got sharedExpertRankNum=%ld.",
                epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckMoeExpertAttrs(const char *nodeName, const int64_t moeExpertNum)
{
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MAX_MOE_EXPERT_NUM_V2),
                    OP_LOGE(nodeName, "The valid range of moeExpertNum is (0, %ld], but actually got moeExpertNum=%ld.",
                            MAX_MOE_EXPERT_NUM_V2, moeExpertNum),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputDataType(const gert::TilingContext *context, const char *nodeName,
                                           const int64_t quantMode)
{
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "Failed to get expandX datatype."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == static_cast<int64_t>(QuantModeA5::NON_QUANT)) &&
            (NON_QUANT_DTYPE.find(static_cast<ge::DataType>(expandXDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
        OP_LOGE(nodeName,
                "Invalid expandX datatype for quantMode %ld. Only bf16/fp16/hif8/fp8_e4m3fn/fp8_e5m2 is supported.",
                quantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == static_cast<int64_t>(QuantModeA5::STATIC_QUANT)) &&
            (expandXDesc->GetDataType() != ge::DT_HIFLOAT8) && (expandXDesc->GetDataType() != ge::DT_INT8),
        OP_LOGE(nodeName, "Invalid expandX datatype for quantMode %ld. Only int8/hif8 is supported", quantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) &&
            (expandXDesc->GetDataType() != ge::DT_INT8) && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
            (expandXDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE(nodeName, "Invalid expandX datatype for quantMode %ld. Only int8/fp8_e4m3fn/fp8_e5m2 is supported",
                quantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((quantMode == static_cast<int64_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) ||
         (quantMode == static_cast<int64_t>(QuantModeA5::MX_QUANT))) &&
            (expandXDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE(nodeName, "Invalid expandX datatype for quantMode %ld. Only fp8_e4m3fn/fp8_e5m2 is supported",
                quantMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline void SetSharedExpertNum(MoeDistributeDispatchV2TilingData &tilingData, const uint32_t sharedExpertRankNum,
                               const uint32_t sharedExpertNum)
{
    if ((sharedExpertRankNum == NUM_0) && (sharedExpertNum == NUM_1)) {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = NUM_0;
    } else {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(sharedExpertNum);
    }
}

inline void SetEpTpInfo(MoeDistributeDispatchV2TilingData &tilingData, const uint32_t epWorldSize,
                        const uint32_t tpWorldSize, const uint32_t epRankId, const uint32_t tpRankId)
{
    tilingData.moeDistributeDispatchV2Info.epWorldSize = epWorldSize;
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = tpWorldSize;
    tilingData.moeDistributeDispatchV2Info.epRankId = epRankId;
    tilingData.moeDistributeDispatchV2Info.tpRankId = tpRankId;
}

inline void SetExpertInfo(MoeDistributeDispatchV2TilingData &tilingData, const uint32_t expertShard,
                          const uint32_t sharedExpertRankNum, const uint32_t moeExpertNum)
{
    tilingData.moeDistributeDispatchV2Info.expertShardType = expertShard;
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = sharedExpertRankNum;
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = moeExpertNum;
}

static ge::graphStatus GetContextAttrs(const gert::TilingContext *context, const char *nodeName,
                                       MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp,
                                       std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "The attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
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

    // 判空
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "The epWorldSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "The tpWorldSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "The epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "The tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "The expertShardPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertRankNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "The moeExpertNumPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "The quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertNum is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "The expertTokenNumsType is null."),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(
        (CheckEpAttrs(nodeName, *epWorldSizePtr, *epRankIdPtr, groupEpPtr, groupEp) != ge::GRAPH_SUCCESS) ||
            (CheckTpAttrs(nodeName, *tpWorldSizePtr, *tpRankIdPtr, groupTpPtr, groupTp) != ge::GRAPH_SUCCESS),
        OP_LOGE(nodeName, "Check EP or TP attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((CheckSharedExpertAttrs(nodeName, *expertShardPtr, *sharedExpertRankNumPtr, *epWorldSizePtr) !=
                     ge::GRAPH_SUCCESS) ||
                        (CheckMoeExpertAttrs(nodeName, *moeExpertNumPtr) != ge::GRAPH_SUCCESS) ||
                        (CheckQuantAndExpertAttrs(nodeName, *sharedExpertNumPtr, *quantModePtr,
                                                  *expertTokenNumsTypePtr) != ge::GRAPH_SUCCESS),
                    OP_LOGE(nodeName, "Check quant or expert attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDataType(context, nodeName, *quantModePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckOutputDataType failed."), return ge::GRAPH_FAILED);
    SetEpTpInfo(tilingData, static_cast<uint32_t>(*epWorldSizePtr), static_cast<uint32_t>(*tpWorldSizePtr),
                static_cast<uint32_t>(*epRankIdPtr), static_cast<uint32_t>(*tpRankIdPtr));
    SetExpertInfo(tilingData, static_cast<uint32_t>(*expertShardPtr), static_cast<uint32_t>(*sharedExpertRankNumPtr),
                  static_cast<uint32_t>(*moeExpertNumPtr));
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    SetSharedExpertNum(tilingData, static_cast<uint32_t>(*sharedExpertRankNumPtr),
                       static_cast<uint32_t>(*sharedExpertNumPtr));
    return ge::GRAPH_SUCCESS;
}

inline uint32_t CheckQuantModeAndExpandXType(const gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    QuantModeA5 quantMode = static_cast<QuantModeA5>(*quantModePtr);
    auto modeToFind = QUANT_MODE_MAP.find({quantMode, static_cast<ge::DataType>(expandXDesc->GetDataType())});
    OP_TILING_CHECK(modeToFind == QUANT_MODE_MAP.end(),
                    OP_LOGE(nodeName, "Failed to find real mode for quantMode=%u ", static_cast<uint32_t>(quantMode)),
                    return static_cast<uint32_t>(RealModeA5::INVALID_MODE));
    OP_LOGD(nodeName, "quantMode=%u, get realMode=%u\n", static_cast<uint32_t>(quantMode),
            static_cast<uint32_t>(modeToFind->second));
    return static_cast<uint32_t>(modeToFind->second);
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName, bool isScales,
                                               const uint32_t quantMode)
{
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)),
                    OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u.", quantMode),
                    return ge::GRAPH_FAILED);
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
                        ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16)),
                    OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u", quantMode),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
                        ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) ||
                         (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN)),
                    OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u", quantMode),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)),
                    OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u.", quantMode),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpWorldSizeV1(const char *nodeName, uint32_t epWorldSize, uint32_t sharedExpertRankNum)
{
    // 校验ep能否均分共享专家
    OP_TILING_CHECK((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0),
                    OP_LOGE(nodeName,
                            "epWorldSize should be non-zero and divisible by sharedExpertRankNum, but epWorldSize=%u, "
                            "sharedExpertRankNum=%u.",
                            epWorldSize, sharedExpertRankNum),
                    return ge::GRAPH_FAILED);

    // Only the value 4 or 2 is supported currently
    if ((epWorldSize == EP_WORLD_SIZE_FOUR) || (epWorldSize == EP_WORLD_SIZE_TWO)) {
        OP_LOGD(nodeName, "epWorldSize=%u, skip validation\n", epWorldSize);
    } else {
        // 检验epWorldSize是否是8的倍数
        OP_TILING_CHECK(epWorldSize % 8 != 0,
                        OP_LOGE(nodeName, "epWorldSize must be a multiple of 8, but got epWorldSize=%u.", epWorldSize),
                        return ge::GRAPH_FAILED);

        OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
                        OP_LOGE(nodeName,
                                "The value of epWorldSize must be in the list[8, 16, 32, 64, 128, 144, 256, 288], but "
                                "got epWorldSize=%u.",
                                epWorldSize),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedAttrs(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U)) ||
                        ((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U)),
                    OP_LOGE(nodeName,
                            "sharedExpertRankNum and sharedExpertNum must both be zero or both non-zero, but got "
                            "sharedExpertRankNum=%u, sharedExpertNum=%u",
                            sharedExpertRankNum, sharedExpertNum),
                    return ge::GRAPH_FAILED);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(
            ((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName,
                    "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
                    "sharedExpertNum=%u.",
                    sharedExpertRankNum, sharedExpertNum),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                  uint32_t &localMoeExpertNum, const bool isTokenMask)
{
    // nodeName已在调用处判空
    const char *nodeName = context->GetNodeName();
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE(nodeName,
                            "The moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
                            "but got moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.",
                            moeExpertNum, epWorldSize, sharedExpertRankNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "The localMoeExpertNum is invalid, localMoeExpertNum=%u", localMoeExpertNum),
                    return ge::GRAPH_FAILED);
    // tpWorldSize 当前仅支持1
    OP_TILING_CHECK(
        tpWorldSize != 1,
        OP_LOGE(nodeName, "The tpWorldSize must be 1 in current version, but got tpWorldSize=%u.", tpWorldSize),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckSharedAttrs(nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckSharedAttrs failed."), return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
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
    OP_LOGD(nodeName, "MoeDistributeDispatch *globalBsPtr=%ld, bs=%ld, epWorldSize=%u\n", *globalBsPtr, xDim0,
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
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isTokenMask),
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

inline ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
                                              const MoeDistributeDispatchV2TilingData &tilingData,
                                              const int64_t scalesDim0, const int64_t scalesDim1)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (sharedExpertRankNum == 0U) {
        OP_TILING_CHECK(
            scalesDim0 != moeExpertNum,
            OP_LOGE(nodeName, "scales's dim0 not equal to moeExpertNum, scales's dim0=%ld, moeExpertNum=%ld.",
                    scalesDim0, moeExpertNum),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scalesDim0 != (moeExpertNum + sharedExpertNum),
                        OP_LOGE(nodeName,
                                "scales's dim0 not equal to moeExpertNum + sharedExpertNum, scales's dim0=%ld, "
                                "(moeExpertNum + sharedExpertNum)=%ld.",
                                scalesDim0, moeExpertNum + sharedExpertNum),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1,
                    OP_LOGE(nodeName,
                            "scales's dim1 not equal to xShape's dim1, "
                            "xShape's dim1=%ld, scales's dim1=%ld.",
                            xDim1, scalesDim1),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAndSetScalesInfo(const gert::TilingContext *context, const char *nodeName,
                                             MoeDistributeDispatchV2TilingData &tilingData, bool isScales,
                                             const uint32_t quantMode)
{
    // 校验scales的维度
    // bs and h have been set in CheckAttrs
    uint32_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            // realMode 1 or 9
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(RealModeA5::STATIC_SCALES)) && (scalesDim0 != h) &&
                                (scalesDim0 != STATIC_SCALE_DIM_0),
                            OP_LOGE(nodeName, "The expected scalesDim0 is %u or %lu in static quant, but got %ld", h,
                                    STATIC_SCALE_DIM_0, scalesDim0),
                            return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(RealModeA5::HIF8_SCALES)) && (scalesDim0 != HIF8_SCALE_DIM_0),
                OP_LOGE(nodeName,
                        "The expected scalesDim0 is 1 when expandX datatype is hif8 in static quant, but got %ld",
                        scalesDim0),
                return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (quantMode == static_cast<uint32_t>(RealModeA5::NO_SCALES)) {
            OP_TILING_CHECK(
                scalesDim0 != bs,
                OP_LOGE(nodeName, "The expected scalesDim0 is %u when scales is not null in non-quant, but got %ld", bs,
                        scalesDim0),
                return ge::GRAPH_FAILED);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(
                CheckTwoDimScalesShape(context, nodeName, tilingData, scalesDim0, scalesDim1) != ge::GRAPH_SUCCESS,
                OP_LOGE(nodeName, "CheckTwoDimScalesShape failed."), return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    tilingData.moeDistributeDispatchV2Info.scalesRow = scalesRow;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scalesCol;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scalesCount;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scalesTypeSize;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpandXShape(const gert::TilingContext *context, const char *nodeName,
                                         const MoeDistributeDispatchV2TilingData &tilingData, uint32_t A)
{
    // H has been set in CheckAndSetH
    const uint32_t xDim1 = tilingData.moeDistributeDispatchV2Info.h;
    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
                    OP_LOGE(nodeName,
                            "expandX's dim0 not greater than or equal to A*tpWorldSize, "
                            "expandX's dim0=%ld, A*tpWorldSize=%ld.",
                            expandXDim0, tpWorldSize * A),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expandXDim1 != static_cast<int64_t>(xDim1),
                    OP_LOGE(nodeName,
                            "expandX's dim1 not equal to xShape's dim1, "
                            "xShape's dim1=%u, expandX's dim1=%ld.",
                            xDim1, expandXDim1),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckDynamicScalesShape(const gert::TilingContext *context, const char *nodeName,
                                               const MoeDistributeDispatchV2TilingData &tilingData,
                                               const uint32_t quantMode, uint32_t A)
{
    // 校验dynamicScales的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize,
                        OP_LOGE(nodeName,
                                "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, "
                                "dynamicScales's dim0=%ld, A*tpWorldSize=%ld.",
                                dynamicScalesDim0, A * tpWorldSize),
                        return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx
        if (quantMode != static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 =
                static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) &&
                    (dynamicScalesDim1 !=
                     Mc2TilingUtils::CeilAlign(static_cast<uint64_t>(CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN)),
                OP_LOGE(nodeName,
                        "dynamicScales's dim1 should be equal to %lu and even when quantMode=%u, but got %lu.",
                        CeilDiv(h, MX_BLOCK_SIZE), quantMode, dynamicScalesDim1),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (dynamicScalesDim1 != CeilDiv(h, PERGROUP_BLOCK_SIZE)) &&
                    (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                        CeilDiv(h, PERGROUP_BLOCK_SIZE), quantMode, dynamicScalesDim1),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpandIdxAndMaskShape(const gert::TilingContext *context, const char *nodeName,
                                                  const int64_t xDim0, const int64_t expertIdsDim1)
{
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    const int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandIdxDim0 < expertIdsDim1 * xDim0,
                    OP_LOGE(nodeName, "expandIdxDim0 < bs * k, expandIdxDim0=%ld, (bs * k)=%ld.", expandIdxDim0,
                            xDim0 * expertIdsDim1),
                    return ge::GRAPH_FAILED);
    // 校验xActiveMask的维度
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    if (xActiveMaskStorageShape != nullptr) {
        const int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != xDim0,
                        OP_LOGE(nodeName, "The dim0 of xActiveMask should be equal to Bs=%ld, but actually got %ld.",
                                xDim0, xActiveMaskDim0),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpertTokenNumsShape(const gert::TilingContext *context, const char *nodeName,
                                                 const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1,
                        OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.", expertTokenNumsDim0),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(expertTokenNumsDim0 != localMoeExpertNum,
                        OP_LOGE(nodeName,
                                "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0=%ld, "
                                "localMoeExpertNum=%ld.",
                                expertTokenNumsDim0, localMoeExpertNum),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpTpTecvTensorShape(const gert::TilingContext *context, const char *nodeName,
                                                const MoeDistributeDispatchV2TilingData &tilingData,
                                                const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 校验epRecvCount和tpRecvCount的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCount is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "tpRecvCount is null."),
                    return ge::GRAPH_FAILED);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = (isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum;
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(
        epRecvCountDim0 < epRecvCount,
        OP_LOGE(nodeName,
                "The dimension 0 of epRecvCount should not be less than epWorldSize * localMoeExpertNum * tpWorldSize, "
                "but dimension 0 of epRecvCount=%ld, epWorldSize=%ld, localMoeExpertNum=%ld, tpWorldSize=%ld.",
                epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tpRecvCountDim0 != tpWorldSize,
        OP_LOGE(nodeName,
                "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount=%ld, "
                "tpWorldSize=%ld.",
                tpRecvCountDim0, tpWorldSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetH(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData,
                                    const int64_t xDim1)
{
    // 校验输入x的维度1并设h, bs已校验过
    OP_TILING_CHECK((xDim1 < H_LOWER_BOUND_V2) || (xDim1 > H_UPPER_BOUND_V2),
                    OP_LOGE(nodeName, "xShape dims1(H) should be in the range [%ld, %ld], but got %ld.",
                            H_LOWER_BOUND_V2, H_UPPER_BOUND_V2, xDim1),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertIdsAndSetK(const gert::TilingContext *context, const char *nodeName,
                                             MoeDistributeDispatchV2TilingData &tilingData, const int64_t xDim0)
{
    // 校验expert_id的维度并设k
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."),
                    return ge::GRAPH_FAILED);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    const int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName,
                            "xShape's dim0 not equal to expertIdShape's dim0, "
                            "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.",
                            xDim0, expertIdsDim0),
                    return ge::GRAPH_FAILED);
    const int64_t maxK = K_UPPER_BOUND_V2;
    OP_TILING_CHECK(
        (expertIdsDim1 <= 0) || (expertIdsDim1 > maxK) || (expertIdsDim1 > moeExpertNum),
        OP_LOGE(
            nodeName,
            "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum=%ld)], but got expertIdShape's dim1=%ld.",
            maxK, moeExpertNum, expertIdsDim1),
        return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context,
                                        MoeDistributeDispatchV2TilingData &tilingData, const bool isSharedExpert,
                                        const int64_t localMoeExpertNum)
{
    // nodeName已在调用处判空
    const char *nodeName = context->GetNodeName();
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t A = 0;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(CheckAndSetH(nodeName, tilingData, xDim1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckAndSetH failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertIdsAndSetK(context, nodeName, tilingData, xDim0) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckExpertIdsAndSetK failed."), return ge::GRAPH_FAILED);
    int64_t k = tilingData.moeDistributeDispatchV2Info.k;
    if (isSharedExpert) {  // 本卡为共享专家
        uint32_t rankNumPerSharedExpert = 0;
        uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
        uint32_t maxBs = globalBs / epWorldSizeU32;
        uint32_t maxSharedGroupNum = 0;
        if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) {
            rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
            maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
        }
        A = maxBs * maxSharedGroupNum;
    } else {  // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, k);
    }
    // 校验expandX、dynamicScales和expandIdx、epSendCount的维度
    OP_TILING_CHECK(CheckExpandXShape(context, nodeName, tilingData, A) != ge::GRAPH_SUCCESS ||
                        CheckDynamicScalesShape(context, nodeName, tilingData, quantMode, A) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check expandX or dynamicScales shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckExpandIdxAndMaskShape(context, nodeName, xDim0, k) != ge::GRAPH_SUCCESS ||
            CheckExpertTokenNumsShape(context, nodeName, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check expandIdx or expertTokenNums shape failed."), return ge::GRAPH_FAILED);
    // 校验epRecvCount和tpRecvCount的维度
    OP_TILING_CHECK(
        CheckEpTpTecvTensorShape(context, nodeName, tilingData, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckEpTpTecvTensorShape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                    uint32_t localMoeExpertNum)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    uint64_t alignCnt = UB_ALIGN;
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint64_t alignedH =
        Mc2TilingUtils::CeilAlign(static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h), alignCnt);

    // Non-null check in CheckQuantModeAndScales
    auto xDesc = context->GetInputDesc(X_INDEX);
    workSpaces[0] =
        SYSTEM_NEED_WORKSPACE + epWorldSize * sizeof(uint64_t) * BUFFER_NUM * BUFFER_NUM +
        epWorldSize *
            (COUNT_OFFSET +
             maxBs * Mc2TilingUtils::CeilAlign(alignedH * GetSizeByDataType(xDesc->GetDataType()), COMM_ALIGN) *
                 localMoeExpertNum);
    tilingData.moeDistributeDispatchV2Info.totalWinSize = Mc2TilingUtils::GetMaxWindowSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommTiling(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                     std::string &groupEp, std::string &groupTp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
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
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckCommAttrs(const char *nodeName, const MoeDistributeDispatchV2TilingData &tilingData,
                                      uint32_t localMoeExpertNum)
{
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t alignCnt = UB_ALIGN;
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t aivNum = tilingData.moeDistributeDispatchV2Info.aivNum;
    if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        alignCnt = MX_PAD_ALIGN;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) {
        alignCnt = PERGROUP_PAD_ALIGN;
    }
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    uint64_t alignedH = Mc2TilingUtils::CeilAlign(h, alignCnt);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    uint64_t actualSize = aivNum * STATUS_SIZE + 2UL * epWorldSize *
                                                     (maxBs * Mc2TilingUtils::CeilAlign(alignedH * 2UL, COMM_ALIGN) *
                                                          static_cast<uint64_t>(localMoeExpertNum) +
                                                      COUNT_OFFSET);
    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, maxBs=%lu, epWorldSize=%lu, localMoeExpertNum=%u,"
                "h=%lu, alignedH=%lu,"
                "aivNum * STATUS_SIZE + 2 * epWorldSize * (maxBs * Align512(alignedH * 2) * localMoeExpertNum + "
                "COUNT_OFFSET)=%luMB, HCCL_BUFFSIZE=%luMB.",
                maxBs, epWorldSize, localMoeExpertNum, h, alignedH, actualSize / MB_SIZE + 1UL,
                maxWindowSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline void SetPlatformInfo(gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                            const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData.moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
}

ge::graphStatus MoeDistributeDispatchTilingImpl(gert::TilingContext *context)
{
    // Tiling implementation
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(nodeName, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "Start MoeDistributeDispatch tiling.");
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "", groupTp = "";
    uint32_t localMoeExpertNum = 1;
    // Attrs
    OP_TILING_CHECK(GetContextAttrs(context, nodeName, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    // Calc real quantMode
    uint32_t quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;
    uint32_t realMode = CheckQuantModeAndExpandXType(context, nodeName);
    OP_TILING_CHECK(realMode == static_cast<uint32_t>(RealModeA5::INVALID_MODE),
                    OP_LOGE(nodeName, "CheckQuantModeAndExpandXType failed."), return ge::GRAPH_FAILED);
    // X active mask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isTokenMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask = isTokenMask;
    // Scales and Quant
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    OP_TILING_CHECK(CheckQuantModeAndScales(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
                            static_cast<int32_t>(isScales), quantMode),
                    return ge::GRAPH_FAILED);
    // Check Attrs
    OP_TILING_CHECK(CheckAttrs(context, *tilingData, localMoeExpertNum, isTokenMask) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    bool isSharedExpert = (tilingData->moeDistributeDispatchV2Info.epRankId >=
                           tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum)
                              ? false
                              : true;

    // Shape
    OP_TILING_CHECK(CheckTensorShape(context, *tilingData, isSharedExpert, static_cast<int64_t>(localMoeExpertNum)) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetScalesInfo(context, nodeName, *tilingData, isScales, realMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check scales info failed."), return ge::GRAPH_FAILED);
    // Comm
    OP_TILING_CHECK(CheckCommAttrs(nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckCommAttrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetCommTiling(context, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "SetCommTiling failed."), return ge::GRAPH_FAILED);
    // Workspace
    OP_TILING_CHECK(SetWorkSpace(context, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "SetWorkSpace failed."), return ge::GRAPH_FAILED);
    uint64_t tilingKey = TILING_KEY_CCU_TYPE;
    tilingKey += static_cast<uint64_t>(quantMode);
    if (isScales) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES);
    }
    context->SetTilingKey(tilingKey);
    // Platform
    SetPlatformInfo(context, *tilingData, nodeName);
    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD(nodeName, "Finish MoeDistributeDispatch tiling.");
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

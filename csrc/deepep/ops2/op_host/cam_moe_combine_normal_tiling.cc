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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/cam_moe_combine_normal_tiling.h"
#include "tiling_args.h"

using namespace AscendC;
using namespace ge;
using namespace Moe;

namespace {
class Mc2TilingUtils
{
public:
    static uint64_t GetMaxWindowSize()
    {
        uint16_t defaultWindowSize = 200;
        const char *hcclBuffSize = getenv("DEEPEP_HCCL_BUFFSIZE") == nullptr ? "HCCL_BUFFSIZE" : "DEEPEP_HCCL_BUFFSIZE";
        if (getenv(hcclBuffSize) == nullptr) {
            OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
        } else {
            try {
                std::string envStr(getenv(hcclBuffSize));
                defaultWindowSize = std::stoi(envStr);
            } catch (const std::invalid_argument &ia) {
                OP_LOGE("", "Invalid argument when parsing HCCL_BUFFSIZE: %s", ia.what());
            } catch (const std::out_of_range &oor) {
                OP_LOGE("", "Out of range when parsing HCCL_BUFFSIZE: %s", oor.what());
            }
        }
        const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
        OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
        return maxWindowSize;
    }
};
constexpr uint32_t RECV_X_INDEX = 0;
constexpr uint32_t TOKEN_SRC_INFO_INDEX = 1;
constexpr uint32_t EP_RECV_COUNTS_INDEX = 2;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 3;
constexpr uint32_t TOPK_IDX_INDEX = 4;
constexpr uint32_t TP_RECV_COUNTS_INDEX = 5;

constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_SEND_COST_INDEX = 1;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 3;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 4;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 5;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 6;
constexpr uint32_t ATTR_REAL_MAX_BS_INDEX = 7;
constexpr uint32_t ATTR_MAX_ROUND_INDEX = 8;
constexpr uint32_t ATTR_PER_ROUND_TOKENS_INDEX = 9;

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;      // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U;  // numeric representation of ReduceScatter

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 384;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 8000;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 16;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 7168;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SCALE_RECV_IDX_BUFFER = 44UL;  // scale32B + 3*4 src info
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr int64_t DISPATCH_STATUS_MAX_SUPPORT_NUM = 1280UL;

enum class CommQuantMode : int32_t { NON_QUANT = 0, INT12_QUANT = 1, INT8_QUANT = 2 };
using CommQuantModeType = std::underlying_type<CommQuantMode>;
}  // namespace

namespace optiling {

// a3专有
static void PrintTilingDataInfo(const char *nodeName, CamMoeCombineNormalTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.camMoeCombineNormalInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.camMoeCombineNormalInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.camMoeCombineNormalInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.camMoeCombineNormalInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.camMoeCombineNormalInfo.expertShardType);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.camMoeCombineNormalInfo.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.camMoeCombineNormalInfo.moeExpertPerRankNum);
    OP_LOGD(nodeName, "realMaxBs is %u.", tilingData.camMoeCombineNormalInfo.realMaxBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.camMoeCombineNormalInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.camMoeCombineNormalInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.camMoeCombineNormalInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.camMoeCombineNormalInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.camMoeCombineNormalInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.camMoeCombineNormalInfo.totalWinSize);
    OP_LOGD(nodeName, "maxRound is %u.", tilingData.camMoeCombineNormalInfo.maxRound);
    OP_LOGD(nodeName, "perRoundTokens is %u.", tilingData.camMoeCombineNormalInfo.perRoundTokens);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, CamMoeCombineNormalTilingData &tilingData,
                                               const char *nodeName, std::string &groupEp, std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    // 判空
    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName, "groupEp is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNum is null."), return ge::GRAPH_FAILED);

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
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
                    OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
                            MOE_EXPERT_MAX_NUM, moeExpertNum),
                    return ge::GRAPH_FAILED);
    int64_t moePerRankNum = moeExpertNum / epWorldSize;
    int64_t curDispatchStatusNum = moePerRankNum * epWorldSize;
    OP_TILING_CHECK((curDispatchStatusNum > DISPATCH_STATUS_MAX_SUPPORT_NUM),
                    OP_LOGE(nodeName,
                            "The moe experts num must meet the conditions,"
                            " (moeExpertNum / epWorldSize) * epWorldSize <= 1280, but cur is %ld.",
                            curDispatchStatusNum),
                    return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.camMoeCombineNormalInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.camMoeCombineNormalInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.camMoeCombineNormalInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.camMoeCombineNormalInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.camMoeCombineNormalInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static bool CheckInputTensorDim(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *recvXStorageShape = context->GetInputShape(RECV_X_INDEX);
    OP_TILING_CHECK(recvXStorageShape == nullptr, OP_LOGE(nodeName, "recvX is null."), return false);
    OP_TILING_CHECK(recvXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "recvX must be 2-dimension, but got %lu dim",
                            recvXStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "recvX dim0 = %ld", recvXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "recvX dim1 = %ld", recvXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *tokenSrcInfoStorageShape = context->GetInputShape(TOKEN_SRC_INFO_INDEX);
    OP_TILING_CHECK(tokenSrcInfoStorageShape == nullptr, OP_LOGE(nodeName, "tokenSrcInfoForCombine is null."),
                    return false);
    OP_TILING_CHECK(tokenSrcInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "tokenSrcInfoForCombine must be 1-dimension, but got %lu dim",
                            tokenSrcInfoStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "tokenSrcInfoForCombine dim0 = %ld", tokenSrcInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE(nodeName, "topkWeights is null."), return false);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "topkWeights must be 2-dimension, but got %lu dim",
                            topkWeightsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *topIdxStorageShape = context->GetInputShape(TOPK_IDX_INDEX);
    OP_TILING_CHECK(topIdxStorageShape == nullptr, OP_LOGE(nodeName, "topkWeights is null."), return false);
    OP_TILING_CHECK(topIdxStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "topkIdx must be 2-dimension, but got %lu dim",
                            topIdxStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "topkIdx dim0 = %ld", topIdxStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "topkIdx dim1 = %ld", topIdxStorageShape->GetStorageShape().GetDim(1));

    return true;
}

static bool CheckOptionalInputTensorDim(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *tpRecvCountsStorageShape = context->GetOptionalInputShape(TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsStorageShape == nullptr, OP_LOGE(nodeName, "tpRecvCounts is null."), return false);
    OP_TILING_CHECK(tpRecvCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "tpRecvCounts must be 1-dimension, but got %lu dim",
                            tpRecvCountsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "tpRecvCounts dim0 = %ld", tpRecvCountsStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckOutputTensorDim(gert::TilingContext *context, const char *nodeName, const bool isEnableDiagnose)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "x must be 2-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    if (isEnableDiagnose) {
        const gert::StorageShape *sendCostStatsStorageShape = context->GetOutputShape(OUTPUT_SEND_COST_INDEX);
        OP_TILING_CHECK(sendCostStatsStorageShape == nullptr, OP_LOGE(nodeName, "combine sendCostStatsShape is null."),
                        return false);
        OP_TILING_CHECK(sendCostStatsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(nodeName, "combine sendCostStatsShape must be 1-dimension, but got %lu dim",
                                sendCostStatsStorageShape->GetStorageShape().GetDimNum()),
                        return false);
    }
    return true;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isEnableDiagnose)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName),
                    OP_LOGE(nodeName, "param shape of input tensor is invalid"), return false);

    OP_TILING_CHECK(!CheckOptionalInputTensorDim(context, nodeName),
                    OP_LOGE(nodeName, "param shape of optional input tensor is invalid"), return false);

    OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName, isEnableDiagnose),
                    OP_LOGE(nodeName, "param shape of output tensor is invalid"), return false);

    return true;
}

// 校验数据类型
static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName, const bool isEnableDiagnose)
{
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    OP_TILING_CHECK(recvXDesc == nullptr, OP_LOGE(nodeName, "recvXDesc is null."), return false);
    OP_TILING_CHECK((recvXDesc->GetDataType() != ge::DT_BF16) && (recvXDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(nodeName, "recvX dataType is invalid, dataType should be bf16 or float16, but is "),
                    return false);
    auto tokenSrcInfoDesc = context->GetInputDesc(TOKEN_SRC_INFO_INDEX);
    OP_TILING_CHECK(tokenSrcInfoDesc == nullptr, OP_LOGE(nodeName, "tokenSrcInfoDesc is null."), return false);
    OP_TILING_CHECK((tokenSrcInfoDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName,
                            "tokenSrcInfoForCombine dataType is invalid,"
                            " dataType should be int32, but is"),
                    return false);
    auto tpRecvCountsDesc = context->GetOptionalInputDesc(TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK((tpRecvCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is "),
                    return false);
    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(nodeName, "topkWeightsDesc is null."), return false);
    OP_TILING_CHECK((topkWeightsDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(nodeName, "topkWeights dataType is invalid, dataType should be float, but is "),
                    return false);
    auto topkIdxDesc = context->GetInputDesc(TOPK_IDX_INDEX);
    OP_TILING_CHECK(topkIdxDesc == nullptr, OP_LOGE(nodeName, "topkIdxDesc is null."), return false);
    OP_TILING_CHECK((topkIdxDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName,
                            "topkIdxForCombine dataType is invalid,"
                            " dataType should be int32, but is"),
                    return false);
    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != recvXDesc->GetDataType()),
                    OP_LOGE(nodeName, "x dataType is invalid, dataType should be equal to recvX dataType , but is "),
                    return false);

    if (isEnableDiagnose) {
        auto sendCostStatsDesc = context->GetOutputDesc(OUTPUT_SEND_COST_INDEX);
        OP_TILING_CHECK(sendCostStatsDesc == nullptr, OP_LOGE(nodeName, "combine sendCostStatsDesc is null."),
                        return false);
        OP_TILING_CHECK(
            sendCostStatsDesc->GetDataType() != ge::DT_INT32,
            OP_LOGE(nodeName, "combine sendCostStatsDesc dataType is invalid, dataType should be int32, but is ."),
            return false);
    }
    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName, const bool isEnableDiagnose)
{
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    OP_TILING_CHECK(recvXDesc == nullptr, OP_LOGE(nodeName, "recvXDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(recvXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "recvXFormat is invalid"), return false);

    auto tokenSrcInfoDesc = context->GetInputDesc(TOKEN_SRC_INFO_INDEX);
    OP_TILING_CHECK(tokenSrcInfoDesc == nullptr, OP_LOGE(nodeName, "tokenSrcInfoDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(tokenSrcInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "tokenSrcInfoFormat is invalid"), return false);

    auto tpRecvCountsDesc = context->GetOptionalInputDesc(TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "tpRecvCountsFormat is invalid"), return false);

    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(nodeName, "topkWeightsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkWeightsFormat is invalid"), return false);

    auto topkIdxsDesc = context->GetOptionalInputDesc(TOPK_IDX_INDEX);
    OP_TILING_CHECK(topkIdxsDesc == nullptr, OP_LOGE(nodeName, "topkIdxsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdxsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkIdxsFormat is invalid"), return false);

    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "xFormat is invalid"), return false);

    if (isEnableDiagnose) {
        auto sendCostStatsDesc = context->GetOutputDesc(OUTPUT_SEND_COST_INDEX);
        OP_TILING_CHECK(sendCostStatsDesc == nullptr, OP_LOGE(nodeName, "combine sendCostStatsDesc is null."),
                        return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(sendCostStatsDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "combine sendCostStatsDesc format is invalid"), return false);
    }
    return true;
}

static bool CheckTensorShape(gert::TilingContext *context, CamMoeCombineNormalTilingData &tilingData,
                             const char *nodeName, uint32_t localExpertNum)
{
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.camMoeCombineNormalInfo.moeExpertNum);
    OP_TILING_CHECK((topkWeightsDim1 <= 0) || (topkWeightsDim1 > K_MAX || (topkWeightsDim1 > moeExpertNum)),
                    OP_LOGE(nodeName,
                            "topkWeights's dim1(K) should be in (0, min(%ld, moeExpertNum %ld)], "
                            "but got topkWeights's dim1=%ld.",
                            K_MAX, moeExpertNum, topkWeightsDim1),
                    return false);
    tilingData.camMoeCombineNormalInfo.k = static_cast<uint32_t>(topkWeightsDim1);

    // 校验recvX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.camMoeCombineNormalInfo.tpWorldSize);
    const gert::StorageShape *recvXStorageShape = context->GetInputShape(RECV_X_INDEX);
    int64_t recvXDim1 = recvXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((recvXDim1 < H_MIN) || (recvXDim1 > H_MAX),
                    OP_LOGE(nodeName, "recvX's dim1(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, recvXDim1),
                    return false);  // 32对齐
    tilingData.camMoeCombineNormalInfo.h = static_cast<uint32_t>(recvXDim1);

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.camMoeCombineNormalInfo.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.camMoeCombineNormalInfo.moeExpertPerRankNum);

    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != topkWeightsDim0,
                    OP_LOGE(nodeName, "x's dim0 not equal to bs, bs = %ld, x's dim0 = %ld", topkWeightsDim0, xDim0),
                    return false);
    OP_TILING_CHECK(xDim1 != recvXDim1,
                    OP_LOGE(nodeName, "x's dim1 not equal to h, x's dim1 = %ld, h = %ld", xDim1, recvXDim1),
                    return false);

    return true;
}

static bool CheckAttrs(gert::TilingContext *context, CamMoeCombineNormalTilingData &tilingData, const char *nodeName,
                       uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.camMoeCombineNormalInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.camMoeCombineNormalInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.camMoeCombineNormalInfo.moeExpertNum;

    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % epWorldSize != 0,
                    OP_LOGE(nodeName,
                            "moeExpertNum should be divisible by epWorldSize, "
                            "but got moeExpertNum=%d, epWorldSize=%d.",
                            moeExpertNum, epWorldSize),
                    return false);
    localMoeExpertNum = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %d", localMoeExpertNum),
                    return false);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((localMoeExpertNum > 1) && (tpWorldSize > 1),
                    OP_LOGE(nodeName, "Cannot support multi-moeExpert %d in a rank when tpWorldSize = %d > 1",
                            localMoeExpertNum, tpWorldSize),
                    return false);
    tilingData.camMoeCombineNormalInfo.moeExpertPerRankNum = localMoeExpertNum;

    // 校验输入topkWeights的维度0并设bs
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((topkWeightsDim0 <= 0) || (topkWeightsDim0 > BS_UPPER_BOUND),
                    OP_LOGE(nodeName, "Invalid topkWeights dims0(BS) %ld. Should be between [1, %ld].", topkWeightsDim0,
                            BS_UPPER_BOUND),
                    return false);
    tilingData.camMoeCombineNormalInfo.bs = static_cast<uint32_t>(topkWeightsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);
    auto realMaxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_REAL_MAX_BS_INDEX);
    OP_TILING_CHECK(realMaxBsPtr == nullptr, OP_LOGE(nodeName, "realMaxBs is null."), return false);
    OP_LOGD(nodeName, "CamMoeCombineNormal *realMaxBsPtr = %ld, bs = %ld\n", *realMaxBsPtr, topkWeightsDim0);

    OP_TILING_CHECK(
        (*realMaxBsPtr != 0) && (*realMaxBsPtr < topkWeightsDim0),
        OP_LOGE(nodeName,
                "realMaxBs is invalid, only support 0 or value greater than or equal to the BS of all ranks."
                "but got realMaxBs=%ld, bs=%ld",
                *realMaxBsPtr, topkWeightsDim0),
        return false);

    tilingData.camMoeCombineNormalInfo.realMaxBs = static_cast<uint32_t>(*realMaxBsPtr);
    if (*realMaxBsPtr == 0) {
        tilingData.camMoeCombineNormalInfo.realMaxBs = static_cast<uint32_t>(topkWeightsDim0);
    }
    auto maxRoundPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_ROUND_INDEX);
    OP_TILING_CHECK(maxRoundPtr == nullptr, OP_LOGE(nodeName, "maxRound is null."), return false);
    auto perRoundTokensPtr = attrs->GetAttrPointer<int64_t>(ATTR_PER_ROUND_TOKENS_INDEX);
    OP_TILING_CHECK(perRoundTokensPtr == nullptr, OP_LOGE(nodeName, "perRoundTokens is null."), return false);
    tilingData.camMoeCombineNormalInfo.maxRound = static_cast<uint32_t>(*maxRoundPtr);
    tilingData.camMoeCombineNormalInfo.perRoundTokens = static_cast<uint32_t>(*perRoundTokensPtr);
    return true;
}

static ge::graphStatus TilingCheckCamMoeCombineNormal(gert::TilingContext *context, const char *nodeName,
                                                      const bool isEnableDiagnose)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isEnableDiagnose), OP_LOGE(nodeName, "param shape is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isEnableDiagnose),
                    OP_LOGE(nodeName, "param dataType is invalid"), return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isEnableDiagnose),
                    OP_LOGE(nodeName, "param Format is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

static void SetHCommCfg(gert::TilingContext *context, CamMoeCombineNormalTilingData *tiling, const std::string groupEp,
                        const std::string groupTp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "CamMoeCombineNormal groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfigReduceScatterStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2);
}

static ge::graphStatus CamMoeCombineNormalA3TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter CamMoeCombineNormal Tiling func");
    CamMoeCombineNormalTilingData *tilingData = context->GetTilingData<CamMoeCombineNormalTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t localMoeExpertNum = 1;

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    auto sendCostStatsStorageShape = context->GetOutputShape(OUTPUT_SEND_COST_INDEX);
    bool isEnableDiagnose = (sendCostStatsStorageShape != nullptr);
    tilingData->camMoeCombineNormalInfo.isEnableDiagnose = isEnableDiagnose;
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckCamMoeCombineNormal(context, nodeName, isEnableDiagnose) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->camMoeCombineNormalInfo.epRankId;

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData->camMoeCombineNormalInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->camMoeCombineNormalInfo.epWorldSize);
    uint64_t k = static_cast<uint64_t>(tilingData->camMoeCombineNormalInfo.k);
    uint64_t perRoundTokens = tilingData->camMoeCombineNormalInfo.perRoundTokens;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize =
        (perRoundTokens * k * tokenNeedSizeCombine + COMBINE_STATE_WIN_OFFSET + NOTIFY_DISPATCH_WIN_OFFSET) *
        DOUBLE_DATA_BUFFER;
    OP_TILING_CHECK(
        (actualSize > maxWindowSize),
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, perRoundTokens = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
                " tokenNeedSizeCombine = %lu, k = %lu, NEEDED_HCCL_BUFFSIZE("
                "((perRoundTokens * k * tokenNeedSizeCombine)) + 8MB + 102MB) * 2) = %luMB, "
                "HCCL_BUFFSIZE=%luMB.",
                perRoundTokens, h, epWorldSize, localMoeExpertNum, tokenNeedSizeCombine, k, actualSize / MB_SIZE + 1UL,
                maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    tilingData->camMoeCombineNormalInfo.totalWinSize = maxWindowSize;

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    SetHCommCfg(context, tilingData, groupEp, groupTp);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->camMoeCombineNormalInfo.tpWorldSize);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData->camMoeCombineNormalInfo.aivNum = aivNum;
    tilingData->camMoeCombineNormalInfo.totalUbSize = ubSize;
    context->SetScheduleMode(1);  // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName, "blockdim = %u, aivNum = %lu, ubsize = %lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CamMoeCombineNormalTilingFunc(gert::TilingContext *context)
{
    // 不支持 recvX数据类型为int32 type
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(recvXDesc == nullptr, OP_LOGE(nodeName, "recvXDesc is null."), return ge::GRAPH_FAILED);
    // 检查recvX数据类型为DT_INT32
    OP_TILING_CHECK((recvXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "recvX dataType is invalid, dataType should be bf16 or float16, but is "),
                    return ge::GRAPH_FAILED);

    ge::graphStatus ret = CamMoeCombineNormalA3TilingFuncImpl(context);
    return ret;
}

struct CamMoeCombineNormalCompileInfo {};
ge::graphStatus TilingParseForCamMoeCombineNormal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CamMoeCombineNormal)
    .Tiling(CamMoeCombineNormalTilingFunc)
    .TilingParse<CamMoeCombineNormalCompileInfo>(TilingParseForCamMoeCombineNormal);
}  // namespace optiling

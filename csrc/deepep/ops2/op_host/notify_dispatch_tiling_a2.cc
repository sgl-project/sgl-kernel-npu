#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "error_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/notify_dispatch_tiling_a2.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#ifdef USE_CANN83_PATH
#include "platform/platform_infos_def.h"
#elif defined(USE_CANN82_PATH)
#include "experiment/platform/platform/platform_infos_def.h"
#else
#error "CANN version not supported or platform_infos_def.h not found. Check CANN_VERSION_MACRO definition."
#endif

using namespace ge;
namespace {
class Mc2TilingUtils
{
public:
#define HCCL_BUFFSIZE "HCCL_BUFFSIZE"
    static uint64_t GetMaxWindowSize()
    {
        uint16_t defaultWindowSize = 200;
        if (getenv(HCCL_BUFFSIZE) == nullptr) {
            OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
        } else {
            try {
                std::string envStr(getenv(HCCL_BUFFSIZE));
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
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;  // numeric representation of AlltoAll

constexpr uint32_t INPUT_SEND_DATA_INDEX = 0;
constexpr uint32_t INPUT_TOKEN_PER_EXPERT_INDEX = 1;
constexpr uint32_t INPUT_TMP_DATA_INDEX = 2;

constexpr uint32_t OUTPUT_SEND_DATA_OFFSET_INDEX = 0;
constexpr uint32_t OUTPUT_RECV_DATA_INDEX = 1;
constexpr uint32_t OUTPUT_TOKEN_SERVER_IDX_INDEX = 2;
constexpr uint32_t OUTPUT_TOKEN_UNIQUE_PER_SERVER_INDEX = 3;
constexpr uint32_t OUTPUT_EP_RANK_TOKEN_CNT_INDEX = 4;
constexpr uint32_t OUTPUT_LOCAL_EP_TOKEN_CNT_INDEX = 5;
constexpr uint32_t OUTPUT_SRC_OFFSET_RANK_TOKEN_INDEX = 6;
constexpr uint32_t OUTPUT_DST_OFFSET_RANK_TOKEN_INDEX = 7;
constexpr uint32_t OUTPUT_OFFSET_INNER_INDEX = 8;
constexpr uint32_t OUTPUT_COUNT_OUTER_INDEX = 9;
constexpr uint32_t OUTPUT_EXPAND_IDX_INDEX = 10;
constexpr uint32_t OUTPUT_TOTAL_RECV_TOKENS_INDEX = 11;

constexpr uint32_t ATTR_SEND_COUNT_INDEX = 0;
constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 1;
constexpr uint32_t ATTR_TOPK_NUM_INDEX = 2;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 3;
constexpr uint32_t ATTR_COMM_GROUP_INDEX = 4;
constexpr uint32_t ATTR_RANK_SIZE_INDEX = 5;
constexpr uint32_t ATTR_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_LOCAL_RANK_SIZE_INDEX = 7;
constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 8;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_COMM_WORLD_SIZE = 384;
constexpr int64_t MAX_A2_WORLD_SIZE = 64;
constexpr int64_t MAX_COMM_LOCAL_SIZE = 16;
constexpr int64_t MAX_A2_LOCAL_SIZE = 8;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 16 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr static int TILING_KEY_FLOAT16 = 20;
constexpr static int TILING_KEY_BFLOAT16 = 21;
constexpr static int TILING_KEY_FLOAT = 22;
constexpr static int TILING_KEY_INT = 23;
constexpr static int TILING_KEY_A2_TYPE = 100;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, NotifyDispatchA2TilingData &tilingData)
{
    OP_LOGD(nodeName, "rankSize is %u.", tilingData.notifyDispatchInfoA2.rankSize);
    OP_LOGD(nodeName, "rankId is %u.", tilingData.notifyDispatchInfoA2.rankId);
    OP_LOGD(nodeName, "localRankSize is %u.", tilingData.notifyDispatchInfoA2.localRankSize);
    OP_LOGD(nodeName, "localRankId is %u.", tilingData.notifyDispatchInfoA2.localRankId);
    OP_LOGD(nodeName, "sendCount is %u.", tilingData.notifyDispatchInfoA2.sendCount);
    OP_LOGD(nodeName, "numTokens is %u.", tilingData.notifyDispatchInfoA2.numTokens);
    OP_LOGD(nodeName, "topkNum is %u.", tilingData.notifyDispatchInfoA2.topkNum);
    OP_LOGD(nodeName, "numExperts is %u.", tilingData.notifyDispatchInfoA2.numExperts);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.notifyDispatchInfoA2.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.notifyDispatchInfoA2.totalUbSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               NotifyDispatchA2TilingData &tilingData, std::string &commGroup)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto sendCountPtr = attrs->GetAttrPointer<int64_t>(ATTR_SEND_COUNT_INDEX);
    auto numTokenPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_TOKENS_INDEX);
    auto topkNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_TOPK_NUM_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto commGroupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_GROUP_INDEX));
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    auto rankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID_INDEX);
    auto localRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_SIZE_INDEX);
    auto localRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_ID_INDEX);

    OP_TILING_CHECK((commGroupPtr == nullptr) || (strnlen(commGroupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(commGroupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName, "commGroupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountPtr == nullptr, OP_LOGE(nodeName, "sendCountPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numTokenPtr == nullptr, OP_LOGE(nodeName, "numTokenPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkNumPtr == nullptr, OP_LOGE(nodeName, "topkNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSizePtr == nullptr, OP_LOGE(nodeName, "rankSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankIdPtr == nullptr, OP_LOGE(nodeName, "rankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankSizePtr == nullptr, OP_LOGE(nodeName, "localRankSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankIdPtr == nullptr, OP_LOGE(nodeName, "localRankIdPtr is null."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*rankSizePtr <= 0) || (*rankSizePtr > MAX_A2_WORLD_SIZE),
                    OP_LOGE(nodeName, "rankSize is invalid, only support (0, %ld], but got rankSize=%ld.",
                            MAX_A2_WORLD_SIZE, *rankSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*rankIdPtr < 0) || (*rankIdPtr >= *rankSizePtr),
        OP_LOGE(nodeName, "rankId is invalid, only support [0, %ld), but got rankId=%ld.", *rankSizePtr, *rankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*localRankSizePtr <= 0) || (*localRankSizePtr > MAX_A2_LOCAL_SIZE),
                    OP_LOGE(nodeName, "localRankSize is invalid, A2 only support (0, %ld], but got localRankSize=%ld.",
                            MAX_A2_LOCAL_SIZE, *localRankSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*localRankIdPtr < 0) || (*localRankIdPtr >= *localRankSizePtr),
                    OP_LOGE(nodeName, "localRankId is invalid, only support [0, %ld), but got localRankId=%ld.",
                            *localRankSizePtr, *localRankIdPtr),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*sendCountPtr <= 0),
                    OP_LOGE(nodeName, "sendCount is invalid, only support > 0, but got sendCount=%ld.", *sendCountPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*numTokenPtr <= 0),
        OP_LOGE(nodeName, "numTokenPtr is invalid, only support > 0, but got numTokenPtr=%ld.", *numTokenPtr),
        return ge::GRAPH_FAILED);

    commGroup = std::string(commGroupPtr);
    tilingData.notifyDispatchInfoA2.rankSize = static_cast<uint32_t>(*rankSizePtr);
    tilingData.notifyDispatchInfoA2.rankId = static_cast<uint32_t>(*rankIdPtr);
    tilingData.notifyDispatchInfoA2.localRankSize = static_cast<uint32_t>(*localRankSizePtr);
    tilingData.notifyDispatchInfoA2.localRankId = static_cast<uint32_t>(*localRankIdPtr);
    tilingData.notifyDispatchInfoA2.sendCount = static_cast<uint32_t>(*sendCountPtr);
    tilingData.notifyDispatchInfoA2.numTokens = static_cast<uint32_t>(*numTokenPtr);
    tilingData.notifyDispatchInfoA2.topkNum = static_cast<uint32_t>(*topkNumPtr);
    tilingData.notifyDispatchInfoA2.numExperts = static_cast<uint32_t>(*numExpertsPtr);

    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext *context, NotifyDispatchA2TilingData *tiling,
                        const std::string commGroup)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "NotifyDispatchA2 commGroup = %s", commGroup.c_str());
    uint32_t opType1 = 18;  // batch write=18,
    std::string algConfigAllToAllStr = "BatchWrite=level1:hierarchy";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(commGroup, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + KERNEL_USE_WORKSPACE + KERNEL_A2_ARG_SIZE;
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName)
{
    OP_LOGD(nodeName, "========CheckTensorDataType============");
    auto sendData = context->GetInputDesc(INPUT_SEND_DATA_INDEX);
    OP_TILING_CHECK(sendData == nullptr, OP_LOGE(nodeName, "sendData is null."), return false);
    OP_TILING_CHECK(
        (sendData->GetDataType() != ge::DT_BF16) && (sendData->GetDataType() != ge::DT_FLOAT16) &&
            (sendData->GetDataType() != ge::DT_FLOAT) && (sendData->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "sendData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(sendData->GetDataType())),
        return false);
    uint64_t dataSize;
    if ((sendData->GetDataType() == ge::DT_BF16) || (sendData->GetDataType() == ge::DT_FLOAT16)) {
        dataSize = 2;
    } else {
        dataSize = 4;
    }
    auto tokenPerExpertData = context->GetInputDesc(INPUT_TOKEN_PER_EXPERT_INDEX);
    OP_TILING_CHECK(tokenPerExpertData == nullptr, OP_LOGE(nodeName, "tokenPerExpertData is null."), return false);
    OP_TILING_CHECK(
        (tokenPerExpertData->GetDataType() != ge::DT_BF16) && (tokenPerExpertData->GetDataType() != ge::DT_FLOAT16) &&
            (tokenPerExpertData->GetDataType() != ge::DT_FLOAT) && (tokenPerExpertData->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName,
            "tokenPerExpertData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(tokenPerExpertData->GetDataType())),
        return false);
    auto tmpData = context->GetInputDesc(INPUT_TMP_DATA_INDEX);  // 用于算子中临时存数的空间，与recvData相同大小
    OP_TILING_CHECK(tmpData == nullptr, OP_LOGE(nodeName, "tmpData is null."), return false);
    OP_TILING_CHECK(
        (tmpData->GetDataType() != ge::DT_BF16) && (tmpData->GetDataType() != ge::DT_FLOAT16) &&
            (tmpData->GetDataType() != ge::DT_FLOAT) && (tmpData->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "tmpData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(tmpData->GetDataType())),
        return false);

    auto sendDataOffset = context->GetOutputDesc(OUTPUT_SEND_DATA_OFFSET_INDEX);
    OP_TILING_CHECK(sendDataOffset == nullptr, OP_LOGE(nodeName, "sendDataOffset is null."), return false);
    OP_TILING_CHECK(
        (sendDataOffset->GetDataType() != ge::DT_BF16) && (sendDataOffset->GetDataType() != ge::DT_FLOAT16) &&
            (sendDataOffset->GetDataType() != ge::DT_FLOAT) && (sendDataOffset->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "sendDataOffset datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(sendDataOffset->GetDataType())),
        return false);

    auto recvData = context->GetOutputDesc(OUTPUT_RECV_DATA_INDEX);
    OP_TILING_CHECK(recvData == nullptr, OP_LOGE(nodeName, "recvData is null."), return false);
    OP_TILING_CHECK(
        (recvData->GetDataType() != ge::DT_BF16) && (recvData->GetDataType() != ge::DT_FLOAT16) &&
            (recvData->GetDataType() != ge::DT_FLOAT) && (recvData->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "recvData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(recvData->GetDataType())),
        return false);

    auto tokenServerIdx = context->GetOutputDesc(OUTPUT_TOKEN_SERVER_IDX_INDEX);
    OP_TILING_CHECK(tokenServerIdx == nullptr, OP_LOGE(nodeName, "tokenServerIdx is null."), return false);
    OP_TILING_CHECK(
        (tokenServerIdx->GetDataType() != ge::DT_BF16) && (tokenServerIdx->GetDataType() != ge::DT_FLOAT16) &&
            (tokenServerIdx->GetDataType() != ge::DT_FLOAT) && (tokenServerIdx->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "tokenServerIdx datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(tokenServerIdx->GetDataType())),
        return false);

    auto tokenUniquePerServer = context->GetOutputDesc(OUTPUT_TOKEN_UNIQUE_PER_SERVER_INDEX);
    OP_TILING_CHECK(tokenUniquePerServer == nullptr, OP_LOGE(nodeName, "tokenUniquePerServer is null."), return false);
    OP_TILING_CHECK(
        (tokenUniquePerServer->GetDataType() != ge::DT_BF16) &&
            (tokenUniquePerServer->GetDataType() != ge::DT_FLOAT16) &&
            (tokenUniquePerServer->GetDataType() != ge::DT_FLOAT) &&
            (tokenUniquePerServer->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName,
            "tokenUniquePerServer datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(tokenUniquePerServer->GetDataType())),
        return false);

    auto epRankTokenCnt = context->GetOutputDesc(OUTPUT_EP_RANK_TOKEN_CNT_INDEX);
    OP_TILING_CHECK(epRankTokenCnt == nullptr, OP_LOGE(nodeName, "epRankTokenCnt is null."), return false);
    OP_TILING_CHECK(
        (epRankTokenCnt->GetDataType() != ge::DT_BF16) && (epRankTokenCnt->GetDataType() != ge::DT_FLOAT16) &&
            (epRankTokenCnt->GetDataType() != ge::DT_FLOAT) && (epRankTokenCnt->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "epRankTokenCnt datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(epRankTokenCnt->GetDataType())),
        return false);

    auto localEpTokenCnt = context->GetOutputDesc(OUTPUT_LOCAL_EP_TOKEN_CNT_INDEX);
    OP_TILING_CHECK(localEpTokenCnt == nullptr, OP_LOGE(nodeName, "localEpTokenCnt is null."), return false);
    OP_TILING_CHECK((localEpTokenCnt->GetDataType() != ge::DT_INT64),
                    OP_LOGE(nodeName, "localEpTokenCnt datatype is invalid, datatype should be int64, but is %d.",
                            static_cast<ge::DataType>(localEpTokenCnt->GetDataType())),
                    return false);

    auto srcOffsetRankTokenIdx = context->GetOutputDesc(OUTPUT_SRC_OFFSET_RANK_TOKEN_INDEX);
    OP_TILING_CHECK(srcOffsetRankTokenIdx == nullptr, OP_LOGE(nodeName, "srcOffsetRankTokenIdx is null."),
                    return false);
    OP_TILING_CHECK(
        (srcOffsetRankTokenIdx->GetDataType() != ge::DT_BF16) &&
            (srcOffsetRankTokenIdx->GetDataType() != ge::DT_FLOAT16) &&
            (srcOffsetRankTokenIdx->GetDataType() != ge::DT_FLOAT) &&
            (srcOffsetRankTokenIdx->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName,
            "srcOffsetRankTokenIdx datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(srcOffsetRankTokenIdx->GetDataType())),
        return false);

    auto dstOffsetRankTokenIdx = context->GetOutputDesc(OUTPUT_DST_OFFSET_RANK_TOKEN_INDEX);
    OP_TILING_CHECK(dstOffsetRankTokenIdx == nullptr, OP_LOGE(nodeName, "dstOffsetRankTokenIdx is null."),
                    return false);
    OP_TILING_CHECK(
        (dstOffsetRankTokenIdx->GetDataType() != ge::DT_BF16) &&
            (dstOffsetRankTokenIdx->GetDataType() != ge::DT_FLOAT16) &&
            (dstOffsetRankTokenIdx->GetDataType() != ge::DT_FLOAT) &&
            (dstOffsetRankTokenIdx->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName,
            "dstOffsetRankTokenIdx datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(dstOffsetRankTokenIdx->GetDataType())),
        return false);

    auto offsetInner = context->GetOutputDesc(OUTPUT_OFFSET_INNER_INDEX);
    OP_TILING_CHECK(offsetInner == nullptr, OP_LOGE(nodeName, "offsetInner is null."), return false);
    OP_TILING_CHECK(
        (offsetInner->GetDataType() != ge::DT_BF16) && (offsetInner->GetDataType() != ge::DT_FLOAT16) &&
            (offsetInner->GetDataType() != ge::DT_FLOAT) && (offsetInner->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "offsetInner datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(offsetInner->GetDataType())),
        return false);

    auto countOuter = context->GetOutputDesc(OUTPUT_COUNT_OUTER_INDEX);
    OP_TILING_CHECK(countOuter == nullptr, OP_LOGE(nodeName, "countOuter is null."), return false);
    OP_TILING_CHECK(
        (countOuter->GetDataType() != ge::DT_BF16) && (countOuter->GetDataType() != ge::DT_FLOAT16) &&
            (countOuter->GetDataType() != ge::DT_FLOAT) && (countOuter->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "countOuter datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(countOuter->GetDataType())),
        return false);

    auto expandIdx = context->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdx == nullptr, OP_LOGE(nodeName, "expandIdx is null."), return false);
    OP_TILING_CHECK(
        (expandIdx->GetDataType() != ge::DT_BF16) && (expandIdx->GetDataType() != ge::DT_FLOAT16) &&
            (expandIdx->GetDataType() != ge::DT_FLOAT) && (expandIdx->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "expandIdx datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(expandIdx->GetDataType())),
        return false);

    auto totalRecvTokens = context->GetOutputDesc(OUTPUT_TOTAL_RECV_TOKENS_INDEX);
    OP_TILING_CHECK(totalRecvTokens == nullptr, OP_LOGE(nodeName, "totalRecvTokens is null."), return false);
    OP_TILING_CHECK(
        (totalRecvTokens->GetDataType() != ge::DT_BF16) && (totalRecvTokens->GetDataType() != ge::DT_FLOAT16) &&
            (totalRecvTokens->GetDataType() != ge::DT_FLOAT) && (totalRecvTokens->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "totalRecvTokens datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(totalRecvTokens->GetDataType())),
        return false);

    // Verify the size of the win area
    NotifyDispatchA2TilingData *tilingData = context->GetTilingData<NotifyDispatchA2TilingData>();
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    uint64_t actualSize = 2 * dataSize * tilingData->notifyDispatchInfoA2.sendCount + 2 * 1024 * 1024;  // 2MB flag位
    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, should larger than %luMB", actualSize / MB_SIZE);
        return false;
    }
    return true;
}

static ge::graphStatus TilingCheckTensor(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchA2TilingFuncImpl(gert::TilingContext *context)
{
    OP_LOGD(nodeName, "Enter NotifyDispatchA2TilingFuncImpl.");
    const char *nodeName = context->GetNodeName();
    NotifyDispatchA2TilingData *tilingData = context->GetTilingData<NotifyDispatchA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string commGroup = "";
    OP_LOGI(nodeName, "Enter NotifyDispatchA2 tiling check func.");

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, commGroup) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, commGroup);

    std::string hcclIntraPcieEnableStr;
    std::string hcclIntraRoceEnableStr;
    const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
    if (hcclIntraPcieEnable != nullptr) {
        hcclIntraPcieEnableStr = hcclIntraPcieEnable;
    }
    const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
    if (hcclIntraRoceEnable != nullptr) {
        hcclIntraRoceEnableStr = hcclIntraRoceEnable;
    }

    OP_TILING_CHECK(hcclIntraPcieEnableStr.empty() || hcclIntraRoceEnableStr.empty(),
                    OP_LOGE(nodeName, "Please set ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclIntraPcieEnableStr != "1" || hcclIntraRoceEnableStr != "0",
                    OP_LOGE(nodeName, "Need set ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0"),
                    return ge::GRAPH_FAILED);

    int tilingKey = TILING_KEY_INT;
    auto sendDtype = context->GetInputDesc(0)->GetDataType();
    if (sendDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FLOAT16;
    } else if (sendDtype == ge::DT_BF16) {
        tilingKey = TILING_KEY_BFLOAT16;
    } else if (sendDtype == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
    }

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    if (socVersion == "Ascend910B") {
        tilingKey = tilingKey + TILING_KEY_A2_TYPE;
    }
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    blockDim = aivNum;
    context->SetBlockDim(blockDim);
    tilingData->notifyDispatchInfoA2.totalUbSize = ubSize;
    tilingData->notifyDispatchInfoA2.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchA2TilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = NotifyDispatchA2TilingFuncImpl(context);
    return ret;
}

struct NotifyDispatchA2CompileInfo {};
ge::graphStatus TilingParseForNotifyDispatchA2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NotifyDispatchA2)
    .Tiling(NotifyDispatchA2TilingFunc)
    .TilingParse<NotifyDispatchA2CompileInfo>(TilingParseForNotifyDispatchA2);
}  // namespace optiling

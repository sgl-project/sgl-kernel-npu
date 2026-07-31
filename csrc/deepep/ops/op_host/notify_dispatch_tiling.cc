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
#include "mc2_tiling_utils.h"
#include "../op_kernel/notify_dispatch_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "platform/platform_infos_def.h"

using namespace ge;
namespace {
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;  // numeric representation of AlltoAll

constexpr uint32_t INPUT_SEND_DATA_INDEX = 0;
constexpr uint32_t INPUT_TOKEN_PER_EXPERT_INDEX = 1;

constexpr uint32_t OUTPUT_SEND_DATA_OFFSET_INDEX = 0;
constexpr uint32_t OUTPUT_RECV_DATA_INDEX = 1;
constexpr uint32_t OUTPUT_RECV_COUNT_INDEX = 2;
constexpr uint32_t OUTPUT_RECV_OFFSET_INDEX = 3;
constexpr uint32_t OUTPUT_EXPERT_GLOBAL_OFFSET_INDEX = 4;
constexpr uint32_t OUTPUT_SRCRANK_IN_EXPERT_OFFSET_INDEX = 5;
constexpr uint32_t OUTPUT_R_IN_SRCRANK_OFFSET_INDEX = 6;
constexpr uint32_t OUTPUT_TOTAL_RECV_TOKENS_INDEX = 7;
constexpr uint32_t OUTPUT_MAX_BS_INDEX = 8;
constexpr uint32_t OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX = 9;

constexpr uint32_t ATTR_SEND_COUNT_INDEX = 0;
constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 1;
constexpr uint32_t ATTR_COMM_GROUP_INDEX = 2;
constexpr uint32_t ATTR_RANK_SIZE_INDEX = 3;
constexpr uint32_t ATTR_RANK_ID_INDEX = 4;
constexpr uint32_t ATTR_LOCAL_RANK_SIZE_INDEX = 5;
constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_ROUND_INDEX = 7;
constexpr uint32_t ATTR_PER_ROUND_TOKENS_INDEX = 8;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_COMM_WORLD_SIZE = 384;
const int64_t MAX_MOE_EXPERTS_NUM = 1024;
const int32_t MIN_ROUND = 1;
const int32_t MAX_ROUND = 256;
const int32_t MIN_PER_ROUND_TOKENS = 32;
const int32_t MAX_PER_ROUND_TOKENS = 8192;
const int64_t MAX_TOTAL_TOKENS = 131072;
const int32_t SEND_PER_GROUP = 3;
const int32_t EXPERT_NORMAL_NUM = 512;
const int32_t BATCH_ROUND = 16;
const int32_t UB_ALIGN_SIZE_BYTES = 32;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr static int TILING_KEY_FLOAT16 = 20;
constexpr static int TILING_KEY_BFLOAT16 = 21;
constexpr static int TILING_KEY_FLOAT = 22;
constexpr static int TILING_KEY_INT = 23;
constexpr static int TILING_KEY_A2_TYPE = 100;
constexpr static int TILING_KEY_A5_TYPE = 200;

constexpr static int ALL_TO_ALL_CORE_NUM = 32;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, NotifyDispatchTilingData &tilingData)
{
    OP_LOGD(nodeName, "rankSize is %u.", tilingData.notifyDispatchInfo.rankSize);
    OP_LOGD(nodeName, "rankId is %u.", tilingData.notifyDispatchInfo.rankId);
    OP_LOGD(nodeName, "localRankSize is %u.", tilingData.notifyDispatchInfo.localRankSize);
    OP_LOGD(nodeName, "localRankId is %u.", tilingData.notifyDispatchInfo.localRankId);
    OP_LOGD(nodeName, "sendCount is %u.", tilingData.notifyDispatchInfo.sendCount);
    OP_LOGD(nodeName, "numTokens is %u.", tilingData.notifyDispatchInfo.numTokens);
    OP_LOGD(nodeName, "round is %u.", tilingData.notifyDispatchInfo.round);
    OP_LOGD(nodeName, "perRoundTokens is %u.", tilingData.notifyDispatchInfo.perRoundTokens);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.notifyDispatchInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.notifyDispatchInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.notifyDispatchInfo.totalWinSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               NotifyDispatchTilingData &tilingData, std::string &commGroup)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto sendCountPtr = attrs->GetAttrPointer<int32_t>(ATTR_SEND_COUNT_INDEX);
    auto numTokenPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_TOKENS_INDEX);
    auto commGroupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_GROUP_INDEX));
    auto rankSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_RANK_SIZE_INDEX);
    auto rankIdPtr = attrs->GetAttrPointer<int32_t>(ATTR_RANK_ID_INDEX);
    auto localRankSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_RANK_SIZE_INDEX);
    auto localRankIdPtr = attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_RANK_ID_INDEX);
    auto roundPtr = attrs->GetAttrPointer<int32_t>(ATTR_ROUND_INDEX);
    auto perRoundTokensPtr = attrs->GetAttrPointer<int32_t>(ATTR_PER_ROUND_TOKENS_INDEX);

    OP_TILING_CHECK((commGroupPtr == nullptr) || (strnlen(commGroupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(commGroupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName, "commGroupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountPtr == nullptr, OP_LOGE(nodeName, "sendCountPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numTokenPtr == nullptr, OP_LOGE(nodeName, "numTokenPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSizePtr == nullptr, OP_LOGE(nodeName, "rankSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankIdPtr == nullptr, OP_LOGE(nodeName, "rankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankSizePtr == nullptr, OP_LOGE(nodeName, "localRankSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankIdPtr == nullptr, OP_LOGE(nodeName, "localRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(roundPtr == nullptr, OP_LOGE(nodeName, "roundPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(perRoundTokensPtr == nullptr, OP_LOGE(nodeName, "perRoundTokensPtr is null."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*rankSizePtr <= 0) || (*rankSizePtr > MAX_COMM_WORLD_SIZE),
                    OP_LOGE(nodeName, "rankSize is invalid, only support (0, %ld], but got rankSize=%d.",
                            MAX_COMM_WORLD_SIZE, *rankSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*rankIdPtr < 0) || (*rankIdPtr >= *rankSizePtr),
        OP_LOGE(nodeName, "rankId is invalid, only support [0, %d), but got rankId=%d.", *rankSizePtr, *rankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sendCountPtr <= 0),
                    OP_LOGE(nodeName, "sendCount is invalid, only support > 0, but got sendCount=%d.", *sendCountPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*numTokenPtr < 0),
        OP_LOGE(nodeName, "numTokenPtr is invalid, only support >= 0, but got numTokenPtr=%d.", *numTokenPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*localRankSizePtr <= 0) || (*localRankSizePtr > *rankSizePtr),
                    OP_LOGE(nodeName, "localRankSize is invalid, only support (0, %d], but got localRankSize=%d.",
                            *rankSizePtr, *localRankSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*localRankIdPtr < 0) || (*localRankIdPtr >= *localRankSizePtr),
                    OP_LOGE(nodeName, "localRankId is invalid, only support [0, %d), but got localRankId=%d.",
                            *localRankSizePtr, *localRankIdPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*roundPtr < MIN_ROUND) || (*roundPtr > MAX_ROUND),
                    OP_LOGE(nodeName, "round is invalid, only support [%d, %d], but got round=%d.", MIN_ROUND,
                            MAX_ROUND, *roundPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*perRoundTokensPtr < MIN_PER_ROUND_TOKENS) || (*perRoundTokensPtr > MAX_PER_ROUND_TOKENS),
                    OP_LOGE(nodeName, "perRoundTokens is invalid, only support [%d, %d], but got perRoundTokens=%d.",
                            MIN_PER_ROUND_TOKENS, MAX_PER_ROUND_TOKENS, *perRoundTokensPtr),
                    return ge::GRAPH_FAILED);
    const int64_t totalTokenCapacity = static_cast<int64_t>(*roundPtr) * *perRoundTokensPtr;
    OP_TILING_CHECK(totalTokenCapacity > MAX_TOTAL_TOKENS,
                    OP_LOGE(nodeName, "round * perRoundTokens should not exceed %ld, but got %d * %d=%ld.",
                            MAX_TOTAL_TOKENS, *roundPtr, *perRoundTokensPtr, totalTokenCapacity),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*numTokenPtr > totalTokenCapacity,
                    OP_LOGE(nodeName, "numTokens should not exceed round * perRoundTokens, but got %d > %ld.",
                            *numTokenPtr, totalTokenCapacity),
                    return ge::GRAPH_FAILED);
    const int64_t expertDataStride = static_cast<int64_t>(*roundPtr) * SEND_PER_GROUP;
    OP_TILING_CHECK((*sendCountPtr % expertDataStride) != 0,
                    OP_LOGE(nodeName, "sendCount=%d is not divisible by round * sendPerGroup=%ld.", *sendCountPtr,
                            expertDataStride),
                    return ge::GRAPH_FAILED);
    const int64_t numExperts = *sendCountPtr / expertDataStride;
    OP_TILING_CHECK(
        (numExperts <= 0) || (numExperts > MAX_MOE_EXPERTS_NUM) || ((numExperts % *rankSizePtr) != 0),
        OP_LOGE(nodeName, "numExperts=%ld derived from sendCount must be in (0, %ld] and divisible by rankSize=%d.",
                numExperts, MAX_MOE_EXPERTS_NUM, *rankSizePtr),
        return ge::GRAPH_FAILED);

    commGroup = std::string(commGroupPtr);
    tilingData.notifyDispatchInfo.rankSize = static_cast<uint32_t>(*rankSizePtr);
    tilingData.notifyDispatchInfo.rankId = static_cast<uint32_t>(*rankIdPtr);
    tilingData.notifyDispatchInfo.localRankSize = static_cast<uint32_t>(*localRankSizePtr);
    tilingData.notifyDispatchInfo.localRankId = static_cast<uint32_t>(*localRankIdPtr);
    tilingData.notifyDispatchInfo.sendCount = static_cast<uint32_t>(*sendCountPtr);
    tilingData.notifyDispatchInfo.numTokens = static_cast<uint32_t>(*numTokenPtr);
    tilingData.notifyDispatchInfo.round = static_cast<uint32_t>(*roundPtr);
    tilingData.notifyDispatchInfo.perRoundTokens = static_cast<uint32_t>(*perRoundTokensPtr);

    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext *context, NotifyDispatchTilingData *tiling,
                        const std::string commGroup)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "NotifyDispatch commGroup = %s", commGroup.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(commGroup, opType1, algConfigAllToAllStr);

    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);  // 通过不拉起AICPU，提高算子退出性能
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

    auto recvCount = context->GetOutputDesc(OUTPUT_RECV_COUNT_INDEX);
    OP_TILING_CHECK(recvCount == nullptr, OP_LOGE(nodeName, "recvCount is null."), return false);
    OP_TILING_CHECK((recvCount->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "recvCount datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(recvCount->GetDataType())),
                    return false);

    auto recvOffset = context->GetOutputDesc(OUTPUT_RECV_OFFSET_INDEX);
    OP_TILING_CHECK(recvOffset == nullptr, OP_LOGE(nodeName, "recvOffset is null."), return false);
    OP_TILING_CHECK((recvOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "recvOffset datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(recvOffset->GetDataType())),
                    return false);

    auto expertGlobalOffset = context->GetOutputDesc(OUTPUT_EXPERT_GLOBAL_OFFSET_INDEX);
    OP_TILING_CHECK(expertGlobalOffset == nullptr, OP_LOGE(nodeName, "expertGlobalOffset is null."), return false);
    OP_TILING_CHECK((expertGlobalOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "expertGlobalOffset datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(expertGlobalOffset->GetDataType())),
                    return false);

    auto srcrankInExpertOffset = context->GetOutputDesc(OUTPUT_SRCRANK_IN_EXPERT_OFFSET_INDEX);
    OP_TILING_CHECK(srcrankInExpertOffset == nullptr, OP_LOGE(nodeName, "srcrankInExpertOffset is null."),
                    return false);
    OP_TILING_CHECK((srcrankInExpertOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "srcrankInExpertOffset datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(srcrankInExpertOffset->GetDataType())),
                    return false);

    auto rInSrcrankOffset = context->GetOutputDesc(OUTPUT_R_IN_SRCRANK_OFFSET_INDEX);
    OP_TILING_CHECK(rInSrcrankOffset == nullptr, OP_LOGE(nodeName, "rInSrcrankOffset is null."), return false);
    OP_TILING_CHECK((rInSrcrankOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "rInSrcrankOffset datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(rInSrcrankOffset->GetDataType())),
                    return false);

    auto totalRecvTokens = context->GetOutputDesc(OUTPUT_TOTAL_RECV_TOKENS_INDEX);
    OP_TILING_CHECK(totalRecvTokens == nullptr, OP_LOGE(nodeName, "totalRecvTokens is null."), return false);
    OP_TILING_CHECK((totalRecvTokens->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "totalRecvTokens datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(totalRecvTokens->GetDataType())),
                    return false);

    auto maxBs = context->GetOutputDesc(OUTPUT_MAX_BS_INDEX);
    OP_TILING_CHECK(maxBs == nullptr, OP_LOGE(nodeName, "maxBs is null."), return false);
    OP_TILING_CHECK((maxBs->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "maxBs datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(maxBs->GetDataType())),
                    return false);

    auto recvTokensPerExpert = context->GetOutputDesc(OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX);
    OP_TILING_CHECK(recvTokensPerExpert == nullptr, OP_LOGE(nodeName, "recvTokensPerExpert is null."), return false);
    OP_TILING_CHECK((recvTokensPerExpert->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "recvTokensPerExpert datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(recvTokensPerExpert->GetDataType())),
                    return false);

    // Verify the size of the win area
    NotifyDispatchTilingData *tilingData = context->GetTilingData<NotifyDispatchTilingData>();
    uint64_t maxWindowSize = Mc2TilingUtils::GetMaxWindowSize();
    const bool isA5 = (mc2tiling::GetSocVersion(context) == "Ascend950");
    OP_TILING_CHECK(isA5 && maxWindowSize <= mc2tiling::MTE_STATE_ZONE_SIZE,
                    OP_LOGE(nodeName, "HCCL_BUFFSIZE=%lu must be larger than the %lu-byte A5 MTE state zone.",
                            maxWindowSize, mc2tiling::MTE_STATE_ZONE_SIZE),
                    return false);
    const uint64_t usableWindowSize = isA5 ? maxWindowSize - mc2tiling::MTE_STATE_ZONE_SIZE : maxWindowSize;
    uint64_t actualSize = dataSize * tilingData->notifyDispatchInfo.sendCount + 2 * 1024 * 1024;  // 2MB flag位
    if (actualSize > usableWindowSize / 2UL) {
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL for ping-pong NotifyDispatch, should larger than %luMB.",
                actualSize * 2UL / MB_SIZE);
        return false;
    }
    tilingData->notifyDispatchInfo.totalWinSize = maxWindowSize;
    return true;
}

static ge::graphStatus TilingCheckTensor(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    NotifyDispatchTilingData *tilingData = context->GetTilingData<NotifyDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string commGroup = "";
    OP_LOGI(nodeName, "Enter NotifyDispatch tiling check func.");

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, commGroup) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, commGroup);

    int tilingKey = TILING_KEY_INT;

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    if (socVersion == "Ascend910B") {
        tilingKey = tilingKey + TILING_KEY_A2_TYPE;
    } else if (socVersion == "Ascend950") {
        tilingKey = tilingKey + TILING_KEY_A5_TYPE;
    }
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey=%d socVersion=%s", tilingKey, socVersion.c_str());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (socVersion == "Ascend950") {
        const uint32_t round = tilingData->notifyDispatchInfo.round;
        const uint32_t numExperts = tilingData->notifyDispatchInfo.sendCount / round / SEND_PER_GROUP;
        const uint32_t numLocalExperts = numExperts / tilingData->notifyDispatchInfo.rankSize;
        uint32_t batchRounds = 1;
        if (round > 1 && numExperts > EXPERT_NORMAL_NUM) {
            batchRounds = BATCH_ROUND / 2;
        } else if (round > 1 && numExperts <= EXPERT_NORMAL_NUM / 2) {
            batchRounds = BATCH_ROUND * 2;
        } else if (round > 1) {
            batchRounds = BATCH_ROUND;
        }
        const uint64_t tokenPerExpertSize =
            Mc2TilingUtils::CeilAlign(batchRounds * numExperts * sizeof(int32_t), UB_ALIGN_SIZE_BYTES);
        const uint64_t sendDataSize =
            Mc2TilingUtils::CeilAlign(batchRounds * numExperts * SEND_PER_GROUP * sizeof(int32_t), UB_ALIGN_SIZE_BYTES);
        const uint64_t sendDataOffsetSize =
            Mc2TilingUtils::CeilAlign(batchRounds * numExperts * sizeof(int32_t), UB_ALIGN_SIZE_BYTES);
        const uint64_t reorderedSendDataSize = Mc2TilingUtils::CeilAlign(
            batchRounds * numLocalExperts * SEND_PER_GROUP * sizeof(int32_t), UB_ALIGN_SIZE_BYTES);
        const uint64_t requiredUbSize = tokenPerExpertSize + sendDataSize + sendDataOffsetSize + reorderedSendDataSize;
        OP_TILING_CHECK(requiredUbSize > ubSize,
                        OP_LOGE(nodeName,
                                "A5 NotifyDispatch UB is too small: required=%lu, available=%lu, numExperts=%u, "
                                "numLocalExperts=%u, batchRounds=%u.",
                                requiredUbSize, ubSize, numExperts, numLocalExperts, batchRounds),
                        return ge::GRAPH_FAILED);
    }

    blockDim = aivNum;
    context->SetBlockDim(blockDim);
    tilingData->notifyDispatchInfo.totalUbSize = ubSize;
    tilingData->notifyDispatchInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = NotifyDispatchTilingFuncImpl(context);
    return ret;
}

struct NotifyDispatchCompileInfo {};
ge::graphStatus TilingParseForNotifyDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NotifyDispatch)
    .Tiling(NotifyDispatchTilingFunc)
    .TilingParse<NotifyDispatchCompileInfo>(TilingParseForNotifyDispatch);
}  // namespace optiling

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
#include "../op_kernel/shmem_notify_dispatch_tiling.h"
#include "tiling/platform/platform_ascendc.h"
// #include "tiling/hccl/hccl_tiling.h"

#ifdef USE_CANN83_PATH
#include "platform/platform_infos_def.h"
#elif defined(USE_CANN82_PATH)
#include "experiment/platform/platform/platform_infos_def.h"
#else
#error "CANN version not supported or platform_infos_def.h not found. Check CANN_VERSION_MACRO definition."
#endif

using namespace ge;
namespace {
constexpr uint32_t INPUT_TOKEN_PER_EXPERT_INDEX = 0;

constexpr uint32_t OUTPUT_RECV_DATA_INDEX = 0;
constexpr uint32_t OUTPUT_TOTAL_RECV_TOKEN_INDEX = 1;
constexpr uint32_t OUTPUT_MAX_BS_INDEX = 2;
constexpr uint32_t OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX = 3;
constexpr uint32_t OUTPUT_PUT_OFFSET_INDEX = 4;

constexpr uint32_t ATTR_SEND_COUNT_INDEX = 0;
constexpr uint32_t ATTR_RANK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_LOCAL_RANK_SIZE_INDEX = 3;
constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 4;
constexpr uint32_t ATTR_TOPK_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHMEM_PTR_INDEX = 6;

const int64_t MAX_COMM_WORLD_SIZE = 384;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr static int TILING_KEY_INT_SHMEM = 223;
constexpr static int TILING_KEY_A2_TYPE = 100;

}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, ShmemNotifyDispatchTilingData &tilingData)
{
    OP_LOGD(nodeName, "sendCount is %u.", tilingData.notifyDispatchInfo.sendCount);
    OP_LOGD(nodeName, "rankSize is %u.", tilingData.notifyDispatchInfo.rankSize);
    OP_LOGD(nodeName, "rankId is %u.", tilingData.notifyDispatchInfo.rankId);
    OP_LOGD(nodeName, "localRankSize is %u.", tilingData.notifyDispatchInfo.localRankSize);
    OP_LOGD(nodeName, "localRankId is %u.", tilingData.notifyDispatchInfo.localRankId);
    OP_LOGD(nodeName, "topkNum is %u.", tilingData.notifyDispatchInfo.topkNum);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.notifyDispatchInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.notifyDispatchInfo.totalUbSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               ShmemNotifyDispatchTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto sendCountPtr = attrs->GetAttrPointer<int64_t>(ATTR_SEND_COUNT_INDEX);
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    auto rankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID_INDEX);
    auto localRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_SIZE_INDEX);
    auto localRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_ID_INDEX);
    auto topkNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_TOPK_NUM_INDEX);
    auto shmemPtrPtr = attrs->GetAttrPointer<uint64_t>(ATTR_SHMEM_PTR_INDEX);

    OP_TILING_CHECK(sendCountPtr == nullptr, OP_LOGE(nodeName, "sendCountPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSizePtr == nullptr, OP_LOGE(nodeName, "rankSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankIdPtr == nullptr, OP_LOGE(nodeName, "rankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankSizePtr == nullptr, OP_LOGE(nodeName, "localRankSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankIdPtr == nullptr, OP_LOGE(nodeName, "localRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkNumPtr == nullptr, OP_LOGE(nodeName, "topkNumPtr is null."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*rankSizePtr <= 0) || (*rankSizePtr > MAX_COMM_WORLD_SIZE),
                    OP_LOGE(nodeName, "rankSize is invalid, only support (0, %ld], but got rankSize=%ld.",
                            MAX_COMM_WORLD_SIZE, *rankSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*rankIdPtr < 0) || (*rankIdPtr >= *rankSizePtr),
        OP_LOGE(nodeName, "rankId is invalid, only support [0, %ld), but got rankId=%ld.", *rankSizePtr, *rankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sendCountPtr <= 0),
                    OP_LOGE(nodeName, "sendCount is invalid, only support > 0, but got sendCount=%ld.", *sendCountPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*topkNumPtr <= 0),
        OP_LOGE(nodeName, "topkNumPtr is invalid, only support > 0, but got topkNumPtr=%ld.", *topkNumPtr),
        return ge::GRAPH_FAILED);

    tilingData.notifyDispatchInfo.sendCount = static_cast<uint32_t>(*sendCountPtr);
    tilingData.notifyDispatchInfo.rankSize = static_cast<uint32_t>(*rankSizePtr);
    tilingData.notifyDispatchInfo.rankId = static_cast<uint32_t>(*rankIdPtr);
    tilingData.notifyDispatchInfo.localRankSize = static_cast<uint32_t>(*localRankSizePtr);
    tilingData.notifyDispatchInfo.localRankId = static_cast<uint32_t>(*localRankIdPtr);
    tilingData.notifyDispatchInfo.topkNum = static_cast<uint32_t>(*topkNumPtr);
    tilingData.shmemPtr = static_cast<uint64_t>(*shmemPtrPtr);

    return ge::GRAPH_SUCCESS;
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

    auto recvData = context->GetOutputDesc(OUTPUT_RECV_DATA_INDEX);
    OP_TILING_CHECK(recvData == nullptr, OP_LOGE(nodeName, "recvData is null."), return false);
    OP_TILING_CHECK(
        (recvData->GetDataType() != ge::DT_BF16) && (recvData->GetDataType() != ge::DT_FLOAT16) &&
            (recvData->GetDataType() != ge::DT_FLOAT) && (recvData->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "recvData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(recvData->GetDataType())),
        return false);
    
    auto totalRecvToken = context->GetOutputDesc(OUTPUT_TOTAL_RECV_TOKEN_INDEX);
    OP_TILING_CHECK(totalRecvToken == nullptr, OP_LOGE(nodeName, "totalRecvToken is null."), return false);
    OP_TILING_CHECK(
        (totalRecvToken->GetDataType() != ge::DT_BF16) && (totalRecvToken->GetDataType() != ge::DT_FLOAT16) &&
            (totalRecvToken->GetDataType() != ge::DT_FLOAT) && (totalRecvToken->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "totalRecvToken datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(totalRecvToken->GetDataType())),
        return false);

    auto maxBS = context->GetOutputDesc(OUTPUT_MAX_BS_INDEX);
    OP_TILING_CHECK(maxBS == nullptr, OP_LOGE(nodeName, "maxBS is null."), return false);
    OP_TILING_CHECK(
        (maxBS->GetDataType() != ge::DT_BF16) && (maxBS->GetDataType() != ge::DT_FLOAT16) &&
            (maxBS->GetDataType() != ge::DT_FLOAT) && (maxBS->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "maxBS datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(maxBS->GetDataType())),
        return false);

    auto recvTokensPerExpert = context->GetOutputDesc(OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX);
    OP_TILING_CHECK(recvTokensPerExpert == nullptr, OP_LOGE(nodeName, "recvTokensPerExpert is null."), return false);
    OP_TILING_CHECK(
        (recvTokensPerExpert->GetDataType() != ge::DT_INT64),
        OP_LOGE(nodeName,
                "recvTokensPerExpert datatype is invalid, datatype should be int64, but is %d.",
                static_cast<ge::DataType>(recvTokensPerExpert->GetDataType())),
        return false);

    auto putOffset = context->GetOutputDesc(OUTPUT_PUT_OFFSET_INDEX);
    OP_TILING_CHECK(putOffset == nullptr, OP_LOGE(nodeName, "putOffset is null."), return false);
    OP_TILING_CHECK(
        (putOffset->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "putOffset datatype is invalid, datatype should be int32, but is %d.",
                static_cast<ge::DataType>(putOffset->GetDataType())),
        return false);

    return true;
}

static ge::graphStatus TilingCheckTensor(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShmemNotifyDispatchTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    ShmemNotifyDispatchTilingData *tilingData = context->GetTilingData<ShmemNotifyDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter ShmemNotifyDispatch tiling check func.");

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    int tilingKey = TILING_KEY_INT_SHMEM;
    auto sendDtype = context->GetInputDesc(0)->GetDataType();

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    if (socVersion == "Ascend910B") {
        tilingKey = tilingKey + TILING_KEY_A2_TYPE;
    }
    context->SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    blockDim = aivNum;
    context->SetBlockDim(blockDim);
    tilingData->notifyDispatchInfo.totalUbSize = ubSize;
    tilingData->notifyDispatchInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShmemNotifyDispatchTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = ShmemNotifyDispatchTilingFuncImpl(context);
    return ret;
}

struct ShmemNotifyDispatchCompileInfo {};
ge::graphStatus TilingParseForShmemNotifyDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ShmemNotifyDispatch)
    .Tiling(ShmemNotifyDispatchTilingFunc)
    .TilingParse<ShmemNotifyDispatchCompileInfo>(TilingParseForShmemNotifyDispatch);
}  // namespace optiling

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
#include "../op_kernel/dispatch_layout_tiling.h"
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
constexpr uint32_t INPUT_TOPK_IDX_INDEX = 0;

constexpr uint32_t OUTPUT_NUM_TOKEN_PER_RANK_INDEX = 0;
constexpr uint32_t OUTPUT_NUM_TOKEN_PER_EXPERT_INDEX = 1;
constexpr uint32_t OUTPUT_IS_TOKEN_IN_RANK_INDEX = 2;
constexpr uint32_t OUTPUT_NOTIFY_SEND_DATA_INDEX = 3;
constexpr uint32_t OUTPUT_SEND_TOKEN_IDX_SMALL_INDEX = 4;

constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_NUM_RANKS_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_TOPK_INDEX = 3;
constexpr uint32_t ATTR_LOCAL_RANKSIZE_INDEX = 4;
constexpr uint32_t ATTR_PER_ROUND_TOKENS_INDEX = 5;
constexpr uint32_t ATTR_RANK_ID_INDEX = 6;
const int64_t MAX_COMM_WORLD_SIZE = 384;
const int64_t MAX_MOE_EXPERTS_NUM = 512;
const int64_t MAX_LOCAL_RANKSIZE = 8;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;

constexpr static int TILING_KEY_INT = 23;
constexpr static int TILING_KEY_A2_TYPE = 100;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t K_MAX = 16;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, DispatchLayoutTilingData &tilingData)
{
    OP_LOGD(nodeName, "numToken is %u.", tilingData.dispatchLayoutInfo.numTokens);
    OP_LOGD(nodeName, "numRanks is %u.", tilingData.dispatchLayoutInfo.numRanks);
    OP_LOGD(nodeName, "numExperts is %u.", tilingData.dispatchLayoutInfo.numExperts);
    OP_LOGD(nodeName, "numTopk is %u.", tilingData.dispatchLayoutInfo.numTopk);
    OP_LOGD(nodeName, "localRankSize is %u.", tilingData.dispatchLayoutInfo.localRankSize);
    OP_LOGD(nodeName, "perRoundTokens is %u.", tilingData.dispatchLayoutInfo.perRoundTokens);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.dispatchLayoutInfo.totalUbSize);
}

static bool CheckIfA2Machine(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    if (socVersion == "Ascend910B") {
        return true;
    }
    return false;
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               DispatchLayoutTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto numTokensPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_TOKENS_INDEX));
    auto numRanksPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_RANKS_INDEX));
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_EXPERTS_INDEX));
    auto numTopkPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_TOPK_INDEX));
    auto localRankSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_LOCAL_RANKSIZE_INDEX));
    auto perRoundTokensPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_PER_ROUND_TOKENS_INDEX));
    auto rankIdPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_RANK_ID_INDEX));

    OP_TILING_CHECK(numTokensPtr == nullptr, OP_LOGE(nodeName, "numTokensPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRanksPtr == nullptr, OP_LOGE(nodeName, "numRanksPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numTopkPtr == nullptr, OP_LOGE(nodeName, "numTopkPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankSizePtr == nullptr, OP_LOGE(nodeName, "localRankSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(perRoundTokensPtr == nullptr, OP_LOGE(nodeName, "perRoundTokensPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankIdPtr == nullptr, OP_LOGE(nodeName, "rankIdPtr is null."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*numRanksPtr <= 0) || (*numRanksPtr > MAX_COMM_WORLD_SIZE),
                    OP_LOGE(nodeName, "rankSize is invalid, only support (0, %ld], but got rankSize=%ld.",
                            MAX_COMM_WORLD_SIZE, *numRanksPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numExpertsPtr <= 0) || (*numExpertsPtr > MAX_MOE_EXPERTS_NUM),
                    OP_LOGE(nodeName, "numExperts is invalid, only support (0, %ld], but got numExperts=%ld.",
                            MAX_MOE_EXPERTS_NUM, *numExpertsPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numExpertsPtr % *numRanksPtr) != 0,
                    OP_LOGE(nodeName, "numExperts must be divisible by numRanks, but numExperts=%ld and numRanks=%ld.",
                            *numExpertsPtr, *numRanksPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*numTopkPtr <= 0) || (*numTopkPtr > K_MAX),
        OP_LOGE(nodeName, "numTopkPtr is invalid, only support (0, %u], but got numTopk=%ld.", K_MAX, *numTopkPtr),
        return ge::GRAPH_FAILED);

    if (CheckIfA2Machine(context)) {
        OP_TILING_CHECK(
            (*localRankSizePtr <= 0) || (*localRankSizePtr > MAX_LOCAL_RANKSIZE),
            OP_LOGE(nodeName, "localRankSizePtr is invalid, only support (0, %ld], but got localRankSize=%ld.",
                    MAX_LOCAL_RANKSIZE, *localRankSizePtr),
            return ge::GRAPH_FAILED);
    }

    tilingData.dispatchLayoutInfo.numTokens = static_cast<uint32_t>(*numTokensPtr);
    tilingData.dispatchLayoutInfo.numRanks = static_cast<uint32_t>(*numRanksPtr);
    tilingData.dispatchLayoutInfo.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    tilingData.dispatchLayoutInfo.numTopk = static_cast<uint32_t>(*numTopkPtr);
    tilingData.dispatchLayoutInfo.localRankSize = static_cast<uint32_t>(*localRankSizePtr);
    tilingData.dispatchLayoutInfo.perRoundTokens = static_cast<uint32_t>(*perRoundTokensPtr);
    tilingData.dispatchLayoutInfo.rankId = static_cast<uint32_t>(*rankIdPtr);

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
    auto topkIdx = context->GetInputDesc(INPUT_TOPK_IDX_INDEX);
    auto numTokensPerRank = context->GetOutputDesc(OUTPUT_NUM_TOKEN_PER_RANK_INDEX);
    auto numTokensPerExpert = context->GetOutputDesc(OUTPUT_NUM_TOKEN_PER_EXPERT_INDEX);
    auto isTokenInRank = context->GetOutputDesc(OUTPUT_IS_TOKEN_IN_RANK_INDEX);
    auto notifySendData = context->GetOutputDesc(OUTPUT_NOTIFY_SEND_DATA_INDEX);
    auto sendTokenIdxSmall = context->GetOutputDesc(OUTPUT_SEND_TOKEN_IDX_SMALL_INDEX);

    OP_TILING_CHECK(topkIdx == nullptr, OP_LOGE(nodeName, "topkIdx is null."), return false);
    OP_TILING_CHECK(numTokensPerRank == nullptr, OP_LOGE(nodeName, "numTokensPerRank is null."), return false);
    OP_TILING_CHECK(numTokensPerExpert == nullptr, OP_LOGE(nodeName, "numTokensPerExpert is null."), return false);
    OP_TILING_CHECK(isTokenInRank == nullptr, OP_LOGE(nodeName, "isTokenInRank is null."), return false);
    OP_TILING_CHECK(notifySendData == nullptr, OP_LOGE(nodeName, "notifySendData is null."), return false);
    OP_TILING_CHECK(sendTokenIdxSmall == nullptr, OP_LOGE(nodeName, "sendTokenIdxSmall is null."), return false);

    OP_TILING_CHECK((topkIdx->GetDataType() != ge::DT_INT64),
                    OP_LOGE(nodeName, "topkIdx datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(topkIdx->GetDataType())),
                    return false);
    OP_TILING_CHECK((numTokensPerRank->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "numTokensPerRank datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(numTokensPerRank->GetDataType())),
                    return false);
    OP_TILING_CHECK((numTokensPerExpert->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "numTokensPerExpert datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(numTokensPerExpert->GetDataType())),
                    return false);
    OP_TILING_CHECK((isTokenInRank->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "isTokenInRank datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(isTokenInRank->GetDataType())),
                    return false);
    OP_TILING_CHECK((notifySendData->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "notifySendData datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(notifySendData->GetDataType())),
                    return false);
    OP_TILING_CHECK((sendTokenIdxSmall->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "sendTokenIdxSmall datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(sendTokenIdxSmall->GetDataType())),
                    return false);

    return true;
}

static bool CheckTensorShape(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *topkIdxStorageShape = context->GetInputShape(INPUT_TOPK_IDX_INDEX);

    OP_TILING_CHECK((topkIdxStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS),
                    OP_LOGE(nodeName, "topkIdx must be 2-dimension, but get %lu dim.",
                            topkIdxStorageShape->GetStorageShape().GetDimNum()),
                    return false);

    return true;
}

static ge::graphStatus TilingCheckTensor(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckTensorShape(context, nodeName), OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchLayoutTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    DispatchLayoutTilingData *tilingData = context->GetTilingData<DispatchLayoutTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter NotifyDispatch tiling check func.");

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    int tilingKey = TILING_KEY_INT;
    if (CheckIfA2Machine(context)) {
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
    tilingData->dispatchLayoutInfo.totalUbSize = ubSize;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchLayoutTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret;
    ret = DispatchLayoutTilingFuncImpl(context);
    return ret;
}

struct DispatchLayoutCompileInfo {};
ge::graphStatus TilingParseForDispatchLayout(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchLayout)
    .Tiling(DispatchLayoutTilingFunc)
    .TilingParse<DispatchLayoutCompileInfo>(TilingParseForDispatchLayout);
}  // namespace optiling

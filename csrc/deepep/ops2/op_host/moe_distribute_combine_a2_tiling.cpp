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
// #include "hcom_topo_info.h"
// #include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/moe_distribute_combine_a2_tiling.h"
#include "tiling_args.h"

#define OPS_CHECK OP_TILING_CHECK
#define OPS_LOG_E OP_LOGE
#define OPS_LOG_I OP_LOGI
#define OPS_LOG_D OP_LOGD
#define OPS_REPORT_VECTOR_INNER_ERR VECTOR_INNER_ERR_REPORT_TILIING

using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t EXPAND_IDX_INDEX = 2;
constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
constexpr uint32_t EXPERT_SCALES_INDEX = 4;
constexpr uint32_t TP_SEND_COUNTS_INDEX = 5;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 6;
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

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t EXPAND_IDX_DIMS = 1U;
constexpr uint64_t INIT_TILINGKEY_TP_2 = 1100UL;
constexpr uint64_t INIT_TILINGKEY_TP_1 = 1000UL;
constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;      // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U;  // numeric representation of AlltoAll

constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr uint32_t MAX_BATCH_SIZE_LAYERED_A2 = 4096;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr uint32_t MIN_K_VALUE_A2 = 2;
const char *K_INNER_DEBUG = "MoeDistributeCombine Tiling Debug";
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 4096;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 8;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
}  // namespace

namespace optiling {
static ge::graphStatus MoeDistributeCombineA2CheckAttrAndSetTiling(gert::TilingContext *context,
                                                                   MoeDistributeCombineA2Info &info)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int>(ATTR_GLOBAL_BS_INDEX);

    OPS_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > MAX_EP_WORLD_SIZE_A2 ||
                  *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
              OPS_LOG_E(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    OPS_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
              OPS_LOG_E(K_INNER_DEBUG, "epRankId is invalid."), return GRAPH_FAILED);
    OPS_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2 ||
                  *moeExpertNumPtr % *epWorldSizePtr != 0,
              OPS_LOG_E(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OPS_CHECK(tpWorldSizePtr == nullptr, OPS_LOG_E(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OPS_CHECK(tpRankIdPtr == nullptr, OPS_LOG_E(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OPS_CHECK(expertSharedTypePtr == nullptr, OPS_LOG_E(K_INNER_DEBUG, "expertSharedType is null."),
              return GRAPH_FAILED);
    OPS_CHECK(sharedExpertRankNumPtr == nullptr, OPS_LOG_E(K_INNER_DEBUG, "sharedExpertRankNum is null."),
              return GRAPH_FAILED);
    OPS_CHECK(globalBsPtr == nullptr, OPS_LOG_E(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OPS_CHECK(expertIdStorageShape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "xShape is null."), return false);
    int32_t globalBs = *epWorldSizePtr * expertIdStorageShape->GetStorageShape().GetDim(0);

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    if (*globalBsPtr == 0) {
        info.globalBs = static_cast<uint32_t>(globalBs);
    } else {
        info.globalBs = *globalBsPtr;
    }

    OPS_LOG_D(K_INNER_DEBUG, "epWorldSize=%u", info.epWorldSize);
    OPS_LOG_D(K_INNER_DEBUG, "tpWorldSize=%u", info.tpWorldSize);
    OPS_LOG_D(K_INNER_DEBUG, "epRankId=%u", info.epRankId);
    OPS_LOG_D(K_INNER_DEBUG, "tpRankId=%u", info.tpRankId);
    OPS_LOG_D(K_INNER_DEBUG, "expertSharedType=%u", info.expertSharedType);
    OPS_LOG_D(K_INNER_DEBUG, "sharedExpertRankNum=%u", info.sharedExpertRankNum);
    OPS_LOG_D(K_INNER_DEBUG, "moeExpertNum=%u", info.moeExpertNum);
    OPS_LOG_D(K_INNER_DEBUG, "globalBs=%u", info.globalBs);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2CheckShapeAndSetTiling(gert::TilingContext *context,
                                                                    MoeDistributeCombineA2Info &info,
                                                                    const bool isLayered)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OPS_CHECK(expandXStorageShape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "expandXShape is null."), return GRAPH_FAILED);
    OPS_CHECK(expertIdStorageShape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);

    OPS_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
              OPS_LOG_E(K_INNER_DEBUG, "expandXshape is invalid"), return GRAPH_FAILED);
    uint32_t h = expandXStorageShape->GetStorageShape().GetDim(1);
    OPS_CHECK(h <= 0 || h > MAX_HIDDEN_SIZE_A2 || h % BLOCK_SIZE_A2 != 0,
              OPS_LOG_E(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    OPS_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
              OPS_LOG_E(K_INNER_DEBUG, "expertIdshape is invalid"), return GRAPH_FAILED);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    OPS_CHECK(bs <= 0, OPS_LOG_E(K_INNER_DEBUG, "batchsize is invalid."), return GRAPH_FAILED);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    OPS_CHECK(k < MIN_K_VALUE_A2 || k > MAX_K_VALUE_A2,
              OPS_LOG_E(K_INNER_DEBUG, "k should be in [%u, %u], but got k=%u.", MIN_K_VALUE_A2, MAX_K_VALUE_A2, k),
              return GRAPH_FAILED);
    const uint32_t maxBatchSize = isLayered ? MAX_BATCH_SIZE_LAYERED_A2 : MAX_BATCH_SIZE_A2;
    OPS_CHECK(bs > maxBatchSize, OPS_LOG_E(K_INNER_DEBUG, "Batchsize must be smaller than %u.", maxBatchSize),
              return ge::GRAPH_FAILED);
    info.bs = bs;
    info.k = k;
    info.h = h;

    OPS_LOG_D(K_INNER_DEBUG, "batchSize=%u", bs);
    OPS_LOG_D(K_INNER_DEBUG, "k=%u", k);
    OPS_LOG_D(K_INNER_DEBUG, "hidenSize=%u", h);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2GetPlatformInfoAndSetTiling(gert::TilingContext *context,
                                                                         MoeDistributeCombineA2Info &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OPS_LOG_D(K_INNER_DEBUG, "aivNum=%u", info.aivNum);
    OPS_LOG_D(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

static bool MoeDistributeCombineA2IsLayered()
{
    const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
    const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");

    if (hcclIntraPcieEnable == nullptr || hcclIntraRoceEnable == nullptr) {
        OPS_LOG_D(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE don't set");
        return false;
    }
    if (strcmp(hcclIntraPcieEnable, "1") == 0 && strcmp(hcclIntraRoceEnable, "0") == 0) {
        OPS_LOG_D(K_INNER_DEBUG,
                  "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use layered solution.");
        return true;
    }
    OPS_LOG_D(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default solution.");
    return false;
}

static uint64_t MoeDistributeCombineA2CalcTilingKey(gert::TilingContext *context, const bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter MoeDistributeCombineA2 calc tiling func.");

    uint64_t tilingKey = TILING_KEY_BASE_A2;

    if (isLayered) {
        tilingKey = TILING_KEY_LAYERED_COMM_A2;
    }

    OPS_LOG_D(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);

    return tilingKey;
}

static ge::graphStatus MoeDistributeCombineA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter MoeDistributeCombineA2 tiling func.");

    // tilingData
    MoeDistributeCombineA2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineA2TilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(nodeName, "tilingData is nullptr."),
              return ge::GRAPH_FAILED);
    OPS_LOG_I(nodeName, "MoeDistributeCombineA2 get tilingData.");
    MoeDistributeCombineA2Info &info = tilingData->moeDistributeCombineInfo;

    bool isLayered = MoeDistributeCombineA2IsLayered();
    OPS_CHECK(
        MoeDistributeCombineA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeDistributeCombineA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        MoeDistributeCombineA2CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeDistributeCombineA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(MoeDistributeCombineA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                          "MoeDistributeCombineA2 GetPlatformInfoAndSetTiling Failed"),
              return ge::GRAPH_FAILED);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);

    uint64_t tilingKey = MoeDistributeCombineA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workSpaces == nullptr, OPS_REPORT_VECTOR_INNER_ERR(nodeName, "workSpaces is nullptr."),
              return ge::GRAPH_FAILED);
    uint32_t userWorkspaceSize = static_cast<uint32_t>(info.moeExpertNum) * sizeof(uint32_t) * 2;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkspaceSize;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    uint32_t opType = 18;  // batch write=18,
    std::string algConfig = "MultiPut=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OPS_LOG_I(nodeName, "Leave MoeDistributeCombineA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineTilingFunc(gert::TilingContext *context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OPS_CHECK(expandXDesc == nullptr, OPS_LOG_E(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OPS_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
              OPS_LOG_E(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %d",
                        static_cast<ge::DataType>(expandXDesc->GetDataType())),
              return ge::GRAPH_FAILED);

    return MoeDistributeCombineA2TilingFuncImpl(context);
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineA2)
    .Tiling(MoeDistributeCombineTilingFunc)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombine);
}  // namespace optiling

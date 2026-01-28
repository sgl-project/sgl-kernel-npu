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
// #include "mc2_tiling_utils.h"
#include "../op_kernel/shmem_moe_combine_normal_tiling.h"
#include "tiling_args.h"

using namespace AscendC;
using namespace ge;
using namespace Moe;

namespace {
constexpr uint32_t RECV_X_INDEX = 0;
constexpr uint32_t EP_RECV_COUNTS_INDEX = 1;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 2;
constexpr uint32_t TOPK_IDX_INDEX = 3;
constexpr uint32_t SEND_TOPKEN_IDX_INDEX = 4;

constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_SEND_COST_INDEX = 1;

constexpr uint32_t ATTR_META_DATA_PTR_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 4;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 6;

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;      // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U;  // numeric representation of ReduceScatter

constexpr int64_t MAX_EP_WORLD_SIZE = 384;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 16;
constexpr int64_t H_MIN = 1;
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

static int GetFactorEnv(const char *name, int value = 0)
{
    int defaultValue = value;
    if (getenv(name) == nullptr) {
        OP_LOGD("", "Env %s don't set", name);
    } else {
        try {
            std::string envStr(getenv(name));
            defaultValue = static_cast<int>(std::stoi(envStr));
        } catch (const std::invalid_argument &ia) {
            OP_LOGE("", "Invalid argument when parsing %s: %s", name, ia.what());
        } catch (const std::out_of_range &oor) {
            OP_LOGE("", "Out of range when parsing %s: %s", name, oor.what());
        }
    }
    OP_LOGI("", "Get factor is %f", defaultValue);
    return defaultValue;
}

// a3专有
static void PrintTilingDataInfo(const char *nodeName, ShmemMoeCombineNormalTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeCombineNormalInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeCombineNormalInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeCombineNormalInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeCombineNormalInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeCombineNormalInfo.expertShardType);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeCombineNormalInfo.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeCombineNormalInfo.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeCombineNormalInfo.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeCombineNormalInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeCombineNormalInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeCombineNormalInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeCombineNormalInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeCombineNormalInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.moeCombineNormalInfo.totalWinSize);
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context,
                                               ShmemMoeCombineNormalTilingData &tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto metaDataPtrPtr = attrs->GetAttrPointer<uint64_t>(ATTR_META_DATA_PTR_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    // 判空
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

    tilingData.shmemPtr = static_cast<uint64_t>(*metaDataPtrPtr);
    tilingData.moeCombineNormalInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeCombineNormalInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeCombineNormalInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeCombineNormalInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeCombineNormalInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);

    tilingData.moeCombineNormalInfo.enableBalance = GetFactorEnv("ENABLE_REBALANCE", 0) > 0;

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

    const gert::StorageShape *epRecvCountStorageShape = context->GetInputShape(EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCount is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "epRecvCount must be 2-dimension, but got %lu dim",
                            epRecvCountStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "epRecvCount dim1 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE(nodeName, "topkWeights is null."), return false);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "topkWeights must be 2-dimension, but got %lu dim",
                            topkWeightsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsStorageShape->GetStorageShape().GetDim(1));

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
    auto epRecvCountsDesc = context->GetOptionalInputDesc(EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK((epRecvCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "epRecvCounts dataType is invalid, dataType should be int32, but is %d",
                            static_cast<ge::DataType>(epRecvCountsDesc->GetDataType())),
                    return false);
    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(nodeName, "topkWeightsDesc is null."), return false);
    OP_TILING_CHECK((topkWeightsDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(nodeName, "topkWeights dataType is invalid, dataType should be float, but is "),
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

    auto epRecvCountsDesc = context->GetOptionalInputDesc(EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "epRecvCountsFormat is invalid"), return false);

    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(nodeName, "topkWeightsDesc is null."), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkWeightsFormat is invalid"), return false);

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

static bool CheckTensorShape(gert::TilingContext *context, ShmemMoeCombineNormalTilingData &tilingData,
                             const char *nodeName, uint32_t localExpertNum)
{
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeCombineNormalInfo.moeExpertNum);
    OP_TILING_CHECK((topkWeightsDim1 <= 0) || (topkWeightsDim1 > K_MAX || (topkWeightsDim1 > moeExpertNum)),
                    OP_LOGE(nodeName,
                            "topkWeights's dim1(K) should be in (0, min(%ld, moeExpertNum %ld)], "
                            "but got topkWeights's dim1=%ld.",
                            K_MAX, moeExpertNum, topkWeightsDim1),
                    return false);
    tilingData.moeCombineNormalInfo.k = static_cast<uint32_t>(topkWeightsDim1);

    // 校验recvX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeCombineNormalInfo.tpWorldSize);
    const gert::StorageShape *recvXStorageShape = context->GetInputShape(RECV_X_INDEX);
    int64_t recvXDim1 = recvXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((recvXDim1 < H_MIN) || (recvXDim1 > H_MAX),
                    OP_LOGE(nodeName, "recvX's dim1(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, recvXDim1),
                    return false);  // 32对齐
    tilingData.moeCombineNormalInfo.h = static_cast<uint32_t>(recvXDim1);

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeCombineNormalInfo.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeCombineNormalInfo.moeExpertPerRankNum);

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

static bool CheckAttrs(gert::TilingContext *context, ShmemMoeCombineNormalTilingData &tilingData, const char *nodeName,
                       uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeCombineNormalInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeCombineNormalInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeCombineNormalInfo.moeExpertNum;

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
    tilingData.moeCombineNormalInfo.moeExpertPerRankNum = localMoeExpertNum;

    // 校验输入topkWeights的维度0并设bs
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    tilingData.moeCombineNormalInfo.bs = static_cast<uint32_t>(topkWeightsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    OP_TILING_CHECK(
        (*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * topkWeightsDim0) ||
                                ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
        OP_LOGE(nodeName,
                "globalBS is invalid, only "
                "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
                "bs=%ld, epWorldSize=%u.",
                *globalBsPtr, topkWeightsDim0, epWorldSize),
        return false);
    tilingData.moeCombineNormalInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    OP_LOGD(nodeName, "*globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, topkWeightsDim0, epWorldSize);

    if (*globalBsPtr == 0) {
        tilingData.moeCombineNormalInfo.globalBs = static_cast<uint32_t>(topkWeightsDim0) * epWorldSize;
    }

    return true;
}

static ge::graphStatus TilingCheckShmemMoeCombineNormal(gert::TilingContext *context, const char *nodeName,
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

static ge::graphStatus ShmemMoeCombineNormalA3TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter ShmemMoeCombineNormal Tiling func");
    ShmemMoeCombineNormalTilingData *tilingData = context->GetTilingData<ShmemMoeCombineNormalTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    uint32_t localMoeExpertNum = 1;

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    auto sendCostStatsStorageShape = context->GetOutputShape(OUTPUT_SEND_COST_INDEX);
    bool isEnableDiagnose = (sendCostStatsStorageShape != nullptr);
    tilingData->moeCombineNormalInfo.isEnableDiagnose = isEnableDiagnose;
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckShmemMoeCombineNormal(context, nodeName, isEnableDiagnose) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeCombineNormalInfo.epRankId;

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    uint64_t h = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.epWorldSize);
    uint64_t k = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.k);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.globalBs) / epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize =
        (maxBs * k * tokenNeedSizeCombine + COMBINE_STATE_WIN_OFFSET + NOTIFY_DISPATCH_WIN_OFFSET) * DOUBLE_DATA_BUFFER;

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->moeCombineNormalInfo.tpWorldSize);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData->moeCombineNormalInfo.aivNum = aivNum;
    tilingData->moeCombineNormalInfo.totalUbSize = ubSize;
    context->SetScheduleMode(1);  // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName, "blockdim = %u, aivNum = %lu, ubsize = %lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShmemMoeCombineNormalTilingFunc(gert::TilingContext *context)
{
    // 不支持 recvX数据类型为int32 type
    auto recvXDesc = context->GetInputDesc(RECV_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(recvXDesc == nullptr, OP_LOGE(nodeName, "recvXDesc is null."), return ge::GRAPH_FAILED);
    // 检查recvX数据类型为DT_INT32
    OP_TILING_CHECK((recvXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "recvX dataType is invalid, dataType should be bf16 or float16, but is "),
                    return ge::GRAPH_FAILED);

    ge::graphStatus ret = ShmemMoeCombineNormalA3TilingFuncImpl(context);
    return ret;
}

struct ShmemMoeCombineNormalCompileInfo {};
ge::graphStatus TilingParseForShmemMoeCombineNormal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ShmemMoeCombineNormal)
    .Tiling(ShmemMoeCombineNormalTilingFunc)
    .TilingParse<ShmemMoeCombineNormalCompileInfo>(TilingParseForShmemMoeCombineNormal);
}  // namespace optiling

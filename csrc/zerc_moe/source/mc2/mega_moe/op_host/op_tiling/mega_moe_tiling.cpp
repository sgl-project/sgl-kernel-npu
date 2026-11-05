/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mega_moe.h"
#include "../../op_kernel/mega_moe_tiling.h"
#include "../../op_kernel/mega_moe_tiling_key.h"
#include "../../op_kernel/mega_moe_workspace_info.h"
#include "../../op_kernel/AscendTimerV2.hpp"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
namespace {
// init routing
const static int64_t SIMT_DCACHE_SIZE = 64 * 1024LL;
const static int64_t SORT_API_MAX_ELEM = 32 * 255LL;
const static int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
const static int64_t MX_QUANT_BLOCK_SIZE = 32LL;

const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t MRG_LIST_NUM = 4LL;
const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
const static int64_t UB_BLOCK_SIZE = 32LL;
const static size_t DIM_TWO = 2ULL;
const static int64_t KV_FACTOR = 2LL;
const static int64_t EXPERT_IDX_MAX = 10240LL;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;

const static int64_t ROW_IDX_GATHER = 0LL;
const static int64_t ROW_IDX_SCATTER = 1LL;
const static int64_t EXPERT_TOKENS_TYPE_CUMSUM = 0LL;
const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;
const static int64_t SORT_CORE_TILINGKEY_BASE = 100000LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_BS = 1LL;
// maxOutputSize = BS * epWorldSize * min(topK, expertPerRank) must fit the
// kernel's int32 row index, and epWorldSize * expertPerRank <= 1024 caps
// that product at 1024.  The formal ceiling is therefore INT32_MAX / 1024
// = 2097151, which 262144 stays eight times inside.  What actually bounds
// BS is device memory, not this constant: even with a tightened
// maxRecvTokenNum, 256K local BS needs tens of GB of workspace and CCL
// budget, and without one it needs hundreds.
const static int64_t MAX_BS = 262144LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = 16LL;
// 共享专家：不参与 Dispatch 的本地专家，上限与参考实现一致。
const static int64_t MAX_SHARED_EXPERT_NUM = 4LL;
const static int64_t H_BASE = 1024LL;
const static int64_t N_BASE = 1024LL;
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_EP_WORLD_SIZE = 768LL;
const static int64_t MAX_MOE_EXPERT_NUM = 1024LL;
const static int64_t DISABLE_EXPERT_CAPACITY = -1LL;
const static int64_t DISPATCH_QUANT_OUT_TYPE_OFFSET = 20LL;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
// BF16 no-quant rows must be 32B aligned so every GM<->UB row copy starts
// on a block boundary (2 * H % 32 == 0).
const static int64_t NO_QUANT_H_ALIGN = 16LL;
}  // namespace

// Attributes are validated after the tensor checks, so tolerate a missing
// attribute here; CheckAttrPtrNullptr reports it later.
static bool IsNoQuantMode(const gert::TilingContext *context, const MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>(config.attrDispatchQuantModeIndex);
    return dispatchQuantModePtr != nullptr && *dispatchQuantModePtr == DISPATCH_QUANT_MODE_NO_QUANT;
}

struct MegaMoeAuxTilingContext {
    // Platform
    int64_t aivCoreNum;
    int64_t totalUbSize;
    int64_t availUbSize;

    // Critical Variable
    int64_t sortLoopMaxElement;
    int64_t totalLength;
    int64_t n;
    int64_t k;
    int64_t cols;
    int64_t inputXDtypeSize;
    int64_t isInputScale;
    int64_t isInputOffset;
    int64_t sortMode;

    // Op Attr
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensNumType;
    int64_t quantMode;
    int64_t expertStart;
    int64_t expertEnd;
    int64_t rowIdxType;

    bool expertTokensNumFlag;
};

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

inline static int64_t CeilLog4(int64_t x)
{
    if (x <= 1) return 0;
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

void PrintMoeV3Arch35VBSComputeTilingData(const MoeV3Arch35VBSComputeTilingData &data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35VBSComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35VMSMiddleComputeTilingData(const MoeV3Arch35VMSMiddleComputeTilingData &data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35VMSMiddleComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
}

void PrintMoeV3Arch35SortOutComputeTilingData(const MoeV3Arch35SortOutComputeTilingData &data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35SortOutComputeTilingData ==========");
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35ExpertTokensCountTilingData(const MoeV3Arch35ExpertTokensCountTilingData &data,
                                                 const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35ExpertTokensCountTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
}

void PrintMoeV3Arch35GatherOutComputeTilingData(const MoeV3Arch35GatherOutComputeTilingData &data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35GatherOutComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreIndicesElements is %ld", data.perCoreIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesElements is %ld", data.lastCoreIndicesElements);
    OP_LOGD(nodeName, "perCoreIndicesLoops is %ld", data.perCoreIndicesLoops);
    OP_LOGD(nodeName, "perCorePerLoopIndicesElements is %ld", data.perCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "perCoreLastLoopIndicesElements is %ld", data.perCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesLoops is %ld", data.lastCoreIndicesLoops);
    OP_LOGD(nodeName, "lastCorePerLoopIndicesElements is %ld", data.lastCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreLastLoopIndicesElements is %ld", data.lastCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "colsLoops is %ld", data.colsLoops);
    OP_LOGD(nodeName, "perLoopCols is %ld", data.perLoopCols);
    OP_LOGD(nodeName, "lastLoopCols is %ld", data.lastLoopCols);
}

void PrintMoeInitRoutingV3Arch35TilingData(const MoeInitRoutingV3Arch35TilingData &data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeInitRoutingV3Arch35TilingData ==========");
    OP_LOGD(nodeName, "coreNum is %ld", data.coreNum);
    OP_LOGD(nodeName, "BS is %ld", data.n);
    OP_LOGD(nodeName, "H is %ld", data.cols);
    OP_LOGD(nodeName, "topK is %ld", data.k);
    OP_LOGD(nodeName, "expertStart is %ld", data.expertStart);
    OP_LOGD(nodeName, "expertEnd is %ld", data.expertEnd);
    OP_LOGD(nodeName, "actualExpertNum is %ld", data.actualExpertNum);
    OP_LOGD(nodeName, "quantMode is %ld", data.quantMode);
    OP_LOGD(nodeName, "rowIdxType is %ld", data.rowIdxType);
    OP_LOGD(nodeName, "isInputScale is %s", data.isInputScale ? "True" : "False");
    OP_LOGD(nodeName, "isInputOffset is %s", data.isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "expertNum is %ld", data.expertNum);
    OP_LOGD(nodeName, "expertTokensNumType is %ld", data.expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %ld", data.expertTokensNumFlag);
    OP_LOGD(nodeName, "gatherFirstFullload is %ld", data.gatherFirstFullload);
    OP_LOGD(nodeName, "epFullload is %ld", data.epFullload);
    OP_LOGD(nodeName, "activeNum is %ld", data.activeNum);
    OP_LOGD(nodeName, "dropPadMode is %ld", data.dropPadMode);
    OP_LOGD(nodeName, "smoothType is %ld", data.smoothType);
    OP_LOGD(nodeName, "InitRouting TilingKey is %ld", data.tilingKey);

    PrintMoeV3Arch35VBSComputeTilingData(data.vbsComputeParamsOp, nodeName);
    PrintMoeV3Arch35VMSMiddleComputeTilingData(data.vmsMiddleComputeParamsOp, nodeName);
    PrintMoeV3Arch35SortOutComputeTilingData(data.sortOutComputeParamsOp, nodeName);
    PrintMoeV3Arch35ExpertTokensCountTilingData(data.expertTokensCountTilingDataOp, nodeName);
    PrintMoeV3Arch35GatherOutComputeTilingData(data.gatherOutComputeParamsOp, nodeName);
}

void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."), return);

    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");

    PrintMoeInitRoutingV3Arch35TilingData(tilingData->moeInitRoutingTilingData, nodeName);

    OP_LOGD(nodeName, "BS is %u", tilingData->m);
    OP_LOGD(nodeName, "H is %u", tilingData->k);
    OP_LOGD(nodeName, "N is %u", tilingData->n);

    OP_LOGD(nodeName, "topK is %u", tilingData->topK);
    OP_LOGD(nodeName, "expertPerRank is %u", tilingData->expertPerRank);
    OP_LOGD(nodeName, "groupListType is %u", tilingData->groupListType);

    OP_LOGD(nodeName, "epWorldSize is %u", tilingData->epWorldSize);
    OP_LOGD(nodeName, "maxOutputSize is %u", tilingData->maxOutputSize);

    OP_LOGD(nodeName, "transX is %s", (tilingData->transX ? "True" : "False"));
    OP_LOGD(nodeName, "transW is %s", (tilingData->transW ? "True" : "False"));
    OP_LOGD(nodeName, "transW2 is %s", (tilingData->transW2 ? "True" : "False"));
    OP_LOGD(nodeName, "noQuant is %s", (tilingData->noQuant ? "True" : "False"));
}

void printWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName)
{
    OP_LOGD(nodeName, "ptrA:                        %ld\n", info->ptrA);
    OP_LOGD(nodeName, "ptrAScale:                   %ld\n", info->ptrAScale);
    OP_LOGD(nodeName, "ptrA2:                       %ld\n", info->ptrA2);
    OP_LOGD(nodeName, "ptrA2Scale:                  %ld\n", info->ptrA2Scale);
    OP_LOGD(nodeName, "ptrFirstTokenPos:            %ld\n", info->ptrFirstTokenPos);
    OP_LOGD(nodeName, "ptrStepIdx:                  %ld\n", info->ptrStepIdx);
    OP_LOGD(nodeName, "ptrTokenChainDescriptors:    %ld\n", info->ptrTokenChainDescriptors);
    OP_LOGD(nodeName, "ptrPosLen:                   %ld\n", info->ptrPosLen);
    OP_LOGD(nodeName, "ptrPosDone:                  %ld\n", info->ptrPosDone);
    OP_LOGD(nodeName, "ptrChainLen:                 %ld\n", info->ptrChainLen);
    OP_LOGD(nodeName, "ptrChainLenReadyFlag:        %ld\n", info->ptrChainLenReadyFlag);
    OP_LOGD(nodeName, "ptrCombineAccum:             %ld\n", info->ptrCombineAccum);
    OP_LOGD(nodeName, "ptrGmm2ReadyFlag:            %ld\n", info->ptrGmm2ReadyFlag);
    OP_LOGD(nodeName, "ptrFlagSwiGluToGmm2:         %ld\n", info->ptrFlagSwiGluToGmm2);
    OP_LOGD(nodeName, "ptrDispatchDoneCores:         %ld\n", info->ptrDispatchDoneCores);
    OP_LOGD(nodeName, "ptrDispatchTileReady:         %ld\n", info->ptrDispatchTileReady);
    OP_LOGD(nodeName, "ptrExpertOrder:               %ld\n", info->ptrExpertOrder);
    OP_LOGD(nodeName, "ptrExpertOrderScratch:        %ld\n", info->ptrExpertOrderScratch);
    OP_LOGD(nodeName, "ptrInitRoutingWorkspace:     %ld\n", info->ptrInitRoutingWorkspace);
    OP_LOGD(nodeName, "workspaceSize:               %ld\n", info->workspaceSize);
}

void printPeermemInfo(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");

    int64_t tokenPerExpertSize =
        tilingData->epWorldSize *
        ops::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize * tilingData->expertPerRank), (int64_t)ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));
    int64_t expertCumsumSize =
        ops::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->expertPerRank, (int64_t)ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));
    int64_t expertIdxSize =
        ops::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->m * tilingData->topK,
                       (int64_t)ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));
    int64_t expertProbSize =
        ops::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->m * tilingData->topK,
                       (int64_t)ALIGN_256) *
        static_cast<int64_t>(sizeof(int16_t));
    int64_t a0Size = static_cast<int64_t>(tilingData->m) * GetA0RowStrideByMode(tilingData->k, tilingData->noQuant);
    int64_t dSize = static_cast<int64_t>(tilingData->m) * tilingData->epWorldSize * tilingData->k *
                    static_cast<int64_t>(sizeof(int16_t));
    int64_t peermemSize =
        PEERMEM_DATA_OFFSET + tokenPerExpertSize + expertCumsumSize + expertIdxSize + expertProbSize + a0Size + dSize;
    OP_LOGD(nodeName, "peermemSize: {%ld}\n", peermemSize);

    int64_t offset = PEERMEM_DATA_OFFSET;
    OP_LOGD(nodeName, "ptrTokenPerExpert: {%ld}\n", offset);
    offset += tokenPerExpertSize;
    OP_LOGD(nodeName, "ptrExpertTokenCumsum: {%ld}\n", offset);
    offset += expertCumsumSize;
    OP_LOGD(nodeName, "offsetExpandedRowIdx: {%ld}\n", offset);
    offset += expertIdxSize;
    OP_LOGD(nodeName, "offsetExpertProb: {%ld}\n", offset);
    offset += expertProbSize;
    OP_LOGD(nodeName, "ptrA0: {%ld}\n", offset);
    offset += a0Size;
    OP_LOGD(nodeName, "ptrD: {%ld}\n", offset);
}

void printAuxiliaryTilingCtx(const MegaMoeAuxTilingContext *ctx, const uint32_t expertPerRank,
                             const uint32_t epWorldSize, const std::vector<int64_t> &expertIdxShape,
                             ge::DataType xDtype, const char *nodeName)
{
    OP_LOGD(nodeName, "========== AuxiliaryTilingCtx ==========");
    OP_LOGD(nodeName, "BS is %ld", ctx->n);
    OP_LOGD(nodeName, "H is %ld", ctx->cols);
    OP_LOGD(nodeName, "topK is %ld", ctx->k);
    OP_LOGD(nodeName, "expertPerRank is %ld", expertPerRank);
    OP_LOGD(nodeName, "epWorldSize is %ld", epWorldSize);

    OP_LOGD(nodeName, "availUbSize is %ld", ctx->availUbSize);
    OP_LOGD(nodeName, "sortLoopMaxElement is %ld", ctx->sortLoopMaxElement);
    OP_LOGD(nodeName, "totalLength is %ld", ctx->totalLength);
    OP_LOGD(nodeName, "activeNum is %ld", ctx->activeNum);
    OP_LOGD(nodeName, "expertCapacity is %ld", ctx->expertCapacity);
    OP_LOGD(nodeName, "expertNum is %ld", ctx->expertNum);

    OP_LOGD(nodeName, "x is %s, inputXDtypeSize = %ld", Ops::Base::ToString(xDtype).c_str(), ctx->inputXDtypeSize);
    OP_LOGD(nodeName, "isInputScale is %s, isInputOffset is %s", ctx->isInputScale ? "True" : "False",
            ctx->isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "dropPadMode is %ld, expertTokensNumType is %ld", ctx->dropPadMode, ctx->expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %s, quantMode is %ld", ctx->expertTokensNumFlag ? "True" : "False",
            ctx->quantMode);

    OP_LOGD(nodeName, "expertIdxShape is [%ld, %ld]", expertIdxShape[0], expertIdxShape[1]);
    OP_LOGD(nodeName, "expertRange is [%ld, %ld]", ctx->expertStart, ctx->expertEnd);
    OP_LOGD(nodeName, "rowIdxType is %ld", ctx->rowIdxType);
}

static int64_t OpQuantModeToInitRoutingQuantMode(const int64_t opQuantMode)
{
    // UNQUANT, STATIC and DYNAMIC are currently unsupported.
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_TYPE_E5M2:
            return 2LL;
        case DISPATCH_QUANT_OUT_TYPE_E4M3FN:
            return 3LL;
        default:
            return -1LL;
    }
    return -1LL;
}

static ge::DataType GetDataTypeByOpQuantMode(const int64_t opQuantMode)
{
    // UNQUANT, STATIC and DYNAMIC are currently unsupported.
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_TYPE_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case DISPATCH_QUANT_OUT_TYPE_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

static int64_t GetOpQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);

    int64_t opQuantMode;
    if (dispatchQuantOutType == (DISPATCH_QUANT_OUT_TYPE_E5M2 + DISPATCH_QUANT_OUT_TYPE_OFFSET)) {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E5M2;
    } else {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E4M3FN;
    }

    return opQuantMode;
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config, MegaMoeTilingData *tilingData,
                             const char *nodeName, const uint32_t aicNum, const uint32_t aivNum)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    if (*dispatchQuantModePtr == DISPATCH_QUANT_MODE_NO_QUANT) {
        // Keep BF16 on the complete-stage baseline. Per-expert context/UB
        // handoffs cannot assume Combine's predecessors are all on AIV1.
        // The output dtype is ignored, including the E5M2 attribute value.
        return GET_TPL_TILING_KEY(static_cast<int64_t>(DISPATCH_QUANT_MODE_NO_QUANT),
                                  static_cast<int64_t>(DISPATCH_QUANT_OUT_TYPE_E4M3FN),
                                  static_cast<int64_t>(MEGA_MOE_PIPELINE_GENERIC));
    }
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);

    // Core-wave is a workload policy, not a catalog/shape whitelist.  Its
    // schedulers use the platform AIC count at runtime.  Select it only when
    // the MXFP-E4M3FN workload can form more than one
    // resource-sized M-group wave; a single-wave workload would pay the
    // histogram/barrier cost without exposing producer/consumer overlap.
    // The device still uses the real post-AllGather histogram to choose
    // identity vs scheduled order and to place exact wave boundaries.
    uint64_t nominalOwnerRows = static_cast<uint64_t>(tilingData->m) * static_cast<uint64_t>(tilingData->topK);
    if (nominalOwnerRows > tilingData->maxOutputSize) {
        nominalOwnerRows = tilingData->maxOutputSize;
    }
    uint64_t estimatedMGroups = ops::CeilDiv(nominalOwnerRows, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS));
    uint64_t gmm1TilesPerMGroup =
        ops::CeilDiv(static_cast<uint64_t>(tilingData->n), static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS));
    uint64_t gmm2TilesPerMGroup =
        ops::CeilDiv(static_cast<uint64_t>(tilingData->k), static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS));
    uint64_t gmm1GroupsPerWave =
        ops::CeilDiv<uint64_t>(static_cast<uint64_t>(aicNum) * static_cast<uint64_t>(4U), gmm1TilesPerMGroup);
    uint64_t gmm2GroupsPerWave = ops::CeilDiv(static_cast<uint64_t>(aicNum), gmm2TilesPerMGroup);
    uint64_t mGroupsPerWave = std::max(gmm1GroupsPerWave, gmm2GroupsPerWave);
    // Every scheduler, readiness target and worker partition in the
    // core-wave kernel is derived from the runtime AIC count.  Do not pin the
    // optimized path to one SKU's 32-core topology: A5 exposes 28 AIC groups
    // and otherwise satisfies the same MXFP capability contract.
    // 共享专家只实现于通用流水线（kernel 侧仅在 !UseCoreWave 下编译共享链），而
    // core-wave / owner-chain / K16 / ragged-N / sparse-decode 都不会发布 shared-ready
    // 标志；一旦被选中，AIC 永远不会置位，AIV 会在 Unpermute 里等一个不存在的标志
    // 而导致整卡死锁(507014)。因此启用共享专家时把全部调度特化一并关掉、强制 generic。
    bool coreWaveCompatible = tilingData->sharedExpertNum == 0U && *dispatchQuantModePtr == DISPATCH_QUANT_MODE_MXFP &&
                              opQuantMode == DISPATCH_QUANT_OUT_TYPE_E4M3FN;
    // Sparse-decode targets launches whose resolved worst-case row count
    // stays strictly below one M-group wave of both GMM stages.
    // maxOutputSize is the reachable upper bound of rows this rank can
    // receive (min of the caller cap and m * EP * min(topK, expertPerRank));
    // the kernel additionally clamps every row range to it.  Below one wave
    // of row groups the per-tile readiness machinery can overlap at most the
    // dispatch feeding that single wave, while its accounting costs
    // (tile-ready resets, per-tile publications and spins, the Combine
    // predecessor pipeline setup) are paid per active expert; decode
    // launches route a handful of rows to most local experts and lose on
    // every term, so one dispatch-complete gate covering all GMM1 input is
    // strictly cheaper there.  The comparison is strict: a launch that can
    // exactly fill a wave keeps the generic protocol and its overlap
    // machinery.  This remains an upper-bound test on maxOutputSize, never
    // an average (m * topK style) estimate, so a skewed routing cannot
    // select the reduced protocol beyond this row bound.
    uint64_t sparseDecodeRows = static_cast<uint64_t>(tilingData->maxOutputSize);
    uint64_t sparseWaveRowCap =
        std::min(gmm1GroupsPerWave, gmm2GroupsPerWave) * static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS);
    bool useSparseDecode =
        coreWaveCompatible && aicNum > 0U && sparseDecodeRows > 0U && sparseDecodeRows < sparseWaveRowCap;
    // Resident-M-group reachability: the device resident branch requires
    // totalMGroups >= aicNum, with totalMGroups = sum_e ceil(rows_e / 256)
    // grouped strictly per expert.  For R rows split across A non-empty
    // experts, sum_e ceil(rows_e / 256) <= ceil(R / 256) + A - 1, and
    // A <= min(expertPerRank, R), so the bound below dominates totalMGroups
    // for every legal routing.  When even this bound cannot fill one AIC
    // wave, the resident (and Full-A) branches are structurally unreachable
    // and core-wave staging would only ever run its sparse
    // (M-group, N-tile) fallback; prefer the generic pipeline instead of
    // paying the wave-scheduling fixed costs with no resident payoff.  The
    // bound proves unreachability only - it makes no claim that generic is
    // faster for every workload it demotes.
    uint64_t residentActiveExpertMax = std::min(static_cast<uint64_t>(tilingData->expertPerRank), sparseDecodeRows);
    uint64_t residentMGroupUpperBound =
        sparseDecodeRows == 0U ? 0U
                               : ops::CeilDiv(sparseDecodeRows, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS)) +
                                     residentActiveExpertMax - 1U;
    bool residentWaveReachable = residentMGroupUpperBound >= static_cast<uint64_t>(aicNum);
    bool useCoreWave = coreWaveCompatible && estimatedMGroups > mGroupsPerWave && residentWaveReachable;
    // A long local top-k chain pays one GM accumulator round-trip and one
    // dependency hand-off for almost every contribution in expert-major
    // Combine.  Select an independently compiled token-owner consumer only
    // when that structural cost is high and there is enough token work to
    // keep every physical AIV busy.  This is deliberately shape/SKU agnostic:
    // decode/sparse launches retain core-wave, while material K8/K16 chains
    // and future workloads with the same cost invariant use owner Combine.
    // Use the guaranteed per-rank contribution count, not ceil(topK/EP).
    // A remainder may land on only some ranks; treating it as present on all
    // ranks would select owner-chain for top-k widths whose average chain is
    // still shorter than the four-row amortization threshold.
    uint64_t averageLocalChain = tilingData->epWorldSize > 0U ? static_cast<uint64_t>(tilingData->topK) /
                                                                    static_cast<uint64_t>(tilingData->epWorldSize)
                                                              : 0U;
    uint64_t sourceTokens = static_cast<uint64_t>(tilingData->epWorldSize) * static_cast<uint64_t>(tilingData->m);
    uint64_t aivWorkers = static_cast<uint64_t>(aivNum);
    bool useOwnerChainCombine = useCoreWave && aicNum != 0U && aivNum != 0U &&
                                tilingData->expertPerRank >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS &&
                                averageLocalChain >= 4U && static_cast<uint64_t>(tilingData->m) >= aivWorkers &&
                                sourceTokens >= aivWorkers * 4U;
    // K16 owner chains have eight local contributions in the common EP2
    // topology.  Compile them independently so the device can fold descriptor
    // arithmetic and use a deeper contribution pipeline without adding exact
    // BS/H/N catalog checks.  Other long-chain workloads retain the generic
    // owner protocol.
    bool useK16OwnerChain = useOwnerChainCombine && tilingData->topK == 16U;
    // Sparse-decode is provably disjoint from every wave-scheduled mode:
    // rows strictly below min(gmm1GroupsPerWave, gmm2GroupsPerWave) * 256
    // bound estimatedMGroups by that same wave minimum, which never exceeds
    // mGroupsPerWave, so the core-wave second-wave condition cannot hold.
    // RAGGED_N_STRIPES inherits that exclusion rather than competing with
    // it: the override below fires only under useK16OwnerChain, which
    // requires owner-chain and therefore core-wave.  Keep the selection
    // order explicit anyway: a below-one-wave decode launch must never fall
    // into the wave-scheduling family.
    uint32_t pipelineMode =
        useSparseDecode
            ? MEGA_MOE_PIPELINE_SPARSE_DECODE
            : (useK16OwnerChain
                   ? MEGA_MOE_PIPELINE_OWNER_CHAIN_K16
                   : (useOwnerChainCombine ? MEGA_MOE_PIPELINE_OWNER_CHAIN
                                           : (useCoreWave ? MEGA_MOE_PIPELINE_CORE_WAVE : MEGA_MOE_PIPELINE_GENERIC)));
    // Independently compile the ragged-N producer/consumer pair. Aligned
    // hidden widths retain their exact original tiling mode and kernel path.
    // Runtime routing still decides whether Full-A/stripe streaming is used;
    // the established whole-M-group fallback remains valid in this key too.
    if (useK16OwnerChain && CanUseRaggedGmm2StripeMask(tilingData->k, tilingData->n / SWIGLU_N_HALF)) {
        pipelineMode = MEGA_MOE_PIPELINE_RAGGED_N_STRIPES;
    }
    return GET_TPL_TILING_KEY(static_cast<int64_t>(*dispatchQuantModePtr), opQuantMode, pipelineMode);
}

static ge::graphStatus CheckInitTilingData(const int64_t expertTokensNumType, int64_t expertNum,
                                           const int64_t dropPadMode, const bool expertTokensNumFlag,
                                           const int64_t quantMode, const int64_t rowIdxType,
                                           const std::vector<int64_t> expertRange, const std::vector<int64_t> xShape,
                                           const std::vector<int64_t> expertIdxShape, const char *nodeName)
{
    OP_TILING_CHECK(
        expertTokensNumType != EXPERT_TOKENS_TYPE_CUMSUM && expertTokensNumType != EXPERT_TOKENS_TYPE_COUNT &&
            expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE,
        OP_LOGE(nodeName,
                "expertTokensNumType(%ld) should be EXPERT_TOKENS_TYPE_CUMSUM(%ld), "
                "EXPERT_TOKENS_TYPE_COUNT(%ld) or EXPERT_TOKENS_TYPE_KEY_VALUE(%ld).",
                expertTokensNumType, EXPERT_TOKENS_TYPE_CUMSUM, EXPERT_TOKENS_TYPE_COUNT, EXPERT_TOKENS_TYPE_KEY_VALUE),
        return ge::GRAPH_FAILED);

    int64_t maxExpertNum = (expertTokensNumType == EXPERT_TOKENS_TYPE_COUNT) ? EXPERT_IDX_MAX : KV_MODE_EXPERT_IDX_MAX;
    OP_TILING_CHECK(
        expertNum <= 0 || expertNum > maxExpertNum,
        OP_LOGE(nodeName, "expertNum out of range, supporting [0, %ld], expertNum is %ld.", maxExpertNum, expertNum),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(dropPadMode != DROP_PAD_MODE_DROPLESS,
                    OP_LOGE(nodeName, "Only support DROP_PAD_MODE_DROPLESS(%ld), but now dropPadMode is %ld.",
                            DROP_PAD_MODE_DROPLESS, dropPadMode),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!expertTokensNumFlag, OP_LOGE(nodeName, "Failed, expertTokensNumFlag must be true."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        quantMode != DISPATCH_QUANT_OUT_TYPE_E5M2 && quantMode != DISPATCH_QUANT_OUT_TYPE_E4M3FN,
        OP_LOGE(nodeName,
                "only support DISPATCH_QUANT_OUT_TYPE_E5M2(%d), DISPATCH_QUANT_OUT_TYPE_E4M3FN(%d), but now is %ld.",
                DISPATCH_QUANT_OUT_TYPE_E5M2, DISPATCH_QUANT_OUT_TYPE_E4M3FN, quantMode),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        rowIdxType != ROW_IDX_SCATTER && rowIdxType != ROW_IDX_GATHER,
        OP_LOGE(nodeName, "Only support ROW_IDX_SCATTER(%ld), ROW_IDX_GATHER(%ld), but now rowIdxType is %ld.",
                ROW_IDX_SCATTER, ROW_IDX_GATHER, rowIdxType),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertRange.size() != TWO_DIMS,
        OP_LOGE(nodeName, "expertRange.size() must be 2, but now expertRange.size() is %zu.", expertRange.size()),
        return ge::GRAPH_FAILED);

    int64_t expertStart = expertRange[0];
    int64_t expertEnd = expertRange[1];
    OP_TILING_CHECK(expertStart < 0 || expertStart >= expertEnd || expertEnd > expertNum,
                    OP_LOGE(nodeName, "expertNum out of range, supporting [%ld, %ld], but now expertNum is %ld.",
                            expertStart, expertEnd, expertNum),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xShape.size() != TWO_DIMS,
                    OP_LOGE(nodeName, "xShape.size() must be 2, but now xShape.size() is %zu.", xShape.size()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertIdxShape.size() != TWO_DIMS || expertIdxShape[0] != xShape[0],
        OP_LOGE(nodeName, "invalid expertIdxShape, expecting expertIdxShape.size() = 2, expertIdxShape[0] is %ld.",
                xShape[0]),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInitTilingData(MegaMoeTilingData &tilingData,
                                                 MoeInitRoutingV3Arch35TilingData &initTilingData,
                                                 const uint32_t aivNum, const int64_t opQuantMode, const char *nodeName)
{
    // Get Param
    auto bs = tilingData.m;
    auto h = tilingData.k;
    auto topK = tilingData.topK;
    auto expertPerRank = tilingData.expertPerRank;
    auto epWorldSize = tilingData.epWorldSize;

    int64_t expertTokensNumType = EXPERT_TOKENS_TYPE_CUMSUM;
    int64_t expertNum = epWorldSize * expertPerRank;
    int64_t dropPadMode = DROP_PAD_MODE_DROPLESS;
    bool expertTokensNumFlag = true;
    int64_t quantMode = opQuantMode;
    int64_t rowIdxType = ROW_IDX_SCATTER;

    const std::vector<int64_t> expertRange = {0, expertNum};
    std::vector<int64_t> xShape = {bs, h};
    std::vector<int64_t> expertIdxShape = {bs, topK};

    bool isInputScale = false;
    bool isInputOffset = false;

    // Check Param
    OP_TILING_CHECK(CheckInitTilingData(expertTokensNumType, expertNum, dropPadMode, expertTokensNumFlag, quantMode,
                                        rowIdxType, expertRange, xShape, expertIdxShape, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "CheckInitTilingData failed."), return ge::GRAPH_FAILED);

    // Set Param
    initTilingData.expertTokensNumType = expertTokensNumType;
    initTilingData.expertNum = expertNum;
    initTilingData.dropPadMode = dropPadMode;
    initTilingData.expertTokensNumFlag = expertTokensNumFlag ? 1 : 0;
    initTilingData.quantMode = OpQuantModeToInitRoutingQuantMode(quantMode);
    initTilingData.rowIdxType = rowIdxType;

    initTilingData.expertStart = expertRange[0];
    initTilingData.expertEnd = expertRange[1];
    initTilingData.actualExpertNum = expertRange[1] - expertRange[0];

    initTilingData.n = bs;
    initTilingData.cols = h;
    initTilingData.k = topK;
    initTilingData.isInputScale = isInputScale ? 1 : 0;
    initTilingData.isInputOffset = isInputOffset ? 1 : 0;

    initTilingData.coreNum = aivNum;
    initTilingData.worldSize = epWorldSize;

    // Reserved
    initTilingData.gatherFirstFullload = 0;
    initTilingData.activeNum = initTilingData.n * initTilingData.k;
    initTilingData.epFullload = 0;  // unsupported
    initTilingData.smoothType = 0;  // unsupported

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAuxiliaryTilingCtx(MoeInitRoutingV3Arch35TilingData &initTilingData,
                                             MegaMoeAuxTilingContext &ctx, const char *nodeName, const uint32_t aivNum,
                                             const uint64_t ubSize, ge::DataType xDtype)
{
    int64_t inputXDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_TILING_CHECK(inputXDtypeSize < 0,
                    OP_LOGE(nodeName, "invalid x dtype(%s), expecting DT_BF16.", Ops::Base::ToString(xDtype).c_str()),
                    return ge::GRAPH_FAILED);

    ctx.aivCoreNum = aivNum;
    ctx.totalUbSize = ubSize;
    ctx.availUbSize = ubSize - SIMT_DCACHE_SIZE;

    ctx.n = initTilingData.n;
    ctx.cols = initTilingData.cols;
    ctx.k = initTilingData.k;
    ctx.inputXDtypeSize = inputXDtypeSize;
    ctx.isInputScale = initTilingData.isInputScale;
    ctx.isInputOffset = initTilingData.isInputOffset;

    ctx.activeNum = initTilingData.activeNum;
    ctx.expertCapacity = DISABLE_EXPERT_CAPACITY;
    ctx.expertNum = initTilingData.expertNum;
    ctx.dropPadMode = initTilingData.dropPadMode;
    ctx.expertTokensNumType = initTilingData.expertTokensNumType;
    ctx.expertTokensNumFlag = initTilingData.expertTokensNumFlag;
    ctx.quantMode = initTilingData.quantMode;

    ctx.expertStart = initTilingData.expertStart;
    ctx.expertEnd = initTilingData.expertEnd;
    ctx.rowIdxType = initTilingData.rowIdxType;
    ctx.totalLength = initTilingData.n * initTilingData.k;

    ctx.sortLoopMaxElement =
        ctx.availUbSize / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    ctx.sortLoopMaxElement = std::min(ctx.sortLoopMaxElement, SORT_API_MAX_ELEM);

    return ge::GRAPH_SUCCESS;
}

static void Tiling4VBSOneCore(MoeV3Arch35VBSComputeTilingData *vbsTiling, const MegaMoeAuxTilingContext &ctx)
{
    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = ctx.totalLength;
    vbsTiling->perCoreLoops = ops::CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCorePerLoopElements = std::min(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

static void Tiling4VBSMultiCore(MoeV3Arch35VBSComputeTilingData *vbsTiling, MegaMoeAuxTilingContext &ctx)
{
    int64_t needCoreNum = ops::CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    needCoreNum = std::min(needCoreNum, ctx.aivCoreNum);

    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (ctx.totalLength / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = ctx.totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = ops::CeilDiv(ctx.totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops = ops::CeilDiv(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = ctx.totalLength - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            ops::CeilAlign(ops::CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops), SORT32_ALIGN_ELEMENT);
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
}

static void Tiling4VBSCompute(MoeInitRoutingV3Arch35TilingData *tilingData, MegaMoeAuxTilingContext &ctx)
{
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = ctx.sortLoopMaxElement;

    if (ctx.totalLength <= ctx.sortLoopMaxElement) {
        ctx.sortMode = 0;
        Tiling4VBSOneCore(vbsTiling, ctx);
    } else {
        ctx.sortMode = 1;
        Tiling4VBSMultiCore(vbsTiling, ctx);
    }
    tilingData->tilingKey = ctx.sortMode * SORT_CORE_TILINGKEY_BASE + 1031000;
}

static void Tiling4VMSMiddleCompute(MoeInitRoutingV3Arch35TilingData *tilingData)
{
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    auto *vmsMiddleTiling = &(tilingData->vmsMiddleComputeParamsOp);

    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) {
        vmsMiddleTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = ops::CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum;
}

static void Tiling4SortOutCompute(MoeInitRoutingV3Arch35TilingData *tilingData)
{
    auto *sortOutTiling = &(tilingData->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;
}

static void Tiling4ExpertTokensCountCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
                                            const MegaMoeAuxTilingContext &ctx)
{
    auto *tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = ops::CeilDiv(totalElements, ctx.aivCoreNum);
    int64_t needCoreNum = ops::CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;

    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingData->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE)
                                   ? tilingData->actualExpertNum
                                   : (tilingData->actualExpertNum + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (ctx.availUbSize -
         ops::CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));

    int64_t perCoreLoops = ops::CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = ops::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = ops::CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = ops::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

static int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols, int64_t inputXDtypeSize, int64_t availUbSize)
{
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize) + AlignBytes(perLoopCols, sizeof(int8_t));
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    return (availUbSize - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

static void Tiling4GatherOutMxQuant(MoeInitRoutingV3Arch35TilingData *tilingData, const MegaMoeAuxTilingContext &ctx)
{
    auto *gatherOutTiling = &(tilingData->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = ops::CeilDiv(ctx.totalLength, ctx.aivCoreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = ops::CeilDiv(ctx.totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = ctx.totalLength - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = ops::CeilAlign(tilingData->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = ops::CeilAlign(ops::CeilDiv(perLoopCols, NUM_TWO), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    }
    int64_t colsLoops = ops::CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = ops::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = ops::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

MoeInitRoutingV3Arch35TilingData ComputeMoeInitRoutingV3Tiling(MoeInitRoutingV3Arch35TilingData &tilingData,
                                                               MegaMoeAuxTilingContext &ctx)
{
    Tiling4VBSCompute(&tilingData, ctx);
    Tiling4VMSMiddleCompute(&tilingData);
    Tiling4SortOutCompute(&tilingData);
    Tiling4ExpertTokensCountCompute(&tilingData, ctx);
    Tiling4GatherOutMxQuant(&tilingData, ctx);  // support fp8_e4m3fn, fp8_e5m2 currently
    return tilingData;
}

static ge::graphStatus GetMoeInitRoutingV3Tiling(MegaMoeTilingData &tilingData, const uint32_t aivNum,
                                                 const uint64_t ubSize, const int64_t opQuantMode, ge::DataType xDtype,
                                                 const char *nodeName)
{
    MoeInitRoutingV3Arch35TilingData initTilingData;
    OP_TILING_CHECK(
        CheckAndSetInitTilingData(tilingData, initTilingData, aivNum, opQuantMode, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "SetInitTilingData failed."), return ge::GRAPH_FAILED);

    MegaMoeAuxTilingContext ctx;
    OP_TILING_CHECK(SetAuxiliaryTilingCtx(initTilingData, ctx, nodeName, aivNum, ubSize, xDtype) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "SetAuxiliaryTilingCtx failed."), return ge::GRAPH_FAILED);

    printAuxiliaryTilingCtx(&ctx, tilingData.expertPerRank, tilingData.epWorldSize, {tilingData.m, tilingData.topK},
                            xDtype, nodeName);

    auto initRoutingTilingData = ComputeMoeInitRoutingV3Tiling(initTilingData, ctx);

    int64_t leastCoreNum = std::max(
        std::max(initTilingData.vbsComputeParamsOp.needCoreNum, initTilingData.vmsMiddleComputeParamsOp.needCoreNum),
        std::max(initTilingData.expertTokensCountTilingDataOp.needCoreNum,
                 initTilingData.gatherOutComputeParamsOp.needCoreNum));
    bool isCoreNumValid = initTilingData.coreNum >= leastCoreNum;
    OP_TILING_CHECK(!isCoreNumValid,
                    OP_LOGE(nodeName, "aivNum(%ld) < leastCoreNum(%ld).", initTilingData.coreNum, leastCoreNum),
                    return ge::GRAPH_FAILED);

    tilingData.moeInitRoutingTilingData = initRoutingTilingData;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto globalBs = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));

    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr <= 0,
                    OP_LOGE(nodeName, "cclBufferSizePtr is nullptr or 0 Bytes."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr, OP_LOGE(nodeName, "maxRecvTokenNumPtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE(nodeName, "dispatchQuantModePtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutTypePtr == nullptr, OP_LOGE(nodeName, "dispatchQuantOutTypePtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr, OP_LOGE(nodeName, "combineQuantModePtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBs == nullptr, OP_LOGE(nodeName, "globalBs is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);

    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t n = weightOneStorageShape->GetStorageShape().GetDim(1);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
                    OP_LOGE(nodeName, "epWorldSize should be in [%ld, %ld], but now is %ld.", MIN_EP_WORLD_SIZE,
                            MAX_EP_WORLD_SIZE, epWorldSize),
                    return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    const bool expertCountOverflow =
        expertPerRank > 0 && static_cast<uint64_t>(expertPerRank) >
                                 static_cast<uint64_t>(MAX_MOE_EXPERT_NUM) / static_cast<uint64_t>(epWorldSize);
    uint64_t expectedExpertNum = expertPerRank > 0 && !expertCountOverflow
                                     ? static_cast<uint64_t>(expertPerRank) * static_cast<uint64_t>(epWorldSize)
                                     : 0U;
    OP_TILING_CHECK(
        expertPerRank <= 0 || expertCountOverflow || moeExpertNum <= 0 || moeExpertNum > MAX_MOE_EXPERT_NUM ||
            static_cast<uint64_t>(moeExpertNum) != expectedExpertNum,
        OP_LOGE(nodeName, "moeExpertNum(%ld) must equal expertPerRank(%ld) * epWorldSize(%ld) and not exceed %ld.",
                moeExpertNum, expertPerRank, epWorldSize, MAX_MOE_EXPERT_NUM),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topK <= 0 || topK > moeExpertNum,
                    OP_LOGE(nodeName, "topK(%ld) should be in [1, moeExpertNum(%ld)].", topK, moeExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(h <= 0 || n <= 0 || static_cast<uint64_t>(h) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(n) > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(nodeName, "H(%ld) and N(%ld) must be positive uint32 values.", h, n),
                    return ge::GRAPH_FAILED);
    // SecondBuffInit allocates four BF16 rows and one world-size rank-flag
    // buffer in the same 192 KiB AIV UB.  Account for both 32-byte-aligned
    // regions.  This is also stricter than Combine's three-row minimum.
    const uint64_t dataResBytesRaw = static_cast<uint64_t>(h) * UNPERMUTE_LIST_NUM * sizeof(uint16_t);
    const uint64_t rankFlagBytesRaw = static_cast<uint64_t>(epWorldSize) * sizeof(int32_t);
    const uint64_t dataResBytes = (dataResBytesRaw + ALIGN_32 - 1U) / ALIGN_32 * ALIGN_32;
    const uint64_t rankFlagBytes = (rankFlagBytesRaw + ALIGN_32 - 1U) / ALIGN_32 * ALIGN_32;
    OP_TILING_CHECK(
        dataResBytes + rankFlagBytes > BUFFER_ALIGN,
        OP_LOGE(nodeName, "H(%ld), epWorldSize(%ld) exceed the AIV UB budget: data=%lu, rankFlag=%lu, limit=%u.", h,
                epWorldSize, dataResBytes, rankFlagBytes, BUFFER_ALIGN),
        return ge::GRAPH_FAILED);
    const bool noQuant = IsNoQuantMode(context, config);
    OP_TILING_CHECK(noQuant && h % NO_QUANT_H_ALIGN != 0,
                    OP_LOGE(nodeName, "H(%ld) must be a multiple of %ld in BF16 no-quant mode.", h, NO_QUANT_H_ALIGN),
                    return ge::GRAPH_FAILED);
    const uint64_t dispatchCopyRowBytes = static_cast<uint64_t>(GetA0RowStrideByMode(h, noQuant));
    OP_TILING_CHECK(
        dispatchCopyRowBytes > DISPATCH_COPY_FALLBACK_SLOT_BYTES,
        OP_LOGE(nodeName, "H(%ld) requires %lu bytes per Dispatch copy slot, exceeding the %u-byte %u-slot ring.", h,
                dispatchCopyRowBytes, DISPATCH_COPY_FALLBACK_SLOT_BYTES, DISPATCH_COPY_FALLBACK_BUFFER_NUM),
        return ge::GRAPH_FAILED);

    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    int64_t maxRecvTokenNum = static_cast<int64_t>(*maxRecvTokenNumPtr);
    uint64_t totalRoutedRows = static_cast<uint64_t>(bs) * epWorldSize * topK;
    OP_TILING_CHECK(
        maxRecvTokenNum < 0 || (maxRecvTokenNum > 0 && static_cast<uint64_t>(maxRecvTokenNum) > totalRoutedRows),
        OP_LOGE(nodeName, "maxRecvTokenNum(%ld) should be in [0, total routed rows(%lu)].", maxRecvTokenNum,
                totalRoutedRows),
        return ge::GRAPH_FAILED);

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(
        dispatchQuantMode != DISPATCH_QUANT_MODE_MXFP && dispatchQuantMode != DISPATCH_QUANT_MODE_NO_QUANT,
        OP_LOGE(nodeName, "Invalid dispatchQuantMode, only support no-quant bf16(%ld) and mxfp(%ld), but now is %ld.",
                DISPATCH_QUANT_MODE_NO_QUANT, DISPATCH_QUANT_MODE_MXFP, dispatchQuantMode),
        return ge::GRAPH_FAILED);

    // The dispatch output type only describes the MXFP8 dispatch payload; the
    // no-quant path dispatches BF16 and ignores it.
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);
    OP_TILING_CHECK(
        !noQuant && dispatchQuantOutType != (DISPATCH_QUANT_OUT_TYPE_E5M2 + DISPATCH_QUANT_OUT_TYPE_OFFSET) &&
            dispatchQuantOutType != (DISPATCH_QUANT_OUT_TYPE_E4M3FN + DISPATCH_QUANT_OUT_TYPE_OFFSET),
        OP_LOGE(nodeName, "Invalid dispatchQuantOutType, only support fp8_e5m2(23) and fp8_e4m3fn(24), but now is %ld.",
                dispatchQuantOutType),
        return ge::GRAPH_FAILED);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType refWeightDataType = noQuant ? ge::DT_BF16 : GetDataTypeByOpQuantMode(opQuantMode);
    OP_TILING_CHECK(
        refWeightDataType == ge::DT_UNDEFINED,
        OP_LOGE(nodeName, "unsupported dispatchQuantMode(%ld), leading out data type to being DT_UNDEFINED.",
                dispatchQuantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(refWeightDataType != weightOneDesc->GetDataType(),
                    OP_LOGE(nodeName, "refWeightDataType(%s) should be equal to weightOne & weightTwo dataType(%s).",
                            Ops::Base::ToString(refWeightDataType).c_str(),
                            Ops::Base::ToString(weightOneDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    OP_TILING_CHECK(
        *combineQuantModePtr != 0,
        OP_LOGE(nodeName, "Invalid combineQuantMode(%ld), expecting default value(0).", *combineQuantModePtr),
        return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
                    OP_LOGE(nodeName,
                            "Invalid commAlg(%s), expecting default value("
                            ").",
                            commAlgPtr),
                    return ge::GRAPH_FAILED);

    auto globalBsPtr = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));
    int64_t globalBs = static_cast<int64_t>(*globalBsPtr);
    if (globalBs != 0) {
        OP_TILING_CHECK(globalBs < 0 || bs * epWorldSize > globalBs || globalBs % epWorldSize != 0,
                        OP_LOGE(nodeName, "globalBs(%ld) either be 0 or (maxBs * EP and mod(globalBs, EP(%ld)) == 0).",
                                globalBs, epWorldSize),
                        return ge::GRAPH_FAILED);
    }

    int64_t yDtypeSize = 2;
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    int64_t cclBufferSize = static_cast<int64_t>(*cclBufferSizePtr);
    // leastCclBufferSize = PEERMEM_DATA_OFFSET + tokenPerExpert + expertTokenCumsum + expandedRowIdx + expertProb + a0
    // + d
    int64_t tokenPerExpertSize = epWorldSize * ops::CeilAlign(epWorldSize * expertPerRank, (int64_t)ALIGN_128) *
                                 static_cast<int64_t>(sizeof(int32_t));
    int64_t expertCumsumSize =
        ops::CeilAlign(epWorldSize * expertPerRank, (int64_t)ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));
    int64_t expertIdxSize =
        ops::CeilAlign(epWorldSize * bs * topK, (int64_t)ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));
    // expertProb在peermem中按bf16布局, 此前按float计账导致leastCclBufferSize虚高2×.
    int64_t expertProbSize =
        ops::CeilAlign(epWorldSize * bs * topK, (int64_t)ALIGN_256) * static_cast<int64_t>(sizeof(int16_t));
    int64_t a0Size = bs * GetA0RowStrideByMode(h, noQuant);
    int64_t dSize = ops::CeilAlign(bs * epWorldSize * h * yDtypeSize, (int64_t)ALIGN_128);
    int64_t leastCclBufferSize =
        PEERMEM_DATA_OFFSET + tokenPerExpertSize + expertCumsumSize + expertIdxSize + expertProbSize + a0Size + dSize;

    OP_LOGD(nodeName, "PEERMEM_DATA_OFFSET is %ld", (int64_t)PEERMEM_DATA_OFFSET);
    OP_LOGD(nodeName, "epWorldSize is %ld", (int64_t)epWorldSize);
    OP_LOGD(nodeName, "expertPerRank is %ld", (int64_t)expertPerRank);
    OP_LOGD(nodeName, "ALIGN_128 is %ld", (int64_t)ALIGN_128);
    OP_LOGD(nodeName, "sizeof(int32_t) is %ld", (int64_t)sizeof(int32_t));
    OP_LOGD(nodeName, "bs is %ld", (int64_t)bs);
    OP_LOGD(nodeName, "topK is %ld", (int64_t)topK);
    OP_LOGD(nodeName, "h is %ld", (int64_t)h);
    OP_LOGD(nodeName, "MXFP_SCALE_GROUP_NUM is %ld", (int64_t)MXFP_SCALE_GROUP_NUM);
    OP_LOGD(nodeName, "sizeof(int8_t) is %ld", (int64_t)sizeof(int8_t));
    OP_LOGD(nodeName, "yDtypeSize is %ld", (int64_t)yDtypeSize);

    // 各项中间结果
    OP_LOGD(nodeName, "tokenPerExpertSize is %ld", tokenPerExpertSize);
    OP_LOGD(nodeName, "expertCumsumSize is %ld", expertCumsumSize);
    OP_LOGD(nodeName, "expertIdxSize is %ld", expertIdxSize);
    OP_LOGD(nodeName, "expertProbSize is %ld", expertProbSize);
    OP_LOGD(nodeName, "a0Size is %ld", a0Size);
    OP_LOGD(nodeName, "dSize is %ld", dSize);

    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE(nodeName, "cclBufferSize(%ld) should equal or larger than leastCclBufferSize(%ld).",
                            cclBufferSize, leastCclBufferSize),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                     MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto activationClampPtr = attrs->GetAttrPointer<float>(config.attrActivationClampIndex);
    // The OpDef supplies the -1.0 disabled default. Missing it can indicate incompatible
    // generated artifacts; do not silently run a requested clamp as disabled.
    OP_TILING_CHECK(activationClampPtr == nullptr, OP_LOGE(nodeName, "activation_clamp attribute is missing."),
                    return ge::GRAPH_FAILED);
    float activationClamp = *activationClampPtr;
    if (activationClamp == std::numeric_limits<float>::infinity()) {
        activationClamp = -1.0f;
    }
    OP_TILING_CHECK(!std::isfinite(activationClamp) || (activationClamp < 0.0f && activationClamp != -1.0f),
                    OP_LOGE(nodeName, "activation_clamp must be nonnegative or the disabled sentinel -1."),
                    return ge::GRAPH_FAILED);
    if (activationClamp >= 0.0f) {
        // DeepGEMM clamps BF16 GMM1 outputs against a BF16 scalar. Match its
        // round-to-nearest-even threshold while storing the result as float32.
        // Values may round to zero or +inf: both remain enabled thresholds.
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(activationClamp), "float32 expected");
        std::memcpy(&bits, &activationClamp, sizeof(bits));
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        bits &= 0xffff0000U;
        std::memcpy(&activationClamp, &bits, sizeof(bits));
    }
    tilingData->activationClamp = activationClamp;

    tilingData->epWorldSize = *epWorldSizePtr;
    const int64_t requestedMaxOutputSize = *maxRecvTokenNumPtr;
    const uint64_t automaticMaxOutputSize = static_cast<uint64_t>(tilingData->m) * tilingData->epWorldSize *
                                            std::min(tilingData->topK, tilingData->expertPerRank);
    const uint64_t requestedOutputSize =
        requestedMaxOutputSize > 0 ? static_cast<uint64_t>(requestedMaxOutputSize) : automaticMaxOutputSize;
    // Some existing callers pass a conservative capacity larger than the legal-input maximum.
    // Preserve that API behavior while sizing the internal workspace to the reachable bound.
    const uint64_t maxOutputSize = std::min(requestedOutputSize, automaticMaxOutputSize);
    OP_TILING_CHECK(
        requestedMaxOutputSize < 0 || maxOutputSize > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        OP_LOGE(nodeName,
                "maxRecvTokenNum(%ld) resolves to maxOutputSize(%lu), which must fit the kernel "
                "int32 row-index limit.",
                requestedMaxOutputSize, maxOutputSize),
        return ge::GRAPH_FAILED);
    tilingData->maxOutputSize = static_cast<uint32_t>(maxOutputSize);
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->dispatchRows = 2;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
                                                 MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, WorkspaceInfo &workspaceInfo, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace is nullptr."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(sysWorkspaceSize < 0 || workspaceInfo.workspaceSize <= 0,
                    OP_LOGE(nodeName, "Invalid workspace size: sys=%ld, operator=%ld.", sysWorkspaceSize,
                            workspaceInfo.workspaceSize),
                    return ge::GRAPH_FAILED);

    constexpr int64_t maxWorkspaceSize = std::numeric_limits<int64_t>::max();
    OP_TILING_CHECK(sysWorkspaceSize > maxWorkspaceSize - RESERVED_WORKSPACE_SIZE ||
                        workspaceInfo.workspaceSize > maxWorkspaceSize - RESERVED_WORKSPACE_SIZE - sysWorkspaceSize,
                    OP_LOGE(nodeName, "Workspace size overflow: sys=%ld, operator=%ld, reserved=%ld.", sysWorkspaceSize,
                            workspaceInfo.workspaceSize, RESERVED_WORKSPACE_SIZE),
                    return ge::GRAPH_FAILED);
    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = static_cast<size_t>(workspaceSize);

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                             const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    // BF16 no-quant weights carry no MX scales; the scale lists are ignored.
    if (!IsNoQuantMode(context, config)) {
        auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
        auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    }

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE(nodeName, "xActiveMaskDesc should be null."),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *timerShape =
        scalesDesc == nullptr ? nullptr : context->GetOptionalInputShape(config.scalesIndex);
    OP_TILING_CHECK(
        scalesDesc != nullptr && (scalesDesc->GetDataType() != ge::DT_INT64 || timerShape == nullptr ||
                                  timerShape->GetStorageShape().GetDimNum() != 1U ||
                                  timerShape->GetStorageShape().GetDim(0) !=
                                      AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE),
        OP_LOGE(nodeName,
                "the reserved scales slot is used by Timer V2 and must be a 1D int64 tensor with %d elements, "
                "but got dtype=%d, dim=%zu, size=%ld; "
                "please rebuild the profile package and scripts from the same version.",
                AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE,
                scalesDesc == nullptr ? -1 : static_cast<int>(scalesDesc->GetDataType()),
                timerShape == nullptr ? 0UL : timerShape->GetStorageShape().GetDimNum(),
                timerShape == nullptr || timerShape->GetStorageShape().GetDimNum() == 0U
                    ? -1L
                    : timerShape->GetStorageShape().GetDim(0)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_TILING_CHECK(weightOneStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE(nodeName, "weightOneStorageShape dims must be 3, but current dim num is %zu.",
                            weightOneStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightOneDim2 = weightOneStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightOne dim0 = %ld", weightOneDim0);
    OP_LOGD(nodeName, "weightOne dim1 = %ld", weightOneDim1);
    OP_LOGD(nodeName, "weightOne dim2 = %ld", weightOneDim2);

    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weightTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    OP_TILING_CHECK(weightTwoStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE(nodeName, "weightTwoStorageShape dims must be 3, but current dim num is %zu.",
                            weightTwoStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t weightTwoDim0 = weightTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightTwoDim1 = weightTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightTwoDim2 = weightTwoStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightTwo dim0 = %ld", weightTwoDim0);
    OP_LOGD(nodeName, "weightTwo dim1 = %ld", weightTwoDim1);
    OP_LOGD(nodeName, "weightTwo dim2 = %ld", weightTwoDim2);

    OP_TILING_CHECK(
        weightOneDim0 != weightTwoDim0,
        OP_LOGE(nodeName, "weightOneDim0(%ld) and weightTwoDim0(%ld) should be equal.", weightOneDim0, weightTwoDim0),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightOneDim2 != weightTwoDim1 || h != weightOneDim2,
                    OP_LOGE(nodeName, "weightOneDim2(%ld) and weightTwoDim1(%ld) should be equal to h(%ld).",
                            weightOneDim2, weightTwoDim1, h),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDim1 != weightTwoDim2 * NUM_TWO,
                    OP_LOGE(nodeName, "weightOneDim1(%ld) should be equal to weightTwoDim2(%ld) * 2.", weightOneDim1,
                            weightTwoDim2),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightScalesTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                  const char *nodeName)
{
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(config.weightScalesOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneStorageShape);
    OP_TILING_CHECK(weightScalesOneStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
                    OP_LOGE(nodeName, "weightScalesOneStorageShape dims must be 4, but current dim num is %zu.",
                            weightScalesOneStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t weightScalesOneDim0 = weightScalesOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesOneDim1 = weightScalesOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesOneDim2 = weightScalesOneStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesOneDim3 = weightScalesOneStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesOne dim0 = %ld", weightScalesOneDim0);
    OP_LOGD(nodeName, "weightScalesOne dim1 = %ld", weightScalesOneDim1);
    OP_LOGD(nodeName, "weightScalesOne dim2 = %ld", weightScalesOneDim2);
    OP_LOGD(nodeName, "weightScalesOne dim3 = %ld", weightScalesOneDim3);

    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(config.weightScalesTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoStorageShape);
    OP_TILING_CHECK(weightScalesTwoStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
                    OP_LOGE(nodeName, "weightScalesTwoStorageShape dims must be 4, but current dim num is %zu.",
                            weightScalesTwoStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t weightScalesTwoDim0 = weightScalesTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesTwoDim1 = weightScalesTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesTwoDim2 = weightScalesTwoStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesTwoDim3 = weightScalesTwoStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesTwo dim0 = %ld", weightScalesTwoDim0);
    OP_LOGD(nodeName, "weightScalesTwo dim1 = %ld", weightScalesTwoDim1);
    OP_LOGD(nodeName, "weightScalesTwo dim2 = %ld", weightScalesTwoDim2);
    OP_LOGD(nodeName, "weightScalesTwo dim3 = %ld", weightScalesTwoDim3);

    OP_TILING_CHECK(weightScalesOneDim0 != weightScalesTwoDim0,
                    OP_LOGE(nodeName, "weightScalesOneDim0(%ld) and weightScalesTwoDim0(%ld) should be equal.",
                            weightScalesOneDim0, weightScalesTwoDim0),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDim3 != NUM_TWO || weightScalesOneDim3 != weightScalesTwoDim3,
                    OP_LOGE(nodeName, "weightScalesOneDim3(%ld) and weightScalesTwoDim3(%ld) should be equal to 2.",
                            weightScalesOneDim3, weightScalesTwoDim3),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightScalesTwoDim1 != h,
                    OP_LOGE(nodeName, "weightScalesTwoDim1(%ld) should be equal to h(%ld).", weightScalesTwoDim1, h),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        weightScalesOneDim2 != ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE(nodeName,
                "weightScalesOneDim2(%ld) should equal CeilDiv(h(%ld), INPUT_WEIGHT_SCALES_CEIL_ALIGN(%ld)) = %ld.",
                weightScalesOneDim2, h, INPUT_WEIGHT_SCALES_CEIL_ALIGN,
                ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN)),
        return ge::GRAPH_FAILED);

    const int64_t n = weightScalesOneDim1;
    OP_TILING_CHECK(
        weightScalesTwoDim2 != ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE(nodeName,
                "weightScalesTwoDim2(%ld) should equal CeilDiv(n(%ld) / 2, INPUT_WEIGHT_SCALES_CEIL_ALIGN(%ld)) = %ld.",
                weightScalesTwoDim2, n, INPUT_WEIGHT_SCALES_CEIL_ALIGN,
                ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 共享专家校验（合列表协议）：shared_expert_num=0 即未启用，所有布局与既有版本
// 一致。启用时共享权重是 weight1/weight2（及其 scale 列表）的追加元素（下标 1），
// 原型保持 pre-shared-expert 输入数。MXFP8 路径要求追加 scale，BF16 no-quant 忽略。
static ge::graphStatus CheckSharedExpertParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                               MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertNumIndex);
    const int64_t sharedExpertNum = sharedExpertNumPtr == nullptr ? 0 : *sharedExpertNumPtr;
    OP_TILING_CHECK(sharedExpertNum < 0,
                    OP_LOGE(nodeName, "shared_expert_num(%ld) must not be negative.", sharedExpertNum),
                    return ge::GRAPH_FAILED);
    // 先落字段：未启用时两者清零，后续布局计算与既有版本逐字节一致。
    tilingData->sharedExpertNum = static_cast<uint32_t>(sharedExpertNum);
    tilingData->sharedN = 0U;
    if (sharedExpertNum == 0) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t h = static_cast<int64_t>(tilingData->k);
    const gert::StorageShape *sharedWeightOneShape = context->GetDynamicInputShape(config.weightOneIndex, 1);
    OP_TILING_CHECK(sharedWeightOneShape == nullptr ||
                        sharedWeightOneShape->GetStorageShape().GetDimNum() != static_cast<size_t>(THREE_DIMS) ||
                        sharedWeightOneShape->GetStorageShape().GetDim(0) != sharedExpertNum,
                    OP_LOGE(nodeName,
                            "enabled shared experts (shared_expert_num=%ld) require weight1 entry 1 of shape "
                            "[shared_expert_num, sharedN, H]; entry 1 is %s.",
                            sharedExpertNum, sharedWeightOneShape == nullptr ? "absent" : "malformed"),
                    return ge::GRAPH_FAILED);
    const int64_t sharedN = sharedWeightOneShape->GetStorageShape().GetDim(1);
    const int64_t sharedK = sharedN / static_cast<int64_t>(SWIGLU_N_HALF);
    tilingData->sharedN = static_cast<uint32_t>(sharedN);

    OP_TILING_CHECK(sharedExpertNum > MAX_SHARED_EXPERT_NUM,
                    OP_LOGE(nodeName, "sharedExpertNum(%ld) exceeds the supported maximum %ld.", sharedExpertNum,
                            MAX_SHARED_EXPERT_NUM),
                    return ge::GRAPH_FAILED);
    // sharedN/sharedK 的对齐粒度随路径变化：MXFP8 受 64 元素 scale 组约束，
    // BF16 no-quant 只要求与 H 相同的 16 元素对齐（GMM 走 BF16 tile）。
    const int64_t sharedAlign = tilingData->noQuant ? NO_QUANT_H_ALIGN : INPUT_WEIGHT_SCALES_CEIL_ALIGN;
    OP_TILING_CHECK(sharedN <= 0 || sharedN % sharedAlign != 0 || sharedK % sharedAlign != 0,
                    OP_LOGE(nodeName, "sharedN(%ld) and sharedN/2(%ld) must be positive multiples of %ld.", sharedN,
                            sharedK, sharedAlign),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *sharedWeightTwoShape = context->GetDynamicInputShape(config.weightTwoIndex, 1);
    OP_TILING_CHECK(sharedWeightTwoShape == nullptr,
                    OP_LOGE(nodeName, "weight2 must carry the shared block (entry 1) together with weight1."),
                    return ge::GRAPH_FAILED);
    // 追加项 dtype 必须与路由项一致（同一描述符数组、同一 kernel 消费路径）。
    auto routedWeightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto sharedWeightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 1);
    auto sharedWeightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 1);
    OP_CHECK_NULL_WITH_CONTEXT(context, routedWeightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedWeightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedWeightTwoDesc);
    OP_TILING_CHECK(sharedWeightOneDesc->GetDataType() != routedWeightOneDesc->GetDataType() ||
                        sharedWeightTwoDesc->GetDataType() != routedWeightOneDesc->GetDataType(),
                    OP_LOGE(nodeName, "shared weight entries must share the routed dtype(%s), but are %s / %s.",
                            Ops::Base::ToString(routedWeightOneDesc->GetDataType()).c_str(),
                            Ops::Base::ToString(sharedWeightOneDesc->GetDataType()).c_str(),
                            Ops::Base::ToString(sharedWeightTwoDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    if (tilingData->noQuant) {
        OP_TILING_CHECK(routedWeightOneDesc->GetDataType() != ge::DT_BF16,
                        OP_LOGE(nodeName, "weight dataType should be DT_BF16 in no-quant mode, but is %s.",
                                Ops::Base::ToString(routedWeightOneDesc->GetDataType()).c_str()),
                        return ge::GRAPH_FAILED);
    }
    if (!tilingData->noQuant) {
        // MXFP8：共享 scale 必须作为 scale 列表的追加元素（下标 1）提供；
        // 缺失时 kernel 对描述符数组取下标 1 会越界。
        OP_TILING_CHECK(context->GetDynamicInputShape(config.weightScalesOneIndex, 1) == nullptr ||
                            context->GetDynamicInputShape(config.weightScalesTwoIndex, 1) == nullptr,
                        OP_LOGE(nodeName,
                                "shared experts in MXFP8 mode require the shared scale block as "
                                "entry 1 of weight_scales1 / weight_scales2."),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(sharedWeightOneShape->GetStorageShape().GetDim(2) != h,
                    OP_LOGE(nodeName, "shared weight1 entry shape must be [%ld, %ld, %ld], but dim2 is %ld.",
                            sharedExpertNum, sharedN, h, sharedWeightOneShape->GetStorageShape().GetDim(2)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sharedWeightTwoShape->GetStorageShape().GetDimNum() != static_cast<size_t>(THREE_DIMS) ||
            sharedWeightTwoShape->GetStorageShape().GetDim(0) != sharedExpertNum ||
            sharedWeightTwoShape->GetStorageShape().GetDim(1) != h ||
            sharedWeightTwoShape->GetStorageShape().GetDim(2) != sharedK,
        OP_LOGE(nodeName, "shared weight2 entry shape must be [%ld, %ld, %ld].", sharedExpertNum, h, sharedK),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE(nodeName, "contextShape is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %zu.",
                            contextStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %zu.",
                            xStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE(nodeName, "topkIdsStorageShape is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "topkIdsStorageShape dims must be 2, but current dim num is %zu.",
                            topkIdsStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE(nodeName, "topkWeightsStorageShape is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "topkWeightsStorageShape dims must be 2, but current dim num is %zu.",
                            topkWeightsStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 || xDim0 != topkWeightsDim0,
                    OP_LOGE(nodeName, "xDim0(%ld), topkIdsDim0(%ld), topkWeightsDim0(%ld) should all be equal.", xDim0,
                            topkIdsDim0, topkWeightsDim0),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        topkIdsDim1 != topkWeightsDim1,
        OP_LOGE(nodeName, "topkIdsDim1(%ld), topkWeightsDim1(%ld) should be equal.", topkIdsDim1, topkWeightsDim1),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !IsNoQuantMode(context, config) && CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "optional params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    const bool noQuant = IsNoQuantMode(context, config);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "context dataType is invalid, dataType should be DT_INT32, but is %s.",
                            Ops::Base::ToString(contextDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE(nodeName, "x dataType is invalid, dataType should be DT_BF16, but is %s.",
                            Ops::Base::ToString(xDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "topkIds dataType is invalid, dataType should be DT_INT32, but is %s.",
                            Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE(nodeName, "topkWeights dataType is invalid, dataType should be DT_FLOAT or DT_BF16, but is %s.",
                Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDesc->GetDataType() != weightTwoDesc->GetDataType(),
                    OP_LOGE(nodeName, "weightOne dataType(%s) and weightTwo dataType(%s) should be equal.",
                            Ops::Base::ToString(weightOneDesc->GetDataType()).c_str(),
                            Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    if (noQuant) {
        OP_TILING_CHECK(
            weightOneDesc->GetDataType() != ge::DT_BF16,
            OP_LOGE(nodeName, "weightOne dataType is invalid, dataType should be DT_BF16 in no-quant mode, but is %s.",
                    Ops::Base::ToString(weightOneDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(
        ((weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)),
        OP_LOGE(nodeName,
                "weightOne dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, but is %s.",
                Ops::Base::ToString(weightOneDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)),
        OP_LOGE(nodeName,
                "weightTwo dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, but is %s.",
                Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);
    OP_TILING_CHECK(
        weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName, "weightScalesOne dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
                Ops::Base::ToString(weightScalesOneDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName, "weightScalesTwo dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
                Ops::Base::ToString(weightScalesTwoDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context, MegaMoeConfig &config,
                                         const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkIds format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkWeights format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightOne format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightTwoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightTwo format is invalid."), return ge::GRAPH_FAILED);

    if (IsNoQuantMode(context, config)) {
        return ge::GRAPH_SUCCESS;
    }

    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesOne format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesTwo format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const char *nodeName)
{
    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(xDim0 < MIN_BS || xDim0 > MAX_BS,
                    OP_LOGE(nodeName, "BS should in [%ld, %ld], but now BS is %ld.", MIN_BS, MAX_BS, xDim0),
                    return ge::GRAPH_FAILED);

    // int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    // OP_TILING_CHECK(xDim1 != 4LL * H_BASE && xDim1 != 5LL * H_BASE && xDim1 != 7LL * H_BASE,
    //     OP_LOGE(nodeName, "H only support 4k/5k/7k, but now H is %ld.", xDim1),
    //     return ge::GRAPH_FAILED);

    // const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    // int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    // OP_TILING_CHECK(topkIdsDim1 != 6LL && topkIdsDim1 != 8LL,
    //     OP_LOGE(nodeName, "topK only support 6/8, but now topK is %ld.", topkIdsDim1),
    //     return ge::GRAPH_FAILED);

    // auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    // OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    // int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    // OP_TILING_CHECK(weightOneDim0 < MIN_EXPERT_PER_RANK || weightOneDim0 > MAX_EXPERT_PER_RANK,
    //     OP_LOGE(nodeName,
    //         "expertPerRank should in [%ld, %ld], but now expertPerRank is %ld.",
    //         MIN_EXPERT_PER_RANK, MAX_EXPERT_PER_RANK, weightOneDim0),
    //     return ge::GRAPH_FAILED);

    // int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    // OP_TILING_CHECK(weightOneDim1 != 1LL * N_BASE && weightOneDim1 != 2LL * N_BASE && weightOneDim1 != 3LL * N_BASE
    // &&
    //                 weightOneDim1 != 4LL * N_BASE && weightOneDim1 != 7LL * N_BASE,
    //     OP_LOGE(nodeName, "N only support 1k/2k/3k/4k/7k, but now N is %ld.", weightOneDim1),
    //     return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                     MegaMoeConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t BS = xStorageShape->GetStorageShape().GetDim(0);
    int64_t H = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t N = weightOneStorageShape->GetStorageShape().GetDim(1);

    tilingData->m = static_cast<uint32_t>(BS);
    tilingData->k = static_cast<uint32_t>(H);
    tilingData->n = static_cast<uint32_t>(N);
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->expertPerRank = static_cast<uint32_t>(expertPerRank);
    tilingData->groupListType = 0;

    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;
    tilingData->noQuant = IsNoQuantMode(context, config);
    // 共享专家字段由 CheckSharedExpertParams 依据 shared_expert_num 属性与
    // weight1 追加元素（下标 1）写入：未启用时保持 0，workspace 与 peermem
    // 布局与既有版本逐字节一致。

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                        MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(aivNum <= 0 || aicNum <= 0,
                    OP_LOGE(nodeName, "aivNum(%u) or aicNum(%u) less than or equal 0.", aivNum, aicNum),
                    return ge::GRAPH_FAILED);

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1);  // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // 共享专家校验（未启用 shared_weight1 时为空操作，不改变任何既有行为）
    OP_TILING_CHECK(CheckSharedExpertParams(context, config, tilingData, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Shared expert params are invalid."), return ge::GRAPH_FAILED);

    // InitRouting only sorts expert ids and counts tokens; its quant mode does
    // not touch x, so the no-quant path reuses the E4M3FN routing tiling.
    int64_t opQuantMode = tilingData->noQuant ? static_cast<int64_t>(DISPATCH_QUANT_OUT_TYPE_E4M3FN)
                                              : GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType xDtype = context->GetInputDesc(config.xIndex)->GetDataType();
    OP_TILING_CHECK(
        GetMoeInitRoutingV3Tiling(*tilingData, aivNum, ubSize, opQuantMode, xDtype, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Moe init routing failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName, aicNum, aivNum);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceInfo workspaceInfo((uint8_t *)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceInfo, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMegaMoeTilingData(tilingData, nodeName);
    printWorkspaceInfo(&workspaceInfo, nodeName);
    printPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext *context)
{
    MegaMoeConfig config;
    ge::graphStatus ret;

    ret = MegaMoeTilingFuncImplPublic(context, config);

    return ret;
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ZercMoe).Tiling(MegaMoeTilingFunc).TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);
}  // namespace optiling

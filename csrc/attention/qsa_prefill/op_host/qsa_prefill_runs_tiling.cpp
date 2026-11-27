// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "../op_kernel/qsa_prefill_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace optiling {
namespace {

ge::graphStatus TilingQsaPrefillRuns(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    constexpr size_t kRanks[6] = {3, 3, 3, 2, 2, 1};
    for (size_t i = 0; i < 6; ++i) {
        const auto *shape = context->GetInputShape(i);
        const auto *desc = context->GetInputDesc(i);
        if (!shape || !desc || shape->GetStorageShape().GetDimNum() != kRanks[i]) {
            return ge::GRAPH_FAILED;
        }
        const auto expectedDtype = (i < 3) ? ge::DT_BF16 : ge::DT_INT32;
        if (desc->GetDataType() != expectedDtype) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &k = context->GetInputShape(1)->GetStorageShape();
    const auto &v = context->GetInputShape(2)->GetStorageShape();
    const auto &slots = context->GetInputShape(3)->GetStorageShape();
    const auto &runs = context->GetInputShape(4)->GetStorageShape();
    const auto &counts = context->GetInputShape(5)->GetStorageShape();

    const int64_t groups = q.GetDim(0);
    const int64_t packedHeads = q.GetDim(1);
    const int64_t capacity = k.GetDim(0);
    const int64_t kvHeads = k.GetDim(1);
    const int64_t rows = slots.GetDim(0);
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kTopK = 2064;
    if (groups <= 0 || q.GetDim(2) != kHeadDim || capacity <= 0 || k.GetDim(2) != kHeadDim || v.GetDim(0) != capacity ||
        v.GetDim(1) != kvHeads || v.GetDim(2) != kHeadDim) {
        return ge::GRAPH_FAILED;
    }
    if ((kvHeads != 1 && kvHeads != 2) || packedHeads <= 0 || packedHeads % (16 * kvHeads) != 0 ||
        packedHeads / kvHeads > 128) {
        return ge::GRAPH_FAILED;
    }
    if (rows <= 0 || rows > std::numeric_limits<int32_t>::max() / 16 || slots.GetDim(1) != kTopK ||
        (rows + 15) / 16 != groups || runs.GetDim(0) != rows || runs.GetDim(1) != kTopK ||
        counts.GetDim(0) != rows * 16) {
        return ge::GRAPH_FAILED;
    }
    if (groups > std::numeric_limits<int32_t>::max() / 2 || capacity > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const auto *scaleAttr = attrs ? attrs->GetAttrPointer<float>(0) : nullptr;
    if (!scaleAttr || !std::isfinite(*scaleAttr) || *scaleAttr <= 0.0f) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t aicNum = platform.GetCoreNumAic();
    if (aicNum == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalTaskNum = static_cast<uint32_t>(groups * kvHeads);
    const uint32_t cores = std::min(totalTaskNum, aicNum);

    SglQsaPrefillRunsTilingData td{};
    td.batch = 1;
    td.numHeads = static_cast<uint32_t>(packedHeads);
    td.kvHeads = static_cast<uint32_t>(kvHeads);
    td.embeddingSize = static_cast<uint32_t>(kHeadDim);
    td.blockSize = 256;
    td.topK = static_cast<uint32_t>(kTopK);
    td.maxBlocksPerBatch = static_cast<uint32_t>(capacity);
    td.totalQTokens = static_cast<uint32_t>(groups);
    td.totalTaskNum = totalTaskNum;
    td.firstBatchTaskNum = static_cast<uint32_t>(kvHeads);
    td.scaleValue = *scaleAttr;
    td.maxQSeqlen = static_cast<uint32_t>(rows);
    td.tilingKey = 20002;
    td.groupSize = static_cast<uint32_t>(packedHeads / kvHeads);
    td.mm1OutSize = static_cast<uint64_t>(cores) * 32768u * 4u * 3u;
    td.smOnlineOutSize = static_cast<uint64_t>(cores) * 32768u * 2u * 3u;
    td.mm2OutSize = td.mm1OutSize;
    td.updateSize = td.mm1OutSize;
    const uint64_t libSize = platform.GetLibApiWorkSpaceSize();
    td.workSpaceSize = libSize + ((kTopK * 4u + 511u) / 512u) * 512u + td.mm1OutSize + td.smOnlineOutSize +
                       td.mm2OutSize + td.updateSize;
    td.qBaseTile = 64;
    td.kvBaseTile = 256;
    td.reserved = 0;

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(td) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &td, sizeof(td));
    raw->SetDataSize(sizeof(td));
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    workspace[0] = td.workSpaceSize;
    return ge::GRAPH_SUCCESS;
}

struct SglQsaPrefillRunsCompileInfo {};

ge::graphStatus TilingPrepareQsaPrefillRuns(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(SglQsaPrefillRuns)
    .Tiling(TilingQsaPrefillRuns)
    .TilingParse<SglQsaPrefillRunsCompileInfo>(TilingPrepareQsaPrefillRuns);

}  // namespace optiling

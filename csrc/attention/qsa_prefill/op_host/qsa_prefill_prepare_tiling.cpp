// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "../op_kernel/qsa_prefill_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace optiling {
namespace {

ge::graphStatus TilingQsaPrefillPrepare(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto *blocksShape = context->GetInputShape(0);
    const auto *tableShape = context->GetInputShape(1);
    const auto *reqShape = context->GetInputShape(2);
    const auto *flagsShape = context->GetInputShape(3);
    const auto *blocksDesc = context->GetInputDesc(0);
    const auto *tableDesc = context->GetInputDesc(1);
    const auto *reqDesc = context->GetInputDesc(2);
    const auto *flagsDesc = context->GetInputDesc(3);
    if (!blocksShape || !tableShape || !reqShape || !flagsShape || !blocksDesc || !tableDesc || !reqDesc ||
        !flagsDesc) {
        return ge::GRAPH_FAILED;
    }
    if (blocksDesc->GetDataType() != ge::DT_INT32 || tableDesc->GetDataType() != ge::DT_INT32 ||
        reqDesc->GetDataType() != ge::DT_INT64 || flagsDesc->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    const auto &b = blocksShape->GetStorageShape();
    const auto &t = tableShape->GetStorageShape();
    const auto &req = reqShape->GetStorageShape();
    const auto &flags = flagsShape->GetStorageShape();
    constexpr int64_t kBlocksWidth = 512;
    constexpr int64_t kOutWidth = 2064;
    if (b.GetDimNum() != 2 || b.GetDim(1) != kBlocksWidth || t.GetDimNum() != 1 || t.GetDim(0) <= 0 ||
        req.GetDimNum() != 1 || req.GetDim(0) != 1 || flags.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = b.GetDim(0);
    if (rows <= 0 || rows > std::numeric_limits<int32_t>::max() / 16 || flags.GetDim(0) != rows * 16) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    if (!attrs) {
        return ge::GRAPH_FAILED;
    }
    const auto *tableStrideAttr = attrs->GetAttrPointer<int64_t>(0);
    const auto *lengthAttr = attrs->GetAttrPointer<int64_t>(1);
    const auto *baseAttr = attrs->GetAttrPointer<int64_t>(2);
    const auto *capacityAttr = attrs->GetAttrPointer<int64_t>(3);
    if (!tableStrideAttr || !lengthAttr || !baseAttr || !capacityAttr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tableStride = *tableStrideAttr;
    const int64_t length = *lengthAttr;
    const int64_t base = *baseAttr;
    const int64_t capacity = *capacityAttr;
    if (tableStride < 0 || tableStride > std::numeric_limits<int32_t>::max() || length < 0 || length > t.GetDim(0) ||
        length > std::numeric_limits<int32_t>::max() || base < std::numeric_limits<int32_t>::min() ||
        base + rows >= std::numeric_limits<int32_t>::max() || capacity <= 0 ||
        capacity > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        return ge::GRAPH_FAILED;
    }

    SglQsaPrefillPrepareTilingData tilingData{};
    tilingData.rows = static_cast<uint32_t>(rows);
    tilingData.blocks = static_cast<uint32_t>(kBlocksWidth);
    tilingData.tableStride = static_cast<uint32_t>(tableStride);
    tilingData.length = static_cast<uint32_t>(length);
    tilingData.base = static_cast<int32_t>(base);
    tilingData.cap = static_cast<uint32_t>(capacity);
    tilingData.outWidth = static_cast<uint32_t>(kOutWidth);
    tilingData.reserved = 0;

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(tilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &tilingData, sizeof(tilingData));
    raw->SetDataSize(sizeof(tilingData));
    const uint32_t launchBlocks = std::min<uint32_t>((tilingData.rows + 15u) / 16u, 64u);
    context->SetBlockDim(launchBlocks);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

struct SglQsaPrefillPrepareCompileInfo {};

ge::graphStatus TilingPrepareQsaPrefillPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(SglQsaPrefillPrepare)
    .Tiling(TilingQsaPrefillPrepare)
    .TilingParse<SglQsaPrefillPrepareCompileInfo>(TilingPrepareQsaPrefillPrepare);

}  // namespace optiling

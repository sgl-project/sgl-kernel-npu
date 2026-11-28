// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include "../fused_scatter_copy_sparse_flash_attention_split_tiling_data.h"
#include "../../sparse_flash_attention/op_host/sparse_flash_attention_tiling.h"
#include <cstring>
#include <initializer_list>
#include <limits>

namespace optiling {
namespace {
bool HasShape(const gert::StorageShape *shape, std::initializer_list<int64_t> expected)
{
    if (!shape || shape->GetStorageShape().GetDimNum() != expected.size()) return false;
    size_t axis = 0;
    for (auto size : expected) {
        const auto value = shape->GetStorageShape().GetDim(axis++);
        if (value <= 0 || value > UINT32_MAX || (size >= 0 && value != size)) return false;
    }
    return true;
}

ge::graphStatus TilingFusedScatterCopySparseFlashAttentionSplit(gert::TilingContext *ctx)
{
    if (!ctx || !ctx->GetPlatformInfo() || !ctx->GetAttrs()) return ge::GRAPH_FAILED;
    auto input = [ctx](size_t i) { return ctx->GetRequiredInputShape(i); };
    auto dim = [&](size_t i, size_t axis) { return input(i)->GetStorageShape().GetDim(axis); };
    if (!HasShape(input(0), {-1, -1, 512}) || !HasShape(input(2), {-1, 128, 1, -1}) || !HasShape(input(4), {-1})) {
        return ge::GRAPH_FAILED;
    }
    const auto tokens = dim(0, 0);
    const auto heads = dim(0, 1);
    const auto batch = dim(4, 0);
    const auto pages = dim(2, 0);
    const bool combined = dim(2, 3) == 576;
    if ((heads > 64 && heads != 128) || tokens < batch || tokens > batch * 16 || (!combined && dim(2, 3) != 512) ||
        !HasShape(input(1), {tokens, heads, 64}) || !HasShape(input(3), {pages, 128, 1, combined ? 576 : 64}) ||
        !HasShape(input(5), {batch}) || !HasShape(input(6), {batch}) || !HasShape(input(7), {tokens, 2048}) ||
        !HasShape(input(8), {tokens, 2048}) || !HasShape(input(9), {tokens}) || !HasShape(input(10), {batch, -1}) ||
        !HasShape(input(11), {batch, dim(10, 1)}) || !HasShape(input(12), {batch}) ||
        !HasShape(input(13), {batch, -1}) || dim(13, 1) > UINT32_MAX / 128 || !HasShape(input(14), {batch, -1}))
        return ge::GRAPH_FAILED;

    auto *attrs = ctx->GetAttrs();
    const auto *scale = attrs->GetAttrPointer<float>(0);
    const auto *ckvPtr = attrs->GetAttrPointer<int64_t>(1);
    const auto *ropePtr = attrs->GetAttrPointer<int64_t>(2);
    const auto *visible = attrs->GetAttrPointer<bool>(3);
    const auto *refresh = attrs->GetAttrPointer<bool>(4);
    const auto *hostRows = attrs->GetAttrPointer<int64_t>(5);
    const auto *desc = ctx->GetInputDesc(0);
    if (!scale || !ckvPtr || *ckvPtr <= 0 || !ropePtr || *ropePtr < 0 || (!combined && *ropePtr == 0) || !visible ||
        !refresh || (*refresh && !*visible) || !hostRows || !desc)
        return ge::GRAPH_FAILED;

    CannCopySfaTiling record{};
    auto &data = record.compute;
    const auto *current = ctx->GetOptionalInputShape(15);
    for (size_t i = 16; i <= 19; ++i) {
        if (bool(current) != bool(ctx->GetOptionalInputShape(i))) return ge::GRAPH_FAILED;
    }
    if (current) {
        auto opt = [&](size_t i) { return ctx->GetOptionalInputShape(i); };
        if (!*visible || !HasShape(current, {-1, 1, 576}) || current->GetStorageShape().GetDim(0) < tokens ||
            !HasShape(opt(16), {-1}) || opt(16)->GetStorageShape().GetDim(0) < tokens || !HasShape(opt(17), {-1}) ||
            opt(17)->GetStorageShape().GetDim(0) < tokens || !HasShape(opt(18), {batch}) ||
            !HasShape(opt(19), {-1, -1}) || !HasShape(input(20), {opt(19)->GetStorageShape().GetDim(0)}) ||
            *hostRows <= 0 || *hostRows > UINT32_MAX)
            return ge::GRAPH_FAILED;
        data.residentEnabled = 1;
        data.residentRows = opt(19)->GetStorageShape().GetDim(0);
        data.residentContext = opt(19)->GetStorageShape().GetDim(1);
        data.residentCurrentRows = tokens;
        data.residentHbmPages = pages;
        data.residentHostRows = *hostRows;
    }
    // Copy-SFA exposes the TND, single-head paged MLA specialization of SFA.
    SFATilingInfo info{};
    info.opName = "SglFusedCopySfa";
    info.platformInfo = ctx->GetPlatformInfo();
    info.bSize = info.actualLenDimsQ = info.actualLenDimsKV = batch;
    info.n1Size = info.gSize = heads;
    info.n2Size = 1;
    info.s1Size = info.qTSize = tokens;
    info.blockSize = 128;
    info.blockTypeSize = sizeof(float);
    info.maxBlockNumPerBatch = dim(13, 1);
    info.s2Size = info.maxActualseq = info.maxBlockNumPerBatch * info.blockSize;
    info.totalBlockNum = pages;
    info.qkHeadDim = info.vHeadDim = 512;
    info.ropeHeadDim = 64;
    info.scaleValue = *scale;
    info.sparseBlockCount = 2048;
    info.sparseBlockSize = 1;
    info.sparseMode = 3;
    info.pageAttentionFlag = info.actualQSeqLenFlag = info.actualSeqLenFlag = true;
    info.kvStorageMode = KvStorageMode::PAGE_ATTENTION;
    info.qLayout = info.topkLayout = info.outLayout = SFALayout::TND;
    info.kvLayout = SFALayout::PA_BSND;
    info.inputQType = info.inputKvType = info.inputQRopeType = info.inputKRopeType = info.outputType =
        desc->GetDataType();
    if (info.inputQType != ge::DT_BF16 && info.inputQType != ge::DT_FLOAT16) return ge::GRAPH_FAILED;

    SFAMlaTiling sfa(ctx);
    auto *raw = ctx->GetRawTilingData();
    auto *workspace = ctx->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(record) || !workspace ||
        sfa.DoOpTiling(&info) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    auto &prefix = sfa.GetTilingData();
    if (prefix.GetDataSize() != offsetof(FusedCopySfaTilingData, copyCap)) return ge::GRAPH_FAILED;
    prefix.SaveToBuffer(&data, prefix.GetDataSize());
    data.copyCap = 2048;
    data.missCap = dim(10, 1);
    data.hbmMaxBlockNum = dim(13, 1);
    data.dramMaxBlockNum = dim(14, 1);
    data.fullVisible = *visible;
    data.residentRefreshTail = *refresh;
    data.splitHost = *ropePtr != 0;
    record.dramCkvAddress = static_cast<uint64_t>(*ckvPtr);
    record.dramRopeAddress = *ropePtr != 0 ? static_cast<uint64_t>(*ropePtr) : record.dramCkvAddress + 512 * 2;
    record.combinedHbm = combined;
    std::memcpy(raw->GetData(), &record, sizeof(record));
    raw->SetDataSize(sizeof(record));
    ctx->SetBlockDim(sfa.GetBlockDim());
    ctx->SetTilingKey(0);
    workspace[0] = sfa.GetWorkspaceSizeBytes();
    return ge::GRAPH_SUCCESS;
}
struct CopySfaCompileInfo {};
ge::graphStatus Prepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_OPTILING(SglFusedCopySfa)
    .Tiling(TilingFusedScatterCopySparseFlashAttentionSplit)
    .TilingParse<CopySfaCompileInfo>(Prepare);
}  // namespace optiling

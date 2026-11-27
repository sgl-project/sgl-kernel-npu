// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "lim_launch_geometry.h"
#include "../op_kernel/arch35/lim_simt_config.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

namespace optiling {
static ge::graphStatus InferShape(gert::InferShapeContext *ctx)
{
    const auto *query = ctx->GetInputShape(0);
    const auto *table = ctx->GetInputShape(5);
    const auto *pool = ctx->GetInputShape(11);
    if (!query || !table || !pool || query->GetDimNum() != 3 || table->GetDimNum() != 2) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 7; ++i) {
        auto *out = ctx->GetOutputShape(i);
        if (!out) return ge::GRAPH_FAILED;
        if (i == 6) {
            *out = *pool;
        } else {
            const bool count = i == 2 || i == 5;
            out->SetDimNum(count ? 1 : 2);
            out->SetDim(0, i < 3 ? query->GetDim(0) : table->GetDim(0));
            if (!count) out->SetDim(1, i < 3 ? lim_request_limits::TOPK : lim_request_limits::MISS_CAPACITY);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext *ctx)
{
    for (size_t i = 0; i < 7; ++i)
        ctx->SetOutputDataType(i, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling(gert::TilingContext *ctx)
{
    const auto *q = ctx->GetInputShape(0);
    const auto *table = ctx->GetInputShape(5);
    const auto *pool = ctx->GetInputShape(11);
    auto *raw = ctx->GetRawTilingData();
    auto *workspace = ctx->GetWorkspaceSizes(1);
    auto *attr = ctx->GetAttrs();
    if (!q || !table || !pool || !raw || !raw->GetData() || !workspace || !attr || !ctx->GetPlatformInfo() ||
        raw->GetCapacity() < sizeof(FusedQuantLightningIndexerManageTilingData))
        return ge::GRAPH_FAILED;
    const auto &query = q->GetStorageShape();
    const auto &blocks = table->GetStorageShape();
    const auto &slots = pool->GetStorageShape();
    if (query.GetDimNum() != 3 || blocks.GetDimNum() != 2 || slots.GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto rows = query.GetDim(0), heads = query.GetDim(1), batch = blocks.GetDim(0), pages = blocks.GetDim(1);
    if (batch <= 0 || batch > UINT32_MAX || rows < batch || rows > UINT32_MAX ||
        rows > batch * lim_request_limits::MAX_ROUTES || (heads != 32 && heads != 64) || query.GetDim(2) != 128 ||
        pages <= 0 || pages > lim_payload_layout::MAX_SOURCE_CAPACITY / lim_request_limits::BLOCK_SIZE ||
        slots.GetDim(0) < 3 || slots.GetDim(0) > UINT32_MAX ||
        slots.GetDim(1) != pages * lim_request_limits::BLOCK_SIZE)
        return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    FusedQuantLightningIndexerManageTilingData t{};
    t.usedCoreNum = std::min(platform.GetCoreNumAic(), platform.GetCoreNumAiv() / 2);
    if (!t.usedCoreNum) return ge::GRAPH_FAILED;
    t.batchSize = batch;
    t.totalQueryRows = rows;
    t.poolSize = slots.GetDim(0);
    t.maxBlockNumPerBatch = pages;
    t.sourceCapacity = slots.GetDim(1);
    t.indexHeads = heads;
    uint32_t *strides[] = {&t.weightStride, &t.keyStride, &t.scaleStride};
    for (size_t i = 0; i < 3; ++i) {
        const auto *value = attr->GetAttrPointer<int64_t>(i);
        if (!value || *value <= 0 || *value > UINT32_MAX) return ge::GRAPH_FAILED;
        *strides[i] = *value;
    }
    // The adapter exposes flat physical spans for the three strided inputs.
    // Check their last complete row before using the strides for GM addressing.
    auto physical_rows = [&](size_t input, uint32_t stride, int64_t width) -> int64_t {
        const auto *shape = ctx->GetInputShape(input);
        if (!shape || shape->GetStorageShape().GetDimNum() != 1 || stride < width) return 0;
        const auto elements = shape->GetStorageShape().GetDim(0);
        if (elements < width || (elements - width) % stride != 0) return 0;
        return (elements - width) / stride + 1;
    };
    const auto key_pages = physical_rows(2, t.keyStride, lim_request_limits::BLOCK_SIZE * 128);
    const auto *query_scale = ctx->GetInputShape(3);
    if (physical_rows(1, t.weightStride, heads) != rows || key_pages <= 0 ||
        physical_rows(4, t.scaleStride, lim_request_limits::BLOCK_SIZE) != key_pages || !query_scale ||
        query_scale->GetStorageShape().GetDimNum() != 2 || query_scale->GetStorageShape().GetDim(0) != rows ||
        query_scale->GetStorageShape().GetDim(1) != heads)
        return ge::GRAPH_FAILED;
    for (size_t i = 6; i <= 10; ++i) {
        const auto *shape = ctx->GetInputShape(i);
        if (!shape || shape->GetStorageShape().GetDimNum() != 1 || shape->GetStorageShape().GetDim(0) != batch)
            return ge::GRAPH_FAILED;
    }
    if (!InitQuantLimGeometry(t)) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    ctx->SetTilingKey(0);
    ctx->SetBlockDim(platform.CalcTschBlockDim(t.usedCoreNum * 2, t.usedCoreNum, t.usedCoreNum * 2));
    if (ctx->SetScheduleMode(1) != ge::GRAPH_SUCCESS ||
        ctx->SetLocalMemorySize(LIM_SIMT_USER_UB_BYTES) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    workspace[0] = QuantLimWorkspaceBytes(t, platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class SglFusedQuantLightningIndexerManage : public OpDef
{
public:
    explicit SglFusedQuantLightningIndexerManage(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN}).Format({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN}).Format({ge::FORMAT_ND});
        for (auto *s : {"query_scale", "key_scale"})
            this->Input(s).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        for (auto *s :
             {"block_table", "query_ends", "lengths", "cache_tokens", "request_state", "pool_entries", "slot_pool"})
            this->Input(s).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        for (auto *s : {"topk_sources", "topk_slots", "topk_miss_counts", "miss_sources", "miss_slots", "miss_counts",
                        "slot_pool_out"})
            this->Output(s).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        for (auto *s : {"weight_stride", "key_stride", "scale_stride"})
            this->Attr(s).AttrType(REQUIRED).Int();
        this->SetInferShape(optiling::InferShape).SetInferDataType(optiling::InferDtype);
        this->AICore().SetTiling(optiling::Tiling).AddConfig("ascend950");
    }
};
OP_ADD(SglFusedQuantLightningIndexerManage);
}  // namespace ops

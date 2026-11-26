#include "../op_kernel/fused_lightning_indexer_manage_tiling_data.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstring>

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *ctx)
{
    auto *q = ctx->GetInputShape(1);
    auto *table = ctx->GetInputShape(3);
    auto *pool = ctx->GetInputShape(10);
    auto *attr = ctx->GetAttrs();
    auto *visible = attr ? attr->GetAttrPointer<bool>(0) : nullptr;
    auto *raw = ctx->GetRawTilingData();
    auto *workspace = ctx->GetWorkspaceSizes(1);
    auto *desc = ctx->GetInputDesc(1);
    if (!q || !table || !pool || !visible || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(FusedLightningIndexerManageTilingData) || !workspace || !desc ||
        !ctx->GetPlatformInfo())
        return ge::GRAPH_FAILED;
    auto &qs = q->GetStorageShape();
    auto &ts = table->GetStorageShape();
    auto &ps = pool->GetStorageShape();
    if (qs.GetDimNum() != 3 || ts.GetDimNum() != 2 || ps.GetDimNum() != 2) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    auto aic = platform.GetCoreNumAic();
    auto aiv = platform.GetCoreNumAiv();
    if (!aic || !aiv) return ge::GRAPH_FAILED;
    FusedLightningIndexerManageTilingData t{};
    t.bSize = ts.GetDim(0);
    t.tSize = qs.GetDim(0);
    t.s2Size = ps.GetDim(1);
    t.scheduleMode = t.bSize % aic == 0 ? 2U : 1U;
    auto owners = t.scheduleMode == 1U ? aic : std::min<uint32_t>(t.bSize, aic);
    t.usedCoreNum = platform.CalcTschBlockDim(std::min(aiv, owners * 2U), aic, aiv);
    t.blockSize = 128;
    t.maxBlockNumPerBatch = ts.GetDim(1);
    t.poolSize = ps.GetDim(0);
    t.n1Size = qs.GetDim(1);
    t.cacheSlotsSize = ps.GetDim(1);
    t.dtype = desc->GetDataType() == ge::DT_FLOAT16 ? 1U : 27U;
    t.fullVisible = *visible ? 1U : 0U;
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    ctx->SetBlockDim(t.usedCoreNum);
    ctx->SetTilingKey(0);
    workspace[0] = LimWorkspaceBytes(t, platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferShape(gert::InferShapeContext *ctx)
{
    auto *q = ctx->GetInputShape(1);
    auto *table = ctx->GetInputShape(3);
    auto *pool = ctx->GetInputShape(10);
    if (!q || !table || !pool) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 7; ++i)
        if (!ctx->GetOutputShape(i)) return ge::GRAPH_FAILED;
    for (size_t i : {0U, 1U}) {
        auto *s = ctx->GetOutputShape(i);
        s->SetDimNum(2);
        s->SetDim(0, q->GetDim(0));
        s->SetDim(1, 2048);
    }
    auto *s = ctx->GetOutputShape(2);
    s->SetDimNum(1);
    s->SetDim(0, q->GetDim(0));
    for (size_t i : {3U, 4U}) {
        auto *v = ctx->GetOutputShape(i);
        v->SetDimNum(2);
        v->SetDim(0, table->GetDim(0));
        v->SetDim(1, 32768);
    }
    s = ctx->GetOutputShape(5);
    s->SetDimNum(1);
    s->SetDim(0, table->GetDim(0));
    *ctx->GetOutputShape(6) = *pool;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDtype(gert::InferDataTypeContext *ctx)
{
    for (size_t i = 0; i < 7; ++i)
        ctx->SetOutputDataType(i, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
namespace ops {
class SglFusedLightningIndexerManage : public OpDef
{
public:
    explicit SglFusedLightningIndexerManage(const char *name) : OpDef(name)
    {
        for (auto *s : {"weights", "query", "key"})
            this->Input(s)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (auto *s : {"block_table", "seq_query", "seq_key", "offload", "cache_tokens", "request_state",
                        "pool_entries", "cache_slots"})
            this->Input(s)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (auto *s :
             {"topk_src", "topk_dst", "topk_counts", "miss_src", "miss_dst", "miss_counts", "cache_slots_out"})
            this->Output(s)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("full_visible").AttrType(REQUIRED).Bool();
        this->SetInferShape(optiling::InferShape).SetInferDataType(optiling::InferDtype);
        this->AICore().SetTiling(optiling::Tiling).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(SglFusedLightningIndexerManage);
}  // namespace ops

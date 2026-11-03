// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "quant_lightning_indexer_packed_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
#include <limits>

namespace {
ge::graphStatus Tiling(gert::TilingContext *context) {
    if (!context || !context->GetPlatformInfo() || !context->GetAttrs())
        return ge::GRAPH_FAILED;
    const ge::DataType types[] = {ge::DT_FLOAT8_E4M3FN, ge::DT_UINT8, ge::DT_BF16, ge::DT_FLOAT,
                                  ge::DT_INT32,         ge::DT_INT32, ge::DT_INT32};
    for (size_t i = 0; i < 7; ++i) {
        const auto *desc = context->GetInputDesc(i);
        if (!context->GetInputShape(i) || !desc || desc->GetDataType() != types[i])
            return ge::GRAPH_FAILED;
    }
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &k = context->GetInputShape(1)->GetStorageShape();
    const auto &w = context->GetInputShape(2)->GetStorageShape();
    const auto &qs = context->GetInputShape(3)->GetStorageShape();
    const auto &sq = context->GetInputShape(4)->GetStorageShape();
    const auto &sk = context->GetInputShape(5)->GetStorageShape();
    const auto &table = context->GetInputShape(6)->GetStorageShape();
    const auto *topk = context->GetAttrs()->GetInt(0);
    const auto *mode = context->GetAttrs()->GetInt(1);
    if (q.GetDimNum() != 3 || k.GetDimNum() != 2 || w.GetDimNum() != 2 || qs.GetDimNum() != 2 || sq.GetDimNum() != 1 ||
        sk.GetDimNum() != 1 || table.GetDimNum() != 2 || !topk || !mode)
        return ge::GRAPH_FAILED;
    const int64_t tokens = q.GetDim(0), heads = q.GetDim(1), batch = sq.GetDim(0);
    const int64_t page = k.GetDim(1) / QLIPacked::TOKEN_BYTES, maxPages = table.GetDim(1);
    constexpr int64_t maxDim = std::numeric_limits<uint32_t>::max();
    if (tokens < 0 || tokens > maxDim || q.GetDim(2) != QLIPacked::HEAD_DIM ||
        !(heads == 8 || heads == 16 || heads == 24 || heads == 32 || heads == 64) || k.GetDim(0) <= 0 ||
        k.GetDim(1) % QLIPacked::TOKEN_BYTES != 0 || page < 16 || page > 1024 || page % 16 != 0 || batch <= 0 || batch > maxDim ||
        sk.GetDim(0) != batch || table.GetDim(0) != batch || maxPages <= 0 || maxPages > maxDim / page ||
        w.GetDim(0) != tokens || qs.GetDim(0) != tokens || w.GetDim(1) != heads || qs.GetDim(1) != heads ||
        *topk <= 0 || *topk > 2048 || (*mode != 0 && *mode != 3))
        return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic(), aiv = platform.GetCoreNumAiv();
    if (!aic || !aiv)
        return ge::GRAPH_FAILED;
    const uint32_t blocks = platform.CalcTschBlockDim(aiv, aic, aiv);
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(QLITilingData) || !workspace)
        return ge::GRAPH_FAILED;
    auto *tiling = context->GetTilingData<QLITilingData>();
    if (!tiling)
        return ge::GRAPH_FAILED;
    *tiling = QLIPacked::MakeTilingData(
        static_cast<uint32_t>(batch), static_cast<uint32_t>(tokens), static_cast<uint32_t>(heads),
        static_cast<uint32_t>(page), static_cast<uint32_t>(maxPages), static_cast<uint32_t>(*topk),
        static_cast<uint32_t>(*mode), blocks);
    raw->SetDataSize(sizeof(QLITilingData));
    context->SetBlockDim(blocks);
    context->SetTilingKey(0);
    context->SetScheduleMode(1);
    const uint64_t maxKeys = maxPages * page;
    workspace[0] = platform.GetLibApiWorkSpaceSize() + 4 * ((maxKeys + 127) / 128) * 128 * sizeof(uint16_t) * aic;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *query = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    const auto *attrs = context->GetAttrs();
    if (!query || !output || !attrs || query->GetDimNum() != 3 || !attrs->GetInt(0))
        return ge::GRAPH_FAILED;
    output->SetDimNum(3);
    output->SetDim(0, query->GetDim(0));
    output->SetDim(1, 1);
    output->SetDim(2, *attrs->GetInt(0));
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace ops {
class SglQuantLightningIndexerPacked : public OpDef {
  public:
    explicit SglQuantLightningIndexerPacked(const char *name) : OpDef(name) {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN}).Format({ge::FORMAT_ND});
        this->Input("packed_cache").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Input("query_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("seq_q").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("seq_k").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("topk").AttrType(OPTIONAL).Int(2048);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        for (const auto *target : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(target, config);
    }
};
OP_ADD(SglQuantLightningIndexerPacked);
} // namespace ops

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace {
constexpr size_t kOutputSources[] = {0, 2, 3, 20};

ge::graphStatus InferShape(gert::InferShapeContext *ctx)
{
    for (size_t i = 0; i < 4; ++i) {
        const auto *input = ctx->GetRequiredInputShape(kOutputSources[i]);
        auto *output = ctx->GetOutputShape(i);
        if (!input || !output) return ge::GRAPH_FAILED;
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext *ctx)
{
    for (size_t i = 0; i < 4; ++i)
        ctx->SetOutputDataType(i, ctx->GetRequiredInputDataType(kOutputSources[i]));
    return ge::GRAPH_SUCCESS;
}
}  // namespace
namespace ops {
class SglFusedCopySfa : public OpDef
{
public:
    explicit SglFusedCopySfa(const char *name) : OpDef(name)
    {
        for (const auto *tensor : {"query", "query_rope", "hbm_ckv", "hbm_rope"}) {
            this->Input(tensor)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        for (const auto *tensor : {"q_ends", "kv_lens", "cache_tokens", "topk_src", "topk_dst", "topk_misses",
                                   "miss_src", "miss_dst", "misses", "hbm_table", "dram_table"}) {
            this->Input(tensor)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("current")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("positions")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("valid")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const auto *tensor : {"request_slots", "req_to_token"}) {
            this->Input(tensor)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("resident_end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const auto *tensor : {"attention_out", "hbm_ckv", "hbm_rope"}) {
            this->Output(tensor)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Output("resident_end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(REQUIRED).Float();
        this->Attr("dram_kv_dev_ptr").AttrType(REQUIRED).Int();
        this->Attr("dram_rope_dev_ptr").AttrType(REQUIRED).Int();
        this->Attr("full_visible").AttrType(REQUIRED).Bool();
        this->Attr("resident_refresh_tail").AttrType(REQUIRED).Bool();
        this->Attr("host_token_capacity").AttrType(REQUIRED).Int();
        this->SetInferShape(InferShape).SetInferDataType(InferType);
        for (auto config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglFusedCopySfa);
}  // namespace ops

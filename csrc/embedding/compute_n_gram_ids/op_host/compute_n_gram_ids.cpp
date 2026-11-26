#include "sgl_operator_config.h"
#include <algorithm>

#include "compute_n_gram_ids_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ComputeNGramIdsTilingData tiling;
    auto batch_size = *(context->GetAttrs()->GetAttrPointer<int64_t>(0));
    auto oe_n = *(context->GetAttrs()->GetAttrPointer<int64_t>(1));
    auto oe_k = *(context->GetAttrs()->GetAttrPointer<int64_t>(2));
    auto maxContextLen = *(context->GetAttrs()->GetAttrPointer<int64_t>(3));
    // const gert::Shape* tokens_shape = context->GetInputShape(3);
    // auto token_num = tokens_shape->GetDim(0);
    // TODO 仅通过batch_size划分对prefill不好，如果token_num长，使用别的切分方式
    uint32_t total_task = batch_size * (oe_n - 1) * oe_k;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    aivNum = std::min(aivNum, total_task);

    // printf("TilingFunc batch_size: %d, oe_n: %d, oe_k: %d, aivNum: %d, total_task: %d\n", batch_size, oe_n, oe_k,
    // aivNum, total_task);

    context->SetBlockDim(aivNum);
    tiling.set_totalTask(total_task);
    tiling.set_batchSize(batch_size);
    tiling.set_coreNum(aivNum);
    tiling.set_oeN(oe_n);
    tiling.set_oeK(oe_k);
    tiling.set_maxContextLen(maxContextLen);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(1);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *tokens_shape = context->GetInputShape(3);
    auto token_num = tokens_shape->GetDim(0);
    auto oe_n = *(context->GetAttrs()->GetAttrPointer<int64_t>(1));
    auto oe_k = *(context->GetAttrs()->GetAttrPointer<int64_t>(2));
    gert::Shape *outShape = context->GetOutputShape(0);
    *outShape = gert::Shape({token_num, (oe_n - 1) * oe_k});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(3);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SglComputeNGramIds : public OpDef
{
public:
    explicit SglComputeNGramIds(const char *name) : OpDef(name)
    {
        this->Input("oe_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("oe_mods")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("exclusive_oe_embeder_size_sums")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("exclusive_req_len_sums")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("oe_token_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("row_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("column_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("oe_n_gram_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batch_size").Int();
        this->Attr("oe_n").Int();
        this->Attr("oe_k").Int();
        this->Attr("max_context_len").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglComputeNGramIds);
}  // namespace ops

#include "../op_kernel/alloc_extend_tiling_data.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingAllocExtend(gert::TilingContext *context)
{
    const auto *prefix = context->GetInputShape(0);
    const auto *output = context->GetInputShape(4);
    const auto *attrs = context->GetAttrs();
    const auto *pageSize = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!prefix || !output || !pageSize || !context->GetPlatformInfo() || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(optiling::AllocExtendTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    if (prefix->GetOriginShape().GetDimNum() < 1 || output->GetOriginShape().GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<optiling::AllocExtendTilingData>();
    if (!data) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    data->batch_size = static_cast<int32_t>(prefix->GetOriginShape().GetDim(0));
    data->page_size = static_cast<int32_t>(*pageSize);
    data->used_core_num = std::min(static_cast<int32_t>(platform.GetCoreNumAiv()), data->batch_size);
    data->total_extend_tokens = output->GetOriginShape().GetDim(0);
    if (data->used_core_num <= 0) return ge::GRAPH_FAILED;
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    raw->SetDataSize(sizeof(*data));
    context->SetBlockDim(data->used_core_num);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeAllocExtend(gert::InferShapeContext *context)
{
    for (int i = 0; i < 2; ++i) {
        const auto *input = context->GetInputShape(4 + i);
        auto *output = context->GetOutputShape(i);
        if (!input || !output) return ge::GRAPH_FAILED;
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAllocExtend(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(4));
    context->SetOutputDataType(1, context->GetInputDataType(5));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class SglAllocExtend : public OpDef
{
public:
    explicit SglAllocExtend(const char *name) : OpDef(name)
    {
        this->Input("pre_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("last_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("free_pages")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("page_size").AttrType(REQUIRED).Int();
        this->SetInferShape(optiling::InferShapeAllocExtend).SetInferDataType(optiling::InferDataTypeAllocExtend);
        this->AICore().SetTiling(optiling::TilingAllocExtend);
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglAllocExtend);
}  // namespace ops

#include "../op_kernel/assign_cache_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingAssignCacheOp(gert::TilingContext *context)
{
    auto *shape = context->GetInputShape(0);
    auto *desc = context->GetInputDesc(0);
    auto *sync = context->GetInputShape(6);
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!shape || !desc || !sync || !raw || !raw->GetData() || !workspace || !context->GetPlatformInfo() ||
        raw->GetCapacity() < sizeof(optiling::CustomAssignTilingData))
        return ge::GRAPH_FAILED;
    if (shape->GetOriginShape().GetDimNum() < 2) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    auto *data = context->GetTilingData<optiling::CustomAssignTilingData>();
    if (!data || platform.GetCoreNumAiv() == 0) return ge::GRAPH_FAILED;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    data->batchSize = shape->GetOriginShape().GetDim(0);
    data->tokenPoolLength = shape->GetOriginShape().GetDim(1);
    data->syncWorkspaceSize = sync->GetOriginShape().GetShapeSize();
    data->ubSize = ubSize;
    switch (desc->GetDataType()) {
        case ge::DT_INT8:
            data->typeBytes = 1;
            break;
        case ge::DT_INT16:
            data->typeBytes = 2;
            break;
        case ge::DT_INT32:
            data->typeBytes = 4;
            break;
        case ge::DT_INT64:
            data->typeBytes = 8;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(platform.GetCoreNumAiv());
    context->SetTilingKey(0);
    raw->SetDataSize(sizeof(*data));
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferShapeAssignCacheOp(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeAssignCacheOp(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
namespace ops {
class SglAssignCache : public OpDef
{
public:
    explicit SglAssignCache(const char *name) : OpDef(name)
    {
        this->Input("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dst_start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dst_end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src_start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src_end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sync")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(optiling::InferShapeAssignCacheOp).SetInferDataType(optiling::InferDataTypeAssignCacheOp);
        this->AICore().SetTiling(optiling::TilingAssignCacheOp);
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglAssignCache);
}  // namespace ops

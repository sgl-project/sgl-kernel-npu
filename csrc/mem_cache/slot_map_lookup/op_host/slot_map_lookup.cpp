#include "../op_kernel/slot_map_lookup_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingSlotMapLookup(gert::TilingContext *context)
{
    const auto *map = context->GetInputShape(0);
    const auto *topk = context->GetInputShape(2);
    const auto *attrs = context->GetAttrs();
    const auto *blockDim = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!map || !topk || !blockDim || !context->GetPlatformInfo() || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(SlotMapLookupTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<SlotMapLookupTilingData>();
    if (!data) {
        return ge::GRAPH_FAILED;
    }
    const auto &mapShape = map->GetStorageShape();
    const auto &topkShape = topk->GetStorageShape();
    if (mapShape.GetDimNum() != 2 || topkShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const auto cores = platform.GetCoreNumAiv();
    const auto blocks = *blockDim == 0 ? static_cast<int64_t>(cores) : *blockDim;
    if (blocks <= 0 || blocks > cores) {
        return ge::GRAPH_FAILED;
    }
    data->size = static_cast<uint32_t>(mapShape.GetDim(0));
    data->maxContextLen = static_cast<uint32_t>(mapShape.GetDim(1));
    data->batchSize = static_cast<uint32_t>(topkShape.GetDim(0));
    data->topk = static_cast<uint32_t>(topkShape.GetDim(1));
    raw->SetDataSize(sizeof(*data));
    context->SetBlockDim(static_cast<uint32_t>(blocks));
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeSlotMapLookup(gert::InferShapeContext *context)
{
    const auto *topk = context->GetInputShape(2);
    auto *hits = context->GetOutputShape(0);
    auto *positions = context->GetOutputShape(1);
    if (!topk || !hits || !positions) {
        return ge::GRAPH_FAILED;
    }
    *hits = *topk;
    *positions = *topk;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSlotMapLookup(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

class SglSlotMapLookup : public OpDef
{
public:
    explicit SglSlotMapLookup(const char *name) : OpDef(name)
    {
        for (const auto *input : {"slot_map", "req_indices", "topk_indices"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const auto *output : {"token_on_device", "device_token_pos"}) {
            this->Output(output)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Attr("block_dim").AttrType(REQUIRED).Int();
        this->SetInferShape(optiling::InferShapeSlotMapLookup).SetInferDataType(optiling::InferDataTypeSlotMapLookup);
        this->AICore().SetTiling(optiling::TilingSlotMapLookup);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglSlotMapLookup);

}  // namespace ops

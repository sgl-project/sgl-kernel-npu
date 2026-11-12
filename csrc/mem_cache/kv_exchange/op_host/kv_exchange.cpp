// BSD 3-Clause License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

#include "../op_kernel/kv_exchange_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingKvExchange(gert::TilingContext *context)
{
    const auto *device = context->GetInputShape(0);
    const auto *host = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *transfer = attrs ? attrs->GetListInt(0) : nullptr;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!device || !host || !transfer || transfer->GetSize() != KV_EXCHANGE_ATTR_SIZE || !context->GetPlatformInfo() ||
        !raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglKvExchangeTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }

    const auto *fields = transfer->GetData();
    const auto &deviceShape = device->GetStorageShape();
    const auto &hostShape = host->GetStorageShape();
    if (deviceShape.GetDimNum() != 1 || hostShape.GetDimNum() != 1 || deviceShape.GetDim(0) <= 0 ||
        deviceShape.GetDim(0) != hostShape.GetDim(0) || fields[PAGE_SIZE] <= 0 ||
        deviceShape.GetDim(0) % fields[PAGE_SIZE] != 0 || fields[NUM_COMPONENTS] < 1 ||
        fields[NUM_COMPONENTS] > KV_EXCHANGE_MAX_COMPONENTS || (fields[DIRECTION] != 1 && fields[DIRECTION] != 2) ||
        (fields[HOST_LAYOUT] != 0 && fields[HOST_LAYOUT] != 1)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t cores = platform.GetCoreNumAiv();
    const int64_t blockDim = fields[BLOCK_DIM] == 0 ? cores : fields[BLOCK_DIM];
    if (blockDim <= 0 || blockDim > cores) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<SglKvExchangeTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    tiling->numComponents = fields[NUM_COMPONENTS];
    tiling->numPages = deviceShape.GetDim(0) / fields[PAGE_SIZE];
    tiling->pageSize = fields[PAGE_SIZE];
    tiling->direction = fields[DIRECTION];
    tiling->hostLayout = fields[HOST_LAYOUT];
    tiling->startDim = fields[START_DIM];
    for (int32_t component = 0; component < KV_EXCHANGE_MAX_COMPONENTS; ++component) {
        for (int32_t field = 0; field < KV_EXCHANGE_META_STRIDE; ++field) {
            tiling->components[component][field] =
                fields[ATTR_HEADER_SIZE + component * KV_EXCHANGE_META_STRIDE + field];
        }
    }
    raw->SetDataSize(sizeof(SglKvExchangeTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) {
        return ge::GRAPH_FAILED;
    }
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

class SglKvExchange : public OpDef
{
public:
    explicit SglKvExchange(const char *name) : OpDef(name)
    {
        this->Input("device_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("host_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Reuse the index descriptor as a Ref output. The kernel only writes
        // the registered destinations described by the transfer attribute.
        this->Output("device_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("transfer").AttrType(REQUIRED).ListInt();
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingKvExchange);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglKvExchange);
}  // namespace ops

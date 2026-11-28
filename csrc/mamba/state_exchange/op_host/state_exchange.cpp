// BSD 3-Clause License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

#include "../op_kernel/state_exchange_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingStateExchange(gert::TilingContext *context)
{
    const auto *device = context->GetInputShape(0);
    const auto *host = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *transfer = attrs ? attrs->GetListInt(0) : nullptr;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!device || !host || !transfer || transfer->GetSize() != STATE_EXCHANGE_ATTR_SIZE ||
        !context->GetPlatformInfo() || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(SglStateExchangeTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }

    const auto *fields = transfer->GetData();
    const auto &deviceShape = device->GetStorageShape();
    const auto &hostShape = host->GetStorageShape();
    if (deviceShape.GetDimNum() != 1 || hostShape.GetDimNum() != 1 || deviceShape.GetDim(0) <= 0 ||
        deviceShape.GetDim(0) != hostShape.GetDim(0) || fields[NUM_ENTRIES] < 1 ||
        fields[NUM_ENTRIES] > STATE_EXCHANGE_MAX_ENTRIES || (fields[DIRECTION] != 1 && fields[DIRECTION] != 2)) {
        return ge::GRAPH_FAILED;
    }
    // Per entry: [0]=device_base [1]=host_dva [2]=device_slot_pitch
    // [3]=device_layer_pitch [4]=host_slot_pitch [5]=host_layer_pitch
    // [6]=width [7]=layer_count. Pitches must be at least the row width so
    // adjacent slots/layers never overlap.
    for (int64_t entry = 0; entry < fields[NUM_ENTRIES]; ++entry) {
        const int64_t *meta = fields + ATTR_HEADER_SIZE + entry * STATE_EXCHANGE_META_STRIDE;
        if (meta[6] <= 0 || meta[7] < 1 || meta[2] < meta[6] || meta[3] < meta[6] || meta[4] < meta[6] ||
            meta[5] < meta[6]) {
            return ge::GRAPH_FAILED;
        }
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t cores = platform.GetCoreNumAiv();
    const int64_t blockDim = fields[BLOCK_DIM] == 0 ? cores : fields[BLOCK_DIM];
    if (blockDim <= 0 || blockDim > cores) {
        return ge::GRAPH_FAILED;
    }
    // -1 selects the native acc_offload occupancy; an explicit START_DIM must
    // stay inside the launched core window, otherwise every core takes the
    // skip branch and the transfer silently drops.
    const int64_t startDim = fields[START_DIM];
    if (startDim < -1 || startDim >= blockDim) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<SglStateExchangeTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    tiling->numEntries = fields[NUM_ENTRIES];
    tiling->indicesLen = deviceShape.GetDim(0);
    tiling->direction = fields[DIRECTION];
    tiling->startDim = fields[START_DIM];
    for (int32_t entry = 0; entry < STATE_EXCHANGE_MAX_ENTRIES; ++entry) {
        for (int32_t field = 0; field < STATE_EXCHANGE_META_STRIDE; ++field) {
            tiling->entries[entry][field] = fields[ATTR_HEADER_SIZE + entry * STATE_EXCHANGE_META_STRIDE + field];
        }
    }
    raw->SetDataSize(sizeof(SglStateExchangeTilingData));
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

class SglStateExchange : public OpDef
{
public:
    explicit SglStateExchange(const char *name) : OpDef(name)
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
        this->AICore().SetTiling(optiling::TilingStateExchange);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglStateExchange);
}  // namespace ops

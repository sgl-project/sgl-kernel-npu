#include "../op_kernel/unidex_copy_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace optiling {

static ge::graphStatus TilingUnidexCopy(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *transfer = attrs ? attrs->GetListInt(0) : nullptr;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!transfer || transfer->GetSize() != UNIDEX_COPY_ATTR_COUNT || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(UnidexCopyTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }

    auto *data = context->GetTilingData<UnidexCopyTilingData>();
    if (!data) {
        return ge::GRAPH_FAILED;
    }
    const auto *fields = transfer->GetData();
    data->srcAddr = static_cast<uint64_t>(fields[SRC_ADDR]);
    data->dstAddr = static_cast<uint64_t>(fields[DST_ADDR]);
    data->srcRows = static_cast<uint32_t>(fields[SRC_ROWS]);
    data->dstRows = static_cast<uint32_t>(fields[DST_ROWS]);
    data->blockBytes = static_cast<uint32_t>(fields[BLOCK_BYTES]);
    data->maxCopy = static_cast<uint32_t>(fields[MAX_COPY]);
    raw->SetDataSize(sizeof(*data));
    context->SetBlockDim(static_cast<uint32_t>(fields[BLOCK_DIM]));
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeUnidexCopy(gert::InferShapeContext *context)
{
    const auto *indices = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    if (!indices || !out) {
        return ge::GRAPH_FAILED;
    }
    *out = *indices;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeUnidexCopy(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

class SglUnidexCopy : public OpDef
{
public:
    explicit SglUnidexCopy(const char *name) : OpDef(name)
    {
        for (const auto *index : {"src_index", "dst_index"}) {
            this->Input(index)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Input("valid_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // The payload DVA can address registered host memory; the kernel writes through it.
        this->Output("dst_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("transfer").AttrType(REQUIRED).ListInt();
        this->SetInferShape(optiling::InferShapeUnidexCopy).SetInferDataType(optiling::InferDataTypeUnidexCopy);
        this->AICore().SetTiling(optiling::TilingUnidexCopy);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglUnidexCopy);

}  // namespace ops

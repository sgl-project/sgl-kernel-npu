#include "../op_kernel/dcp_sparse_index_remap_tiling.h"

#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

static ge::graphStatus Tiling(gert::TilingContext *context)
{
    constexpr int64_t kMaxBlocks = 32;
    const auto *attrs = context->GetAttrs();
    if (!attrs || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto *attr0 = attrs->GetAttrPointer<int64_t>(0);
    const auto *attr1 = attrs->GetAttrPointer<int64_t>(1);
    const auto *attr2 = attrs->GetAttrPointer<int64_t>(2);
    if (!attr0 || !attr1 || !attr2) {
        return ge::GRAPH_FAILED;
    }

    const auto *indices = context->GetInputShape(0);
    if (!indices || indices->GetStorageShape().GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = indices->GetStorageShape();
    const int64_t rows = shape.GetDim(0);
    const int64_t dcp_size = *attr0;
    const int64_t rank = *attr1;
    const int64_t page_size = *attr2;
    constexpr int64_t kTopK = 2048;
    if (rows <= 0 || rows > std::numeric_limits<uint32_t>::max() / kTopK || shape.GetDim(1) != kTopK ||
        (dcp_size != 2 && dcp_size != 4) || rank < 0 || rank >= dcp_size || page_size <= 0 || page_size > 65536 ||
        (page_size & (page_size - 1)) != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t jobs = rows;

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t cores = platform.GetCoreNumAiv();
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (cores <= 0 || !raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglDcpSparseIndexRemapTilingData) ||
        !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<SglDcpSparseIndexRemapTilingData>();
    if (!data) {
        return ge::GRAPH_FAILED;
    }
    data->rows = static_cast<uint32_t>(rows);
    data->page_shift = __builtin_ctzll(static_cast<uint64_t>(page_size));
    data->dcp_shift = dcp_size == 2 ? 1 : 2;
    data->dcp_rank = static_cast<int32_t>(rank);
    raw->SetDataSize(sizeof(SglDcpSparseIndexRemapTilingData));
    context->SetBlockDim(std::min({cores, jobs, kMaxBlocks}));
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *indices = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (!indices || !output) {
        return ge::GRAPH_FAILED;
    }
    *output = *indices;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class SglDcpSparseIndexRemap : public OpDef
{
public:
    explicit SglDcpSparseIndexRemap(const char *name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dcp_size").AttrType(REQUIRED).Int();
        this->Attr("dcp_rank").AttrType(REQUIRED).Int();
        this->Attr("page_size").AttrType(REQUIRED).Int();
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::Tiling);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglDcpSparseIndexRemap);

}  // namespace ops

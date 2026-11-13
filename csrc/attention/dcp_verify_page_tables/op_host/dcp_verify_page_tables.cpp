#include "../op_kernel/dcp_verify_page_tables_tiling.h"

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
    const auto *attr3 = attrs->GetAttrPointer<int64_t>(3);
    if (!attr0 || !attr1 || !attr2 || !attr3) {
        return ge::GRAPH_FAILED;
    }

    const auto *slots = context->GetInputShape(0);
    const auto *requests = context->GetInputShape(1);
    const auto *ordinary = context->GetInputShape(2);
    const auto *local = context->GetInputShape(3);
    const auto *request_desc = context->GetInputDesc(1);
    if (!slots || !requests || !ordinary || !local || !request_desc) {
        return ge::GRAPH_FAILED;
    }
    const auto &slot_shape = slots->GetStorageShape();
    const auto &request_shape = requests->GetStorageShape();
    const auto &ordinary_shape = ordinary->GetStorageShape();
    const auto &local_shape = local->GetStorageShape();
    if (slot_shape.GetDimNum() != 2 || request_shape.GetDimNum() != 1 || ordinary_shape.GetDimNum() != 2 ||
        local_shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t active_cols = *attr0;
    const int64_t query_len = *attr1;
    const int64_t page_size = *attr2;
    const int64_t dcp_size = *attr3;
    const int64_t rows = request_shape.GetDim(0);
    const int64_t slot_rows = slot_shape.GetDim(0);
    const int64_t slot_cols = slot_shape.GetDim(1);
    const int64_t ordinary_cols = ordinary_shape.GetDim(1);
    const int64_t local_cols = local_shape.GetDim(1);
    constexpr int64_t kMaxIndex = std::numeric_limits<uint32_t>::max();
    constexpr int64_t kTile = 128;
    if (rows <= 0 || rows > kMaxIndex || slot_rows < 0 || slot_rows > kMaxIndex || slot_cols < 0 ||
        slot_cols > kMaxIndex || query_len <= 0 || query_len > kMaxIndex / rows || (dcp_size != 2 && dcp_size != 4) ||
        page_size <= 0 || page_size > 65536 || (page_size & (page_size - 1)) != 0 || ordinary_shape.GetDim(0) != rows ||
        local_shape.GetDim(0) != rows * query_len || ordinary_cols < 0 || ordinary_cols > kMaxIndex / rows ||
        local_cols < 0 || local_cols > kMaxIndex / (rows * query_len) || active_cols < 0 ||
        active_cols > (slot_cols + page_size - 1) / page_size || active_cols > ordinary_cols ||
        (active_cols + dcp_size - 1) / dcp_size > local_cols) {
        return ge::GRAPH_FAILED;
    }
    const int64_t cols = std::max(ordinary_cols, local_cols * dcp_size);
    const int64_t tiles = (cols + kTile - 1) / kTile;
    if (tiles == 0 || cols > kMaxIndex - kTile || rows > (kMaxIndex - kMaxBlocks) / tiles) {
        return ge::GRAPH_FAILED;
    }
    const int64_t jobs = rows * tiles;
    const auto request_type = request_desc->GetDataType();
    if (request_type != ge::DT_INT32 && request_type != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t cores = platform.GetCoreNumAiv();
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (cores <= 0 || !raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglDcpVerifyPageTablesTilingData) ||
        !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<SglDcpVerifyPageTablesTilingData>();
    if (!data) {
        return ge::GRAPH_FAILED;
    }
    data->rows = static_cast<uint32_t>(rows);
    data->slot_rows = static_cast<uint32_t>(slot_rows);
    data->slot_cols = static_cast<uint32_t>(slot_cols);
    data->ordinary_cols = static_cast<uint32_t>(ordinary_cols);
    data->local_cols = static_cast<uint32_t>(local_cols);
    data->active_cols = static_cast<uint32_t>(active_cols);
    data->query_len = static_cast<uint32_t>(query_len);
    data->page_size = static_cast<uint32_t>(page_size);
    data->dcp_size = static_cast<uint32_t>(dcp_size);
    data->index_bytes = request_type == ge::DT_INT64 ? sizeof(int64_t) : sizeof(int32_t);
    raw->SetDataSize(sizeof(SglDcpVerifyPageTablesTilingData));
    context->SetBlockDim(std::min({cores, jobs, kMaxBlocks}));
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // The output ports are references to the preallocated input tables.
    for (size_t i = 0; i < 2; ++i) {
        const auto *input = context->GetInputShape(i + 2);
        auto *output = context->GetOutputShape(i);
        if (!input || !output) {
            return ge::GRAPH_FAILED;
        }
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class SglDcpVerifyPageTables : public OpDef
{
public:
    explicit SglDcpVerifyPageTables(const char *name) : OpDef(name)
    {
        this->Input("slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("requests")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ordinary")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("local")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // Matching input and output names generate ACLNN reference parameters.
        this->Output("ordinary")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("local")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("active_cols").AttrType(REQUIRED).Int();
        this->Attr("query_len").AttrType(REQUIRED).Int();
        this->Attr("page_size").AttrType(REQUIRED).Int();
        this->Attr("dcp_size").AttrType(REQUIRED).Int();
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::Tiling);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglDcpVerifyPageTables);

}  // namespace ops

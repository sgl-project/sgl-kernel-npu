/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "../op_kernel/situ_mxfp8_quant_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

static ge::graphStatus TilingSglSituMxfp8Quant(gert::TilingContext *context)
{
    const auto *xShapePtr = context->GetInputShape(0);
    const auto *groupShapePtr = context->GetInputShape(1);
    const auto *xDesc = context->GetInputDesc(0);
    const auto *groupDesc = context->GetInputDesc(1);
    const auto *attrs = context->GetAttrs();
    if (!xShapePtr || !groupShapePtr || !xDesc || !groupDesc || !attrs || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &xShape = xShapePtr->GetStorageShape();
    const auto &groupShape = groupShapePtr->GetStorageShape();
    if (xShape.GetDimNum() != 2 || groupShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = xShape.GetDim(0);
    const int64_t cols = xShape.GetDim(1);
    const int64_t numExperts = groupShape.GetDim(0);
    if (rows <= 0 || cols != static_cast<int64_t>(SITU_INPUT_COLS) || numExperts <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    if (groupDesc->GetDataType() != ge::DT_INT32 && groupDesc->GetDataType() != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }
    const auto *groupTypeAttr = attrs->GetAttrPointer<int64_t>(0);
    const auto *betaAttr = attrs->GetAttrPointer<float>(1);
    const auto *linearBetaAttr = attrs->GetAttrPointer<float>(2);
    if (!groupTypeAttr || !betaAttr || !linearBetaAttr) {
        return ge::GRAPH_FAILED;
    }
    if ((*groupTypeAttr != 0 && *groupTypeAttr != 1) || !(*betaAttr > 0.0f) || !(*linearBetaAttr > 0.0f)) {
        return ge::GRAPH_FAILED;
    }
    constexpr uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    if (static_cast<uint64_t>(rows) > maxUint32 || static_cast<uint64_t>(numExperts) > maxUint32) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    uint32_t blockDim = coreNum > 0 ? static_cast<uint32_t>(coreNum) : 1U;
    blockDim = std::min<uint32_t>(blockDim, static_cast<uint32_t>(rows));

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglSituMxfp8QuantTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SglSituMxfp8QuantTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    tiling->capacityRows = static_cast<uint32_t>(rows);
    tiling->numExperts = static_cast<uint32_t>(numExperts);
    tiling->groupListType = static_cast<uint32_t>(*groupTypeAttr);
    tiling->groupDtype = groupDesc->GetDataType() == ge::DT_INT64 ? 1U : 0U;
    tiling->beta = *betaAttr;
    tiling->linearBeta = *linearBetaAttr;

    raw->SetDataSize(sizeof(SglSituMxfp8QuantTilingData));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

// payload is [rows, 3072]; scales keeps the vendor MX layout [rows, 48, 2],
// which is the same 96 contiguous E8M0 bytes consumed by the downstream GMM.
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *xShapePtr = context->GetInputShape(0);
    auto *payloadShape = context->GetOutputShape(0);
    auto *scalesShape = context->GetOutputShape(1);
    if (!xShapePtr || !payloadShape || !scalesShape) {
        return ge::GRAPH_FAILED;
    }
    // InferShapeContext already returns gert::Shape, which owns the dimensional
    // queries. GetStorageShape() only exists on gert::StorageShape from a tiling
    // context.
    if (xShapePtr->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = xShapePtr->GetDim(0);
    payloadShape->SetDimNum(2);
    payloadShape->SetDim(0, rows);
    payloadShape->SetDim(1, static_cast<int64_t>(optiling::SITU_OUTPUT_COLS));
    scalesShape->SetDimNum(3);
    scalesShape->SetDim(0, rows);
    scalesShape->SetDim(1, static_cast<int64_t>(optiling::SITU_SCALE_COLS / 2));
    scalesShape->SetDim(2, 2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

class SglSituMxfp8Quant : public OpDef
{
public:
    explicit SglSituMxfp8Quant(const char *name) : OpDef(name)
    {
        // Two dtype columns: the only varying input is the group_list index type.
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("payload")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(1);
        this->Attr("beta").AttrType(OPTIONAL).Float(4.0f);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(25.0f);
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingSglSituMxfp8Quant);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglSituMxfp8Quant);

}  // namespace ops

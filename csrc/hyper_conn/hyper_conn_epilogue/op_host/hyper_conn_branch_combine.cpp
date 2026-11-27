// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../op_kernel/hyper_conn_tiling_data.h"
#include "hyper_conn_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace optiling {
namespace {

using sglang::npu_kernel::hyper_conn::EpilogueKernel;
using sglang::npu_kernel::hyper_conn::LaunchBlocks;
using sglang::npu_kernel::hyper_conn::MAX_BRANCHES;
using sglang::npu_kernel::hyper_conn::MAX_WIDTH;
using sglang::npu_kernel::hyper_conn::SelectColumnTile;
using sglang::npu_kernel::hyper_conn::TunedBlocks;

ge::graphStatus TilingHyperConnBranchCombine(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto *blockShape = context->GetInputShape(0);
    const auto *residualShape = context->GetInputShape(1);
    const auto *logitsShape = context->GetInputShape(2);
    const auto *blockDesc = context->GetInputDesc(0);
    const auto *residualDesc = context->GetInputDesc(1);
    const auto *logitsDesc = context->GetInputDesc(2);
    if (!blockShape || !residualShape || !logitsShape || !blockDesc || !residualDesc || !logitsDesc) {
        return ge::GRAPH_FAILED;
    }
    if (blockDesc->GetDataType() != ge::DT_BF16 || residualDesc->GetDataType() != ge::DT_BF16 ||
        logitsDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const auto &b = blockShape->GetStorageShape();
    const auto &r = residualShape->GetStorageShape();
    const auto &l = logitsShape->GetStorageShape();
    if (b.GetDimNum() != 2 || r.GetDimNum() != 2 || l.GetDimNum() != 1 || b.GetDim(0) != r.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const auto *branchesAttr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *logitsStrideAttr = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    if (!branchesAttr || !logitsStrideAttr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = r.GetDim(0);
    const int64_t branches = *branchesAttr;
    const int64_t logitsStride = *logitsStrideAttr;
    if (rows <= 0 || rows > std::numeric_limits<uint32_t>::max() || branches <= 0 ||
        branches > static_cast<int64_t>(MAX_BRANCHES) || (branches & (branches - 1)) != 0 ||
        r.GetDim(1) % branches != 0 || logitsStride < branches || logitsStride > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const int64_t width = r.GetDim(1) / branches;
    if (width <= 0 || width > static_cast<int64_t>(MAX_WIDTH) || width % 16 != 0 || b.GetDim(1) != width) {
        return ge::GRAPH_FAILED;
    }
    const int64_t requiredLogitsSpan = (rows - 1) * logitsStride + branches;
    if (l.GetDim(0) < requiredLogitsSpan) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t tile =
        SelectColumnTile(static_cast<uint32_t>(width), static_cast<uint32_t>(branches), static_cast<uint32_t>(rows));
    if (tile == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tasks = static_cast<uint64_t>(rows) * (static_cast<uint64_t>(width) / tile);
    if (tasks > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        return ge::GRAPH_FAILED;
    }
    const int64_t aivNum = platform.GetCoreNumAiv();
    if (aivNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t blockDim = LaunchBlocks(
        TunedBlocks(EpilogueKernel::BRANCH_COMBINE, static_cast<uint32_t>(width), static_cast<uint32_t>(rows)), tasks,
        aivNum);

    SglHyperConnBranchCombineTilingData td{};
    td.rows = static_cast<uint32_t>(rows);
    td.hc = static_cast<uint32_t>(branches);
    td.width = static_cast<uint32_t>(width);
    td.tile = tile;
    td.logitsStride = static_cast<uint32_t>(logitsStride);
    td.invHc = 1.0f / static_cast<float>(branches);

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(td) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &td, sizeof(td));
    raw->SetDataSize(sizeof(td));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace
}  // namespace optiling

namespace ops {
namespace {

ge::graphStatus InferShapeHyperConnBranchCombine(gert::InferShapeContext *context)
{
    const auto *residual = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    if (!residual || !out) {
        return ge::GRAPH_FAILED;
    }
    *out = *residual;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeHyperConnBranchCombine(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

class SglHyperConnBranchCombine : public OpDef
{
public:
    explicit SglHyperConnBranchCombine(const char *name) : OpDef(name)
    {
        this->Input("block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("branches").AttrType(OPTIONAL).Int(4);
        this->Attr("logits_stride").AttrType(OPTIONAL).Int(4);
        this->SetInferShape(InferShapeHyperConnBranchCombine).SetInferDataType(InferDataTypeHyperConnBranchCombine);
        this->AICore().SetTiling(optiling::TilingHyperConnBranchCombine);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglHyperConnBranchCombine);

}  // namespace ops

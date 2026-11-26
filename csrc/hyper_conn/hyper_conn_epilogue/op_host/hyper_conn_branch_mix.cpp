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

ge::graphStatus TilingHyperConnBranchMix(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto *normedShape = context->GetInputShape(0);
    const auto *logitsShape = context->GetInputShape(1);
    const auto *normedDesc = context->GetInputDesc(0);
    const auto *logitsDesc = context->GetInputDesc(1);
    if (!normedShape || !logitsShape || !normedDesc || !logitsDesc) {
        return ge::GRAPH_FAILED;
    }
    if (normedDesc->GetDataType() != ge::DT_BF16 || logitsDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const auto &n = normedShape->GetStorageShape();
    const auto &l = logitsShape->GetStorageShape();
    if (n.GetDimNum() != 2 || l.GetDimNum() != 2 || n.GetDim(0) != l.GetDim(0) || n.GetDim(1) != l.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const auto *branchesAttr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    if (!branchesAttr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = n.GetDim(0);
    const int64_t branches = *branchesAttr;
    if (rows <= 0 || rows > std::numeric_limits<uint32_t>::max() || branches <= 0 ||
        branches > static_cast<int64_t>(MAX_BRANCHES) || (branches & (branches - 1)) != 0 ||
        n.GetDim(1) % branches != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t width = n.GetDim(1) / branches;
    if (width <= 0 || width > static_cast<int64_t>(MAX_WIDTH) || width % 16 != 0) {
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
    const uint32_t blockDim =
        LaunchBlocks(TunedBlocks(EpilogueKernel::BRANCH_MIX, static_cast<uint32_t>(width), static_cast<uint32_t>(rows)),
                     tasks, aivNum);

    SglHyperConnBranchMixTilingData td{};
    td.rows = static_cast<uint32_t>(rows);
    td.hc = static_cast<uint32_t>(branches);
    td.width = static_cast<uint32_t>(width);
    td.tile = tile;
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

ge::graphStatus InferShapeHyperConnBranchMix(gert::InferShapeContext *context)
{
    const auto *normed = context->GetInputShape(0);
    const auto *attrs = context->GetAttrs();
    const auto *branchesAttr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    auto *mixed = context->GetOutputShape(0);
    if (!normed || !branchesAttr || !mixed || normed->GetDimNum() != 2 || *branchesAttr <= 0) {
        return ge::GRAPH_FAILED;
    }
    mixed->SetDimNum(2);
    mixed->SetDim(0, normed->GetDim(0));
    mixed->SetDim(1, normed->GetDim(1) / *branchesAttr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeHyperConnBranchMix(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

class SglHyperConnBranchMix : public OpDef
{
public:
    explicit SglHyperConnBranchMix(const char *name) : OpDef(name)
    {
        this->Input("normed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mixed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("branches").AttrType(OPTIONAL).Int(4);
        this->SetInferShape(InferShapeHyperConnBranchMix).SetInferDataType(InferDataTypeHyperConnBranchMix);
        this->AICore().SetTiling(optiling::TilingHyperConnBranchMix);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglHyperConnBranchMix);

}  // namespace ops

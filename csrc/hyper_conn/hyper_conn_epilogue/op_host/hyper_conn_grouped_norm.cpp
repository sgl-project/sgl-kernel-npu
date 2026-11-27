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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace optiling {
namespace {

using sglang::npu_kernel::hyper_conn::EpilogueKernel;
using sglang::npu_kernel::hyper_conn::LaunchBlocks;
using sglang::npu_kernel::hyper_conn::MAX_WIDTH;
using sglang::npu_kernel::hyper_conn::TunedBlocks;

ge::graphStatus TilingHyperConnGroupedNorm(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto *srcShape = context->GetInputShape(0);
    const auto *weightShape = context->GetInputShape(1);
    const auto *srcDesc = context->GetInputDesc(0);
    const auto *weightDesc = context->GetInputDesc(1);
    if (!srcShape || !weightShape || !srcDesc || !weightDesc) {
        return ge::GRAPH_FAILED;
    }
    if (srcDesc->GetDataType() != ge::DT_BF16 || weightDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const auto &x = srcShape->GetStorageShape();
    const auto &w = weightShape->GetStorageShape();
    if (x.GetDimNum() < 2 || w.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const auto *groupSizeAttr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *epsAttr = attrs ? attrs->GetAttrPointer<float>(1) : nullptr;
    if (!groupSizeAttr || !epsAttr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t lastDim = x.GetDim(x.GetDimNum() - 1);
    const int64_t width = (*groupSizeAttr > 0) ? *groupSizeAttr : lastDim;
    const int64_t weightLen = w.GetDim(0);
    const float eps = *epsAttr;
    if (width <= 0 || width > static_cast<int64_t>(MAX_WIDTH) || width % 8 != 0 || lastDim <= 0 ||
        lastDim % width != 0 || weightLen <= 0 || weightLen % width != 0 ||
        weightLen > std::numeric_limits<uint32_t>::max() || !std::isfinite(eps) || eps <= 0.0f) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numel = x.GetShapeSize();
    if (numel <= 0 || numel % width != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t groups = numel / width;
    if (groups <= 0 || groups > std::numeric_limits<uint32_t>::max() || groups % (weightLen / width) != 0) {
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
    const uint32_t tunedRows = static_cast<uint32_t>(
        std::min<int64_t>(x.GetDim(0), static_cast<int64_t>(std::numeric_limits<uint32_t>::max())));
    const uint32_t blockDim =
        LaunchBlocks(TunedBlocks(EpilogueKernel::GROUPED_NORM, static_cast<uint32_t>(width), tunedRows),
                     static_cast<uint64_t>(groups), aivNum);

    SglHyperConnGroupedNormTilingData td{};
    td.groups = static_cast<uint32_t>(groups);
    td.width = static_cast<uint32_t>(width);
    td.weightLen = static_cast<uint32_t>(weightLen);
    td.eps = eps;
    td.invWidth = 1.0f / static_cast<float>(width);

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

ge::graphStatus InferShapeHyperConnGroupedNorm(gert::InferShapeContext *context)
{
    const auto *src = context->GetInputShape(0);
    auto *dst = context->GetOutputShape(0);
    if (!src || !dst) {
        return ge::GRAPH_FAILED;
    }
    *dst = *src;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeHyperConnGroupedNorm(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

class SglHyperConnGroupedNorm : public OpDef
{
public:
    explicit SglHyperConnGroupedNorm(const char *name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-6f);
        this->SetInferShape(InferShapeHyperConnGroupedNorm).SetInferDataType(InferDataTypeHyperConnGroupedNorm);
        this->AICore().SetTiling(optiling::TilingHyperConnGroupedNorm);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglHyperConnGroupedNorm);

}  // namespace ops

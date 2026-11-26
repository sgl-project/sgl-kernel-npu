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
using sglang::npu_kernel::hyper_conn::TUNED_WIDTHS;
using sglang::npu_kernel::hyper_conn::TunedBlocks;

ge::graphStatus TilingHyperConnScaledSilu(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto *srcShape = context->GetInputShape(0);
    const auto *srcDesc = context->GetInputDesc(0);
    if (!srcShape || !srcDesc || srcDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const auto &src = srcShape->GetStorageShape();
    if (src.GetDimNum() != 1 || src.GetDim(0) <= 0) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const auto *rowsAttr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *colsAttr = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    const auto *strideAttr = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    const auto *scaleAttr = attrs ? attrs->GetAttrPointer<float>(3) : nullptr;
    if (!rowsAttr || !colsAttr || !strideAttr || !scaleAttr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = *rowsAttr;
    const int64_t cols = *colsAttr;
    const int64_t srcStride = *strideAttr;
    const float scale = *scaleAttr;
    int exponent = 0;
    if (rows <= 0 || rows > std::numeric_limits<uint32_t>::max() || cols <= 0 ||
        cols > static_cast<int64_t>(MAX_WIDTH) || cols % 16 != 0 || srcStride < cols ||
        srcStride > std::numeric_limits<uint32_t>::max() || !std::isfinite(scale) || scale <= 0.0f ||
        std::frexp(static_cast<double>(scale), &exponent) != 0.5) {
        return ge::GRAPH_FAILED;
    }
    const int64_t requiredSpan = (rows - 1) * srcStride + cols;
    if (src.GetDim(0) < requiredSpan) {
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
        LaunchBlocks(TunedBlocks(EpilogueKernel::SCALED_SILU, TUNED_WIDTHS[0], static_cast<uint32_t>(rows)),
                     static_cast<uint64_t>(rows), aivNum);

    SglHyperConnScaledSiluTilingData td{};
    td.rows = static_cast<uint32_t>(rows);
    td.cols = static_cast<uint32_t>(cols);
    td.srcStride = static_cast<uint32_t>(srcStride);
    td.invScale = 1.0f / scale;

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

ge::graphStatus InferShapeHyperConnScaledSilu(gert::InferShapeContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *rowsAttr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *colsAttr = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    auto *dst = context->GetOutputShape(0);
    if (!rowsAttr || !colsAttr || !dst) {
        return ge::GRAPH_FAILED;
    }
    dst->SetDimNum(2);
    dst->SetDim(0, *rowsAttr);
    dst->SetDim(1, *colsAttr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeHyperConnScaledSilu(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

class SglHyperConnScaledSilu : public OpDef
{
public:
    explicit SglHyperConnScaledSilu(const char *name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("rows").AttrType(OPTIONAL).Int(0);
        this->Attr("cols").AttrType(OPTIONAL).Int(0);
        this->Attr("src_stride").AttrType(OPTIONAL).Int(0);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0f);
        this->SetInferShape(InferShapeHyperConnScaledSilu).SetInferDataType(InferDataTypeHyperConnScaledSilu);
        this->AICore().SetTiling(optiling::TilingHyperConnScaledSilu);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglHyperConnScaledSilu);

}  // namespace ops

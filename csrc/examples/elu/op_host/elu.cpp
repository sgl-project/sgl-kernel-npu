// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../op_kernel/elu_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr uint64_t RESERVED_UB_BYTES = 16384;
constexpr uint64_t VECTOR_ALIGN_ELEMENTS = 512;
constexpr uint64_t TILE_ELEMENTS_CAP = 4096;
// inX(2 double-buffered) + outY(2 double-buffered) + tmpA(1) + tmpB(1)
constexpr uint64_t BUFFERS_PER_TILE = 6;

/*
 * Choose a per-tile element count that:
 *   - is a multiple of the 32B vector unit for the selected dtype;
 *   - keeps the peak UB footprint of six tiles inside the usable buffer;
 *   - is capped so that small inputs are not split into needlessly many tiles.
 */
uint32_t ComputeTileLength(uint64_t ubSize, int64_t elementSize)
{
    uint64_t usableUb = (ubSize > RESERVED_UB_BYTES) ? (ubSize - RESERVED_UB_BYTES) : ubSize;
    uint64_t bytesPerUnit = VECTOR_ALIGN_ELEMENTS * static_cast<uint64_t>(elementSize) * BUFFERS_PER_TILE;
    uint64_t alignUnits = (bytesPerUnit > 0) ? (usableUb / bytesPerUnit) : 0;
    if (alignUnits == 0) {
        alignUnits = 1;
    }
    uint64_t tileElements = alignUnits * VECTOR_ALIGN_ELEMENTS;
    if (tileElements > TILE_ELEMENTS_CAP) {
        tileElements = TILE_ELEMENTS_CAP;
    }
    return static_cast<uint32_t>(tileElements);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingElu(gert::TilingContext *context)
{
    const auto *xShape = context->GetInputShape(0);
    const auto *xDesc = context->GetInputDesc(0);
    const auto *attrs = context->GetAttrs();
    if (!xShape || !xDesc || !attrs || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }

    const auto &xStorageShape = xShape->GetStorageShape();
    const int64_t numel = xStorageShape.GetShapeSize();
    const auto *alphaAttr = attrs->GetAttrPointer<float>(0);
    if (!alphaAttr || !std::isfinite(*alphaAttr)) {
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_FLOAT16 && xDesc->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    // The Torch adapter always allocates the output with the input dtype, so
    // the Kernel is compiled per dtype and needs no runtime dtype branch.
    if (numel <= 0 || static_cast<uint64_t>(numel) > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t elementSize = xDesc->GetDataType() == ge::DT_FLOAT ? 4 : 2;
    const uint32_t tileLength = ComputeTileLength(ubSize, elementSize);

    // One core per tile, up to the number of AIV cores. Later the work is
    // split evenly over the participating cores, so a core may own a partial
    // tile; the Kernel's DataCopyPad handles that without host padding.
    int64_t tileCount = (numel + tileLength - 1) / tileLength;
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, tileCount));
    if (blockDim == 0) {
        blockDim = 1;
    }
    const uint32_t baseLength = static_cast<uint32_t>(numel / blockDim);
    const uint32_t extraBlocks = static_cast<uint32_t>(numel % blockDim);

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglEluTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SglEluTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = static_cast<uint32_t>(numel);
    tiling->tileLength = tileLength;
    tiling->blockDim = blockDim;
    tiling->baseLength = baseLength;
    tiling->extraBlocks = extraBlocks;
    tiling->alpha = *alphaAttr;

    raw->SetDataSize(sizeof(SglEluTilingData));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetInputShape(0);
    auto *yShape = context->GetOutputShape(0);
    if (!xShape || !yShape) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

class SglElu : public OpDef
{
public:
    explicit SglElu(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0f);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingElu);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglElu);

}  // namespace ops

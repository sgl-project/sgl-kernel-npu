// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Provenance: sgl-project/sglang sgl-kernel/csrc/speculative/eagle_utils.cu
// -> sgl-kernel-npu csrc/build_tree/ -> this independent CANN implementation.
// The original import revision is unknown. Computation is preserved; the native
// CANN interface and tiling transport are adapted here.

#include "../op_kernel/build_tree_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>
#include <initializer_list>

namespace optiling {

static ge::graphStatus TilingBuildTree(gert::TilingContext *context)
{
    const auto *parents = context->GetInputShape(0);
    const auto *mask = context->GetInputShape(3);
    const auto *attrs = context->GetAttrs();
    if (!parents || !mask || !attrs || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    const auto *topk = attrs->GetAttrPointer<int64_t>(0);
    const auto *depth = attrs->GetAttrPointer<int64_t>(1);
    const auto *draftTokenNum = attrs->GetAttrPointer<int64_t>(2);
    const auto *mode = attrs->GetAttrPointer<int64_t>(3);
    if (!topk || !depth || !draftTokenNum || !mode || *mode == sglang::npu_kernel::QLEN_ONLY_BITPACKING)
        return ge::GRAPH_FAILED;
    for (size_t index = 0; index < 8; ++index) {
        const auto *desc = context->GetInputDesc(index);
        if (!desc || desc->GetDataType() != (index == 3 ? ge::DT_BOOL : ge::DT_INT64)) return ge::GRAPH_FAILED;
    }
    const auto &parentShape = parents->GetStorageShape();
    const auto &maskShape = mask->GetStorageShape();
    if (parentShape.GetDimNum() == 0 || maskShape.GetDimNum() == 0) return ge::GRAPH_FAILED;

    // Preserve the old Host's field widths, per-core batch split and workspace.
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int32_t batchSize = parentShape.GetDim(0);
    const int32_t maskSize = maskShape.GetDim(0);
    const int32_t maxAivCore = static_cast<int32_t>(platform.GetCoreNumAiv());
    const int32_t blockDim = std::min(maxAivCore, batchSize);
    // Zero batches had no defined legacy execution (division by zero in Host).
    if (blockDim <= 0) return ge::GRAPH_FAILED;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    using TilingData = sglang::npu_kernel::BuildTreeTilingData;
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(TilingData) || !workspace) return ge::GRAPH_FAILED;
    auto *tiling = context->GetTilingData<TilingData>();
    if (!tiling) return ge::GRAPH_FAILED;
    tiling->batch_size = batchSize;
    tiling->mask_size = maskSize;
    tiling->topk = *topk;
    tiling->depth = *depth;
    tiling->draft_token_num = *draftTokenNum;
    tiling->tree_mask_mode = *mode;
    const auto numBigCore = batchSize % blockDim;
    tiling->big_core_num = numBigCore == 0 ? blockDim : numBigCore;
    tiling->big_core_tile_num = (batchSize + blockDim - 1) / blockDim;
    tiling->small_core_tile_num = batchSize / blockDim;
    raw->SetDataSize(sizeof(TilingData));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    workspace[0] = static_cast<int32_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferShapeBuildTree(gert::InferShapeContext *context)
{
    for (size_t index = 0; index < 5; ++index) {
        const auto *input = context->GetInputShape(index + 3);
        auto *output = context->GetOutputShape(index);
        if (!input || !output) return ge::GRAPH_FAILED;
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBuildTree(gert::InferDataTypeContext *context)
{
    for (size_t index = 0; index < 5; ++index)
        context->SetOutputDataType(index, context->GetInputDataType(index + 3));
    return ge::GRAPH_SUCCESS;
}

class SglBuildTree : public OpDef
{
public:
    explicit SglBuildTree(const char *name) : OpDef(name)
    {
        for (const auto *input : {"parent_list", "selected_index", "verified_seq_len"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Input("tree_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        for (const auto *input : {"positions", "retrive_index", "retrive_next_token", "retrive_next_sibling"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        // Same-name required outputs generate the ACLNN in-place Ref arguments.
        this->Output("tree_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        for (const auto *output : {"positions", "retrive_index", "retrive_next_token", "retrive_next_sibling"}) {
            this->Output(output)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const auto *attr : {"topk", "depth", "draft_token_num", "tree_mask_mode"})
            this->Attr(attr).AttrType(REQUIRED).Int();
        this->SetInferShape(InferShapeBuildTree).SetInferDataType(InferDataTypeBuildTree);
        this->AICore().SetTiling(optiling::TilingBuildTree);
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglBuildTree);
}  // namespace ops

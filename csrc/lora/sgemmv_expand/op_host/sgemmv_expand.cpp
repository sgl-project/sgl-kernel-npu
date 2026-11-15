// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#include "../op_kernel/sgemmv_expand_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>
#include <limits>

namespace optiling {

static ge::graphStatus TilingSglSgemmvExpand(gert::TilingContext *context)
{
    // Input order: x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y.
    const auto *xShapePtr = context->GetInputShape(0);
    const auto *weightShapePtr = context->GetInputShape(1);
    const auto *loraIndicesShapePtr = context->GetInputShape(2);
    const auto *seqLenShapePtr = context->GetInputShape(3);
    const auto *loraRanksShapePtr = context->GetInputShape(4);
    const auto *sliceOffsetsShapePtr = context->GetInputShape(5);
    const auto *yShapePtr = context->GetInputShape(6);
    const auto *yDesc = context->GetInputDesc(6);
    if (!xShapePtr || !weightShapePtr || !loraIndicesShapePtr || !seqLenShapePtr || !loraRanksShapePtr ||
        !sliceOffsetsShapePtr || !yShapePtr || !yDesc || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    // StorageShape exposes only GetStorageShape(); the dimensional queries live on gert::Shape.
    const auto &xShape = xShapePtr->GetStorageShape();
    const auto &weightShape = weightShapePtr->GetStorageShape();
    const auto &loraIndicesShape = loraIndicesShapePtr->GetStorageShape();
    const auto &seqLenShape = seqLenShapePtr->GetStorageShape();
    const auto &loraRanksShape = loraRanksShapePtr->GetStorageShape();
    const auto &sliceOffsetsShape = sliceOffsetsShapePtr->GetStorageShape();
    const auto &yShape = yShapePtr->GetStorageShape();

    // Mirror the legacy HOST_API checks that are observable from shapes.
    if (xShape.GetDimNum() != 2 || (weightShape.GetDimNum() != 3 && weightShape.GetDimNum() != 4) ||
        yShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (yDesc->GetDataType() != ge::DT_FLOAT16 && yDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    const int64_t loraIndicesSize = loraIndicesShape.GetDim(0);
    const int64_t seqLenSize = seqLenShape.GetDim(0);
    const int64_t loraRanksSize = loraRanksShape.GetDim(0);
    const int64_t sliceOffsetsSize = sliceOffsetsShape.GetDim(0);
    const int64_t batchSize = xShape.GetDim(0);
    const int64_t hiddenIn = xShape.GetDim(1);
    const int64_t outputFullDim = yShape.GetDim(1);
    // sliceOffsets holds slice_count + 1 entries; a smaller tensor would divide by zero.
    if (sliceOffsetsSize < 2 || loraIndicesSize < 0 || seqLenSize < 0 || loraRanksSize < 0 || batchSize < 0 ||
        hiddenIn < 0 || outputFullDim < 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t sliceCount = sliceOffsetsSize - 1;
    const int64_t maxLoRARank = hiddenIn / sliceCount;

    // The Kernel addresses every dimension with uint32, so reject values that
    // would be truncated instead of silently wrapping.
    constexpr uint64_t maxU32 = std::numeric_limits<uint32_t>::max();
    if (static_cast<uint64_t>(loraIndicesSize) > maxU32 || static_cast<uint64_t>(seqLenSize) > maxU32 ||
        static_cast<uint64_t>(loraRanksSize) > maxU32 || static_cast<uint64_t>(sliceOffsetsSize) > maxU32 ||
        static_cast<uint64_t>(batchSize) > maxU32 || static_cast<uint64_t>(maxLoRARank) > maxU32 ||
        static_cast<uint64_t>(outputFullDim) > maxU32) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t aivNum = platform.GetCoreNumAiv();
    if (aivNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t numTokensPerCore =
        (static_cast<uint64_t>(batchSize) + static_cast<uint64_t>(aivNum) - 1) / static_cast<uint64_t>(aivNum);
    // The legacy handler rejects batch == 0 through this same check.
    if (numTokensPerCore == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalBlocks = static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(sliceCount);
    const uint64_t blockDim = (totalBlocks + numTokensPerCore - 1) / numTokensPerCore;
    if (blockDim == 0 || blockDim > maxU32) {
        return ge::GRAPH_FAILED;
    }

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglSgemmvExpandTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SglSgemmvExpandTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    tiling->loraIndicesSize = static_cast<uint32_t>(loraIndicesSize);
    tiling->seqLenSize = static_cast<uint32_t>(seqLenSize);
    tiling->loraRanksSize = static_cast<uint32_t>(loraRanksSize);
    tiling->sliceOffsetsSize = static_cast<uint32_t>(sliceOffsetsSize);
    tiling->batchSize = static_cast<uint32_t>(batchSize);
    tiling->numTokensPerCore = static_cast<uint32_t>(numTokensPerCore);
    tiling->maxLoRARank = static_cast<uint32_t>(maxLoRARank);
    tiling->outputFullDim = static_cast<uint32_t>(outputFullDim);
    raw->SetDataSize(sizeof(SglSgemmvExpandTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // y is the seventh input and the only output; it is written in place.
    const auto *input = context->GetInputShape(6);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) {
        return ge::GRAPH_FAILED;
    }
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(6));
    return ge::GRAPH_SUCCESS;
}

class SglSgemmvExpand : public OpDef
{
public:
    explicit SglSgemmvExpand(const char *name) : OpDef(name)
    {
        // Type combinations: {float16, bfloat16}. x stays float32; the weights
        // and the in-place output share one dtype per combination.
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("lora_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("seq_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("lora_ranks")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("slice_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // A same-name output asks the generator for an in-place Ref parameter.
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingSglSgemmvExpand);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglSgemmvExpand);

}  // namespace ops

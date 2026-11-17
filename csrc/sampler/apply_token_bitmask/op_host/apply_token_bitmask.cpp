// BSD 3-Clause License
// Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

#include "../op_kernel/apply_token_bitmask_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

static ge::graphStatus TilingApplyTokenBitMask(gert::TilingContext *context)
{
    const auto *logits = context->GetInputShape(0);
    const auto *bitmask = context->GetInputShape(1);
    const auto *logitsDesc = context->GetInputDesc(0);
    const auto *bitmaskDesc = context->GetInputDesc(1);
    if (!logits || !bitmask || !logitsDesc || !bitmaskDesc || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &logitsShape = logits->GetStorageShape();
    const auto &bitmaskShape = bitmask->GetStorageShape();
    // Validate the existing Kernel's working-tensor contract, not the public Torch input.
    if (logitsShape.GetDimNum() != 2 || bitmaskShape.GetDimNum() != 2 || logitsShape.GetDim(0) < 0 ||
        logitsShape.GetDim(0) != bitmaskShape.GetDim(0) || logitsShape.GetDim(1) <= 0 ||
        logitsShape.GetDim(1) % 256 != 0 || bitmaskShape.GetDim(1) < logitsShape.GetDim(1) / 32 ||
        bitmaskDesc->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    int64_t dtypeSize = 0;
    switch (logitsDesc->GetDataType()) {
        case ge::DT_FLOAT:
            dtypeSize = 4;
            break;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            dtypeSize = 2;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t numIndices = logitsShape.GetDim(0);
    const int64_t paddedVocabSize = logitsShape.GetDim(1);
    const int64_t bitmaskStride = bitmaskShape.GetDim(1);
    const int64_t coreNum = platform.GetCoreNumAiv();
    constexpr uint64_t maxIndex = std::numeric_limits<uint32_t>::max();
    if (coreNum <= 0 || static_cast<uint64_t>(numIndices) > maxIndex ||
        static_cast<uint64_t>(paddedVocabSize) > maxIndex || static_cast<uint64_t>(bitmaskStride) > maxIndex)
        return ge::GRAPH_FAILED;
    if (numIndices > 0 &&
        ((static_cast<uint64_t>(numIndices) * paddedVocabSize - 1) > maxIndex ||
         ((static_cast<uint64_t>(numIndices) - 1) * bitmaskStride + paddedVocabSize / 32 - 1) > maxIndex)) {
        return ge::GRAPH_FAILED;
    }
    constexpr int64_t ALIGN_UNIT = 256;
    uint32_t blockDim = static_cast<uint32_t>(std::min(numIndices, coreNum));
    if (blockDim == 0) blockDim = 1;
    uint32_t baseRows = static_cast<uint32_t>(numIndices) / blockDim;
    uint32_t extraCores = static_cast<uint32_t>(numIndices) % blockDim;

    constexpr int32_t hostBufferNum = 2;
    uint64_t usableUb = (ubSize > 16384) ? (ubSize - 16384) : ubSize;
    int64_t bytesPerUnit = static_cast<int64_t>(hostBufferNum) *
                           (2 * ALIGN_UNIT * dtypeSize + (ALIGN_UNIT / 32) * static_cast<int64_t>(sizeof(int32_t)));
    if (usableUb < static_cast<uint64_t>(bytesPerUnit)) return ge::GRAPH_FAILED;
    uint32_t tileLength =
        static_cast<uint32_t>((usableUb / static_cast<uint64_t>(bytesPerUnit)) * static_cast<uint64_t>(ALIGN_UNIT));
    if (tileLength > static_cast<uint32_t>(paddedVocabSize)) {
        tileLength = static_cast<uint32_t>(paddedVocabSize);
    }
    if (tileLength < static_cast<uint32_t>(ALIGN_UNIT)) {
        tileLength = static_cast<uint32_t>(ALIGN_UNIT);
    }

    if (static_cast<uint64_t>(paddedVocabSize) + tileLength - 1 > maxIndex) return ge::GRAPH_FAILED;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglApplyTokenBitMaskTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SglApplyTokenBitMaskTilingData>();
    if (!tiling) return ge::GRAPH_FAILED;
    tiling->numRows = static_cast<uint32_t>(numIndices);
    tiling->vocabSize = static_cast<uint32_t>(paddedVocabSize);
    tiling->logitsStride = static_cast<uint32_t>(paddedVocabSize);
    tiling->bitmaskStride = static_cast<uint32_t>(bitmaskStride);
    tiling->baseRows = baseRows;
    tiling->extraCores = extraCores;
    tiling->tileLength = tileLength;
    tiling->blockDim = blockDim;
    tiling->dtypeSize = static_cast<uint32_t>(dtypeSize);
    raw->SetDataSize(sizeof(SglApplyTokenBitMaskTilingData));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

class SglApplyTokenBitMask : public OpDef
{
public:
    explicit SglApplyTokenBitMask(const char *name) : OpDef(name)
    {
        // These are the working tensors prepared by the existing caller.
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bitmask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // A same-name output asks the generator for an in-place Ref parameter.
        this->Output("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingApplyTokenBitMask);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglApplyTokenBitMask);

}  // namespace ops

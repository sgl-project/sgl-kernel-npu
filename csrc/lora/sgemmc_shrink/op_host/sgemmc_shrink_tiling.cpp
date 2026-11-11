// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
//
// Tiling for sgemmc_shrink: the legacy Host generated the cube tiling with
// MatmulApiTiling and launched one block per (token, slice) pair. The same
// formulas now produce a tiling blob consumed by the framework.

#include "../op_kernel/sgemmc_shrink_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <cstdint>
#include <limits>

namespace optiling {

static matmul_tiling::DataType ToMatmulDtype(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 ? matmul_tiling::DataType::DT_BFLOAT16 : matmul_tiling::DataType::DT_FLOAT16;
}

// Same settings as the legacy GenerateTiling call: A is a GM vector (M = 1),
// B is a transposed ND matrix and C stays float32 in VECIN.
static bool BuildCubeTiling(platform_ascendc::PlatformAscendC &platform, ge::DataType dtype, int64_t innerSize,
                            int64_t outputSize, AscendC::tiling::TCubeTiling &cubeTiling)
{
    matmul_tiling::MatmulApiTiling tiling(platform);
    const auto dataType = ToMatmulDtype(dtype);
    tiling.EnableBias(false);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::VECTOR, dataType, false);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType, true);
    tiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    tiling.SetOrgShape(1, outputSize, innerSize);
    tiling.SetShape(1, outputSize, innerSize);
    tiling.SetBufferSpace(-1, -1, -1);
    return tiling.GetTiling(cubeTiling) != -1;
}

static ge::graphStatus TilingSglSgemmcShrink(gert::TilingContext *context)
{
    // Input order: x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y.
    const auto *xShapePtr = context->GetInputShape(0);
    const auto *weightShapePtr = context->GetInputShape(1);
    const auto *loraIndicesShapePtr = context->GetInputShape(2);
    const auto *seqLenShapePtr = context->GetInputShape(3);
    const auto *loraRanksShapePtr = context->GetInputShape(4);
    const auto *loraScalesShapePtr = context->GetInputShape(5);
    const auto *yShapePtr = context->GetInputShape(6);
    const auto *xDesc = context->GetInputDesc(0);
    if (!xShapePtr || !weightShapePtr || !loraIndicesShapePtr || !seqLenShapePtr || !loraRanksShapePtr ||
        !loraScalesShapePtr || !yShapePtr || !xDesc || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &xShape = xShapePtr->GetStorageShape();
    const auto &weightShape = weightShapePtr->GetStorageShape();
    const auto &loraIndicesShape = loraIndicesShapePtr->GetStorageShape();
    const auto &seqLenShape = seqLenShapePtr->GetStorageShape();
    const auto &loraRanksShape = loraRanksShapePtr->GetStorageShape();
    const auto &loraScalesShape = loraScalesShapePtr->GetStorageShape();
    const auto &yShape = yShapePtr->GetStorageShape();

    // Mirror the legacy HOST_API checks that are observable from shapes.
    if (xShape.GetDimNum() != 2 || (weightShape.GetDimNum() != 3 && weightShape.GetDimNum() != 4) ||
        yShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_FLOAT16 && xDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    const auto *sliceCountPtr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const int64_t sliceCount = sliceCountPtr ? *sliceCountPtr : 1;

    const int64_t loraIndicesSize = loraIndicesShape.GetDim(0);
    const int64_t seqLenSize = seqLenShape.GetDim(0);
    const int64_t loraRanksSize = loraRanksShape.GetDim(0);
    const int64_t loraScalesSize = loraScalesShape.GetDim(0);
    const int64_t batchSize = xShape.GetDim(0);
    const int64_t inputHiddenDim = xShape.GetDim(1);
    const int64_t outputDim = yShape.GetDim(1);
    // The legacy Host required hidden_in > hidden_out (the shrink direction).
    if (sliceCount < 1 || loraIndicesSize < 0 || seqLenSize < 0 || loraRanksSize < 0 || loraScalesSize < 0 ||
        batchSize <= 0 || inputHiddenDim <= outputDim || outputDim < sliceCount) {
        return ge::GRAPH_FAILED;
    }
    const int64_t maxLoRARank = outputDim / sliceCount;

    // The Kernel addresses every dimension with uint32, so reject values that
    // would be truncated instead of silently wrapping.
    constexpr uint64_t maxU32 = std::numeric_limits<uint32_t>::max();
    if (static_cast<uint64_t>(loraIndicesSize) > maxU32 || static_cast<uint64_t>(seqLenSize) > maxU32 ||
        static_cast<uint64_t>(loraRanksSize) > maxU32 || static_cast<uint64_t>(loraScalesSize) > maxU32 ||
        static_cast<uint64_t>(batchSize) > maxU32 || static_cast<uint64_t>(inputHiddenDim) > maxU32 ||
        static_cast<uint64_t>(maxLoRARank) > maxU32 || static_cast<uint64_t>(sliceCount) > maxU32) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t blockDim = static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(sliceCount);
    if (blockDim == 0 || blockDim > maxU32) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglSgemmcShrinkTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SglSgemmcShrinkTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    if (!BuildCubeTiling(platform, xDesc->GetDataType(), inputHiddenDim, maxLoRARank, tiling->cubeTiling)) {
        return ge::GRAPH_FAILED;
    }

    tiling->loraIndicesSize = static_cast<uint32_t>(loraIndicesSize);
    tiling->seqLenSize = static_cast<uint32_t>(seqLenSize);
    tiling->loraRanksSize = static_cast<uint32_t>(loraRanksSize);
    tiling->loraScalesSize = static_cast<uint32_t>(loraScalesSize);
    tiling->batchSize = static_cast<uint32_t>(batchSize);
    tiling->inputHiddenDim = static_cast<uint32_t>(inputHiddenDim);
    tiling->maxLoRARank = static_cast<uint32_t>(maxLoRARank);
    tiling->slicesCount = static_cast<uint32_t>(sliceCount);
    raw->SetDataSize(sizeof(SglSgemmcShrinkTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);

    // The Kernel indexes per-block scratch as blockIdx * baseM * N: the cube
    // stages every N chunk of a block under that block's scratch base, so one
    // chunk per block is not enough and neighbouring blocks would overwrite
    // each other.
    const uint64_t perBlock =
        static_cast<uint64_t>(tiling->cubeTiling.baseM) * static_cast<uint64_t>(tiling->cubeTiling.N) * sizeof(float);
    workspace[0] = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize()) + blockDim * perBlock;
    return ge::GRAPH_SUCCESS;
}

// Keep the same empty prepare hook as the reference migrated operator.
struct SglSgemmcShrinkCompileInfo {};
static ge::graphStatus TilingPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SglSgemmcShrink).Tiling(TilingSglSgemmcShrink).TilingParse<SglSgemmcShrinkCompileInfo>(TilingPrepare);

}  // namespace optiling

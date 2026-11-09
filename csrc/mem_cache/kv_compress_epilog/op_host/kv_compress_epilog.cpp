/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/kv_compress_epilog_tiling_data.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstring>
namespace optiling {
constexpr int64_t QUANT_MODE_GROUP_MXFP8 = 2;
constexpr int64_t SLICE_SIZE = 64;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t DEFAULT_QUANT_GROUP_SIZE = 128;
constexpr int64_t DOUBLE_BUFFER = 2;
static int64_t CeilDiv(int64_t x, int64_t y)
{
    return y ? (x + y - 1) / y : x;
}
static int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}
static ge::graphStatus TilingKvCompressEpilog(gert::TilingContext *context)
{
    const auto *xShape = context->GetInputShape(1);
    const auto *cacheShape = context->GetInputShape(0);
    const auto *attrs = context->GetAttrs();
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!xShape || !cacheShape || !attrs || !raw || !raw->GetData() || !workspace || !context->GetPlatformInfo() ||
        !context->GetInputDesc(0) || !context->GetInputDesc(2) ||
        raw->GetCapacity() < sizeof(SglKvCompressEpilogTilingData))
        return ge::GRAPH_FAILED;
    const auto *group = attrs->GetAttrPointer<int64_t>(0);
    const auto *mode = attrs->GetAttrPointer<int64_t>(1);
    const auto *round = attrs->GetAttrPointer<bool>(2);
    const auto *layoutAttr = attrs->GetAttrPointer<int64_t>(3);
    const auto *strideAttr = attrs->GetAttrPointer<int64_t>(4);
    if (!group || !mode || !round || !layoutAttr || !strideAttr) return ge::GRAPH_FAILED;
    const auto &x = xShape->GetOriginShape();
    const auto &cache = cacheShape->GetOriginShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) <= 0 || x.GetDim(1) <= 0) return ge::GRAPH_FAILED;
    int64_t bs = x.GetDim(0), d = x.GetDim(1);
    int64_t quant_group_size = *group, quant_mode = *mode, round_scale = *round ? 1 : 0;
    int64_t layout = *layoutAttr, stride = *strideAttr;
    if (quant_mode == 2 && quant_group_size != 64 && quant_group_size != 128) return ge::GRAPH_FAILED;
    if (layout == 2 && (quant_mode != 2 || cache.GetDimNum() < 2 || cache.GetDim(1) <= 0)) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t coreNum = std::max<int64_t>(platform.GetCoreNumAiv(), 1);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t rowOfFormerBlock = CeilDiv(bs, coreNum);
    int64_t usedCoreNums = std::min(CeilDiv(bs, rowOfFormerBlock), coreNum);
    int64_t rowOfTailBlock = bs - (usedCoreNums - 1) * rowOfFormerBlock;

    int64_t scaleCol;
    if (quant_mode == QUANT_MODE_GROUP_MXFP8) {
        scaleCol = CeilDiv(d - SLICE_SIZE, quant_group_size);
    } else {
        scaleCol = CeilDiv(d - SLICE_SIZE, PER_BLOCK_FP16);
    }

    int64_t scaleBytes = 4;
    if (quant_mode == QUANT_MODE_GROUP_MXFP8) {
        scaleBytes = 1;
    }
    int64_t concatCol = d - SLICE_SIZE + SLICE_SIZE * 2 + scaleCol * scaleBytes;
    int64_t kvCacheCol = RoundUp(concatCol, DEFAULT_QUANT_GROUP_SIZE);
    int64_t padCol = kvCacheCol - concatCol;

    // Layout-2 specific calculations
    int64_t valuePerToken = 0;
    int64_t scalePerToken = 0;
    int64_t blockSize = 0;
    int64_t blockStride = 0;
    if (layout == 2) {
        blockSize = cache.GetDim(1);

        int64_t quantCol = d - SLICE_SIZE;
        valuePerToken = quantCol + SLICE_SIZE * 2;  // FP8 quant bytes + BF16 rope bytes
        scalePerToken = RoundUp(scaleCol, 8);

        int64_t autoBlockStride = blockSize * (valuePerToken + scalePerToken);
        int64_t blockStrideAttr = stride;
        if (blockStrideAttr > 0) {
            blockStride = blockStrideAttr;

        } else {
            blockStride = autoBlockStride;
        }
    } else {
        int64_t autoRowStride = kvCacheCol;
        int64_t blockStrideAttr = stride;
        if (blockStrideAttr > 0) {
            blockStride = blockStrideAttr;

        } else {
            blockStride = autoRowStride;
        }
    }

    // UB estimation: pre-compute per-row sizes
    int64_t xSizePerRow = RoundUp(d, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySizePerRow;
    int64_t scaleSizePerRow;
    if (layout == 2) {
        ySizePerRow = RoundUp(valuePerToken, 32) * 1 * DOUBLE_BUFFER;
        scaleSizePerRow = RoundUp(scalePerToken, 32) * 1;
    } else {
        ySizePerRow = RoundUp(kvCacheCol, 32) * 1 * DOUBLE_BUFFER;
        scaleSizePerRow = 0;
    }

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock, minRowPerCore);
    int64_t rowFactor = rowOnceLoop;
    while (rowFactor <= rowOfFormerBlock) {
        int64_t xSize = rowFactor * xSizePerRow;
        int64_t ySize = rowFactor * ySizePerRow;
        int64_t scaleSize = rowFactor * scaleSizePerRow;
        int64_t tmpBufferSize = RoundUp(rowFactor, 8) * 4;
        int64_t totalSize = xSize + ySize + scaleSize + tmpBufferSize;
        if (totalSize > static_cast<int64_t>(ubSize)) {
            rowFactor = rowFactor - 1;
            break;
        }
        rowFactor = rowFactor + 1;
    }
    if (rowFactor > rowOfFormerBlock) {
        rowFactor--;
    }

    if (rowFactor <= 0) return ge::GRAPH_FAILED;
    int64_t rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, rowFactor);
    int64_t rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, rowFactor);
    int64_t tailRowFactorOfFormerBlock = rowOfFormerBlock % rowFactor == 0 ? rowFactor : rowOfFormerBlock % rowFactor;
    int64_t tailRowFactorOfTailBlock = rowOfTailBlock % rowFactor == 0 ? rowFactor : rowOfTailBlock % rowFactor;

    // ---- Runtime dtype code: bit0 slot_mapping int64, bit1 cache e4m3fn ----
    int32_t dtypeCode = 0;
    if (context->GetInputDesc(2)->GetDataType() == ge::DT_INT64) {
        dtypeCode |= 1;
    }
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT8_E4M3FN) {
        dtypeCode |= 2;
    }

    SglKvCompressEpilogTilingData tilingData;
    tilingData.bs = bs;
    tilingData.d = d;
    tilingData.kvCacheCol = kvCacheCol;
    tilingData.scaleCol = scaleCol;
    tilingData.concatCol = concatCol;
    tilingData.padCol = padCol;
    tilingData.quantMode = quant_mode;
    tilingData.roundScale = round_scale;
    tilingData.perGroupSize = quant_group_size;
    tilingData.rowOfFormerBlock = rowOfFormerBlock;
    tilingData.rowOfTailBlock = rowOfTailBlock;
    tilingData.rowLoopOfFormerBlock = rowLoopOfFormerBlock;
    tilingData.rowLoopOfTailBlock = rowLoopOfTailBlock;
    tilingData.rowFactor = rowFactor;
    tilingData.tailRowFactorOfFormerBlock = tailRowFactorOfFormerBlock;
    tilingData.tailRowFactorOfTailBlock = tailRowFactorOfTailBlock;
    tilingData.layout = layout;
    tilingData.blockSize = blockSize;
    tilingData.valuePerToken = valuePerToken;
    tilingData.scalePerToken = scalePerToken;
    tilingData.blockStride = blockStride;
    tilingData.dtype = dtypeCode;
    tilingData.tilingKey = 0;

    std::memcpy(raw->GetData(), &tilingData, sizeof(tilingData));
    raw->SetDataSize(sizeof(tilingData));
    context->SetBlockDim(usedCoreNums);
    context->SetTilingKey(0);
    workspace[0] = platform.GetLibApiWorkSpaceSize() + 32;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferShapeKvCompressEpilog(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeKvCompressEpilog(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class SglKvCompressEpilog : public OpDef
{
public:
    explicit SglKvCompressEpilog(const char *name) : OpDef(name)
    {
        this->Input("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("quant_group_size").AttrType(REQUIRED).Int();
        this->Attr("quant_mode").AttrType(REQUIRED).Int();
        this->Attr("round_scale_flag").AttrType(REQUIRED).Bool();
        this->Attr("layout").AttrType(REQUIRED).Int();
        this->Attr("block_stride").AttrType(REQUIRED).Int();
        this->SetInferShape(optiling::InferShapeKvCompressEpilog)
            .SetInferDataType(optiling::InferDataTypeKvCompressEpilog);
        this->AICore().SetTiling(optiling::TilingKvCompressEpilog);
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglKvCompressEpilog);
}  // namespace ops

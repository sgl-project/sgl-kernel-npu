/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_compress_epilog.cpp
 * \brief Host-side tiling + launch for kv_compress_epilog (A5-only).
 */

#include <cstdio>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <functional>

#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/kv_compress_epilog_tiling_data.h"
#include "defines.h"
#include "torch_helper.h"
#include "common_tiling.h"
#include "common.h"
#include "aclrtlaunch_kv_compress_epilog.h"

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
constexpr int64_t WORKSPACE_SIZE = 32;

constexpr int64_t QUANT_MODE_GROUP_FP8 = 1;
constexpr int64_t QUANT_MODE_GROUP_MXFP8 = 2;
constexpr int64_t SLICE_SIZE = 64;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t DEFAULT_QUANT_GROUP_SIZE = 128;
constexpr int64_t DOUBLE_BUFFER = 2;

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

// Graph mode tiling cache
uint32_t actualCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> captureMap;

}  // namespace

HOST_API void kv_compress_epilog(at::Tensor &kv_compress_cache, const at::Tensor &x, const at::Tensor &slot_mapping,
                                 int64_t quant_group_size, int64_t quant_mode, bool round_scale_flag, int64_t layout)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D tensor, but got dimensions: ", x.dim());
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x dimensions must be positive, but got: [", x.size(0), ", ", x.size(1),
                "]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16, but got ", x.scalar_type());
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D tensor, but got dimensions: ", slot_mapping.dim());
    TORCH_CHECK(slot_mapping.size(0) == x.size(0),
                "slot_mapping size must equal x's first dimension, but got slot_mapping_size=", slot_mapping.size(0),
                ", x.dim(0)=", x.size(0));
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt || slot_mapping.scalar_type() == at::kLong,
                "slot_mapping must be INT32 or INT64, but got ", slot_mapping.scalar_type());
    TORCH_CHECK(
        kv_compress_cache.scalar_type() == at::kFloat8_e5m2 || kv_compress_cache.scalar_type() == at::kFloat8_e4m3fn,
        "kv_compress_cache must be FP8_E5M2 or FP8_E4M3FN, but got ", kv_compress_cache.scalar_type());

    at::Tensor cache = kv_compress_cache;
    if (cache.dim() == 4) {
        TORCH_CHECK(cache.size(2) == 1, "kv_compress_cache 4D tensor requires headnum (dim 2) == 1, but got ",
                    cache.size(2));
        cache = cache.squeeze(2);
    }

    const int64_t bs = x.size(0);
    const int64_t d = x.size(1);
    int64_t round_scale = round_scale_flag ? 1 : 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    if (coreNum <= 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int64_t rowOfFormerBlock = CeilDiv(bs, coreNum);
    int64_t usedCoreNums = std::min(CeilDiv(bs, rowOfFormerBlock), coreNum);
    int64_t rowOfTailBlock = bs - (usedCoreNums - 1) * rowOfFormerBlock;

    int64_t scaleCol;
    if (quant_mode == QUANT_MODE_GROUP_MXFP8) {
        TORCH_CHECK(quant_group_size == 64 || quant_group_size == 128, "MXFP8 quant_group_size must be 64 or 128, got ",
                    quant_group_size);
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
        TORCH_CHECK(quant_mode == QUANT_MODE_GROUP_MXFP8,
                    "layout=2 only supports MXFP8 (quant_mode=2), but got quant_mode=", quant_mode);
        TORCH_CHECK(cache.dim() >= 2, "layout=2 requires kv_compress_cache to be at least 2D, got dim ", cache.dim());
        blockSize = cache.size(1);
        TORCH_CHECK(blockSize > 0, "blockSize must be positive, got ", blockSize);

        int64_t quantCol = d - SLICE_SIZE;
        valuePerToken = quantCol + SLICE_SIZE * 2;  // FP8 quant bytes + BF16 rope bytes
        scalePerToken = RoundUp(scaleCol, 8);

        int64_t autoBlockStride = blockSize * (valuePerToken + scalePerToken);
        int64_t blockStrideAttr = cache.stride(0);
        if (blockStrideAttr > 0) {
            blockStride = blockStrideAttr;
            TORCH_CHECK(blockStride >= autoBlockStride, "block_stride (", blockStride,
                        ") must be >= auto-computed stride (", autoBlockStride, ")");
        } else {
            blockStride = autoBlockStride;
        }
    } else {
        int64_t autoRowStride = kvCacheCol;
        int64_t blockStrideAttr = cache.stride(0);
        if (blockStrideAttr > 0) {
            blockStride = blockStrideAttr;
            TORCH_CHECK(blockStride >= autoRowStride, "block_stride (", blockStride,
                        ") must be >= layout=1 row width (", autoRowStride, ")");
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

    int64_t rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, rowFactor);
    int64_t rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, rowFactor);
    int64_t tailRowFactorOfFormerBlock = rowOfFormerBlock % rowFactor == 0 ? rowFactor : rowOfFormerBlock % rowFactor;
    int64_t tailRowFactorOfTailBlock = rowOfTailBlock % rowFactor == 0 ? rowFactor : rowOfTailBlock % rowFactor;

    // ---- Runtime dtype code: bit0 slot_mapping int64, bit1 cache e4m3fn ----
    int32_t dtypeCode = 0;
    if (slot_mapping.scalar_type() == at::kLong) {
        dtypeCode |= 1;
    }
    if (kv_compress_cache.scalar_type() == at::kFloat8_e4m3fn) {
        dtypeCode |= 2;
    }

    // ---- Prepare tiling data ----
    sglang::KvCompressEpilogTilingData tilingData;
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

    uint32_t tilingSize = (sizeof(sglang::KvCompressEpilogTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    at::Tensor tilingTensor;

    // ---- Hash for the capture cache ----
    auto tup =
        std::make_tuple(bs, d, kvCacheCol, scaleCol, concatCol, padCol, quant_mode, round_scale, quant_group_size,
                        rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock, rowLoopOfTailBlock, rowFactor,
                        tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock, layout, blockSize, valuePerToken,
                        scalePerToken, blockStride, dtypeCode, tilingData.tilingKey);
    uint64_t hashValue = host_utils::TupleHasher::Hash(tup);

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(sglang::KvCompressEpilogTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(kv_compress_cache.device()));

    if (captureMap.find(hashValue) != captureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        captureMap[hashValue] = actualCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer.slice(0, actualCaptureNum * tilingSize, actualCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        actualCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    // ---- Workspace ----
    auto workspace_tensor =
        at::empty({WORKSPACE_SIZE}, at::TensorOptions().dtype(at::kByte).device(kv_compress_cache.device()));

    // ---- Launch (in-place: cache passed for both input and output slots) ----
    EXEC_KERNEL_CMD(kv_compress_epilog, usedCoreNums, cache, x, slot_mapping, cache, workspace_tensor, tilingTensor);
}

}  // namespace npu_kernel
}  // namespace sglang

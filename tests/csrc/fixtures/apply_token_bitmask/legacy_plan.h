// BSD 3-Clause License
// Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
// Test baseline: d974d3de5b7b0d6586a41f227cba93a861f07fe1, original Host tiling arithmetic.

#pragma once
#include <algorithm>
#include <cstdint>
inline sgl::BitmaskPlan LegacyPlan(int64_t numIndices, int64_t paddedVocabSize, int64_t bitmaskStride, int64_t dtypeSize, int64_t coreNum, uint64_t ubSize) {
    constexpr int64_t ALIGN_UNIT = 256;
    // Block dimension: one block per row, capped at coreNum
    uint32_t blockDim = static_cast<uint32_t>(std::min(static_cast<int64_t>(numIndices), coreNum));
    if (blockDim == 0) blockDim = 1;

    // Evenly distribute rows: first (numIndices % blockDim) cores get one extra row
    uint32_t baseRows = static_cast<uint32_t>(numIndices) / blockDim;
    uint32_t extraCores = static_cast<uint32_t>(numIndices) % blockDim;

    // Compute tileLength from UB size
    // UB per tile (double buffered, BUFFER_NUM=2):
    //   logitsQueue:  2 * tileLength * sizeof(T)
    //   bitmaskQueue: 2 * (tileLength/32) * sizeof(int32_t)
    //   outQueue:     2 * tileLength * sizeof(T)
    // Calculate at ALIGN_UNIT granularity to avoid integer division truncation
    constexpr int32_t hostBufferNum = 2;

    // Subtract 16KB for pipe overhead
    uint64_t usableUb = (ubSize > 16384) ? (ubSize - 16384) : ubSize;

    int64_t bytesPerUnit = static_cast<int64_t>(hostBufferNum) *
                           (2 * ALIGN_UNIT * dtypeSize + (ALIGN_UNIT / 32) * static_cast<int64_t>(sizeof(int32_t)));
    uint32_t tileLength =
        static_cast<uint32_t>((usableUb / static_cast<uint64_t>(bytesPerUnit)) * static_cast<uint64_t>(ALIGN_UNIT));

    // Cap tileLength to paddedVocabSize
    if (tileLength > static_cast<uint32_t>(paddedVocabSize)) {
        tileLength = static_cast<uint32_t>(paddedVocabSize);
    }

    // Safety floor
    if (tileLength < static_cast<uint32_t>(ALIGN_UNIT)) {
        tileLength = static_cast<uint32_t>(ALIGN_UNIT);
    }

    return {static_cast<uint32_t>(numIndices), static_cast<uint32_t>(paddedVocabSize),
            static_cast<uint32_t>(paddedVocabSize), static_cast<uint32_t>(bitmaskStride),
            baseRows, extraCores, tileLength, blockDim, static_cast<uint32_t>(dtypeSize)};
}

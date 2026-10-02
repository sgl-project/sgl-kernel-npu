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
 * \file sparse_attn_sharedkv_common.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_COMMON_H
#define SPARSE_ATTN_SHAREDKV_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../op_host/sparse_attn_sharedkv_tiling_data.h"

#define SWA_TEMPLATE 0
#define CFA_TEMPLATE 1
#define SCFA_TEMPLATE 2

namespace SASKernel {
using namespace AscendC;
using optiling::SparseAttnSharedkvTilingData;
// Set isCheckTiling to false; the input/output max, sum, and exp shapes are (m, 1).
constexpr SoftmaxConfig SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SAS_RUN_MODE {
    SWA_MODE = 0,
    SCFA_MODE = 1,
    CFA_MODE = 2,
};

enum class SAS_LAYOUT { BSND = 0, TND = 1, PA_ND = 2 };

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
          SAS_LAYOUT LAYOUT_T = SAS_LAYOUT::BSND, SAS_LAYOUT KV_LAYOUT_T = SAS_LAYOUT::PA_ND, int TEMPLATE_MODE = 0,
          typename... Args>
struct SASType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SAS_LAYOUT layout = LAYOUT_T;
    static constexpr SAS_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SAS_LAYOUT::PA_ND);
    static constexpr int templateMode = TEMPLATE_MODE;
};

// ================================Util functions==================================
template <typename T1, typename T2>
__aicore__ inline T1 SASAlign(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? a : b;
}

template <typename T>
__aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum;  // Usually the KV head count, corresponding to N2.
    uint32_t headDim;  // 512 corresponds to D.
    uint32_t kvStride;
    uint32_t maxblockNumPerBatch;  // Maximum number of entries in each block-table row.
    uint32_t actHeadDim;           // Actual copied column size accounting for N tiling; s*d, corresponding to D.
    uint32_t copyRowNum;           // Total number of rows to copy.
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
    uint32_t s1Idx;
};

// Scenario: copy query, key, and value from GM to L1.
// GM is stored in ND format.
// L1 is stored in NZ format.
// GM row count, column count, and column stride.
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,        // D
                                        uint32_t colStride)  // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;  // Number of rows in the ND matrix.
    // For int4 T, dValue = col / 2 and srcDValue = colStride / 2.
    nd2nzPara.dValue = col;           // Number of columns in the ND matrix.
    nd2nzPara.srcDValue = colStride;  // Offset between the start addresses of adjacent rows in the same ND matrix.
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

/*
    Copies PA data from GM to L1 and supports ND and NZ data.
    PA layouts include BNBD (blockNum, N, blockSize, D) and BBH (blockNum, blockSize, N*D).
    BSH, BSND, and TND use BBH.
    shape.copyRowNumAlign must be 16-byte aligned. For example, when copying a 128*512 K matrix,
    a 10*512 tail block must be aligned to 16*512.
*/
template <typename T>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,   // l1
                                  GlobalTensor<T> &srcTensor,  // gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,      // blockSize, headNum, headDim
                                  const Position &startPos)  // bacthIdx nIdx curSeqIdx
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;  // Get the index in the block table.
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;   // Get the row offset within a block.
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);  // Get the block ID from the block table.
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt;             // Process only one block at a time.
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;  // The current block is only partially copied.
        }
        // uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA offset.
        uint64_t offset = idInBlockTable * shape.kvStride;  // PA offset.
        uint64_t dStride = shape.headDim;
        offset +=
            (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim + startPos.dIdx;

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];
        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename T>
__aicore__ inline void DataCopyPABySlots(LocalTensor<T> &dstTensor,   // l1
                                         GlobalTensor<T> &srcTensor,  // gm
                                         GlobalTensor<int32_t> &sparseIndicesGm, const PAShape &shape,
                                         const Position &startPos, uint64_t sparseIndexBaseOffset,
                                         uint32_t sparseIndexStart)
{
    uint32_t blockElementCnt = 32 / sizeof(T);
    for (uint32_t row = 0; row < shape.copyRowNum; ++row) {
        int32_t slotId = sparseIndicesGm.GetValue(sparseIndexBaseOffset + sparseIndexStart + row);
        if (slotId < 0) {
            slotId = 0;
        }
        uint64_t blockId = static_cast<uint64_t>(slotId) / shape.blockSize;
        uint64_t blockOffset = static_cast<uint64_t>(slotId) % shape.blockSize;
        uint64_t offset = blockId * shape.kvStride;
        offset += static_cast<uint64_t>(startPos.n2Idx * shape.headDim * shape.blockSize) +
                  blockOffset * shape.headDim + startPos.dIdx;

        LocalTensor<T> tmpDstTensor = dstTensor[row * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];
        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, 1, shape.copyRowNumAlign, shape.actHeadDim, shape.headDim);
    }
}

struct RunInfo {
    uint32_t loop = 0;
    uint32_t cmpLoop = 0;  // Select one of the four GM blocks used for merging.
    uint32_t bIdx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t n2IdxReal = 0;
    uint32_t relativeS2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tndBIdxOffsetForQ = 0;
    uint64_t tndBIdxOffsetForKV = 0;
    uint64_t tensorCmpBOffset = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t qTokenOffset = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t topKBaseOffset = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    bool isFirstSInnerLoop = false;
    uint32_t s2BatchOffset = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t mSize = 0;
    uint32_t mSizeV = 0;
    uint32_t mSizeVStart = 0;
    uint32_t tndIsS2SplitCore = 0;
    uint32_t tndCoreStartKVSplitPos = 0;
    bool isBmm2Output = false;
    bool isValid = false;

    static constexpr uint32_t n2Idx = 0;
    uint64_t actS1Size = 1;
    uint64_t actS2SizeOri = 0ULL;
    uint32_t gS1Idx = 0;
    uint64_t actS2Size = 1;
    uint64_t actOriS2Size = 1;
    uint32_t actMBaseSize = 0;
    bool isLastS2Loop = 0;
    int32_t nextTokensPerBatch = 0;
    int64_t threshold = 0;
    uint32_t curTopKIdx = 0;
    uint64_t curOffsetInSparseBlock = 0;
    bool isOri = true;  // Whether the current block belongs to the Ori or Cmp part.
    uint64_t s2StartPoint = 0;
    int64_t cmpS2IdLimit = 0;
    int32_t v0S2DealSize = 0;
    int32_t v0S2Start = 0;
};

struct ConstInfo {
    // Synchronization mode between CUBE and VEC cores.
    static constexpr uint32_t SAS_SYNC_MODE2 = 2;
    // BUFFER size in bytes.
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // FP32 zero and maximum values.
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;

    // Total number of preload operations.
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // Event IDs for synchronization between CUBE and VEC cores.
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;

    uint32_t mmResUbSize = 0U;    // Matmul1 output size in GM.
    uint32_t vec1ResUbSize = 0U;  // Vector1 output size in GM.
    uint32_t bmm2ResUbSize = 0U;  // Matmul2 output size in GM.
    uint32_t usedCoreNum = 0U;
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0;
    uint64_t headDim = 0;
    uint64_t kvSeqSize = 0ULL;     // Maximum KV sequence length.
    uint64_t qSeqSize = 1ULL;      // Maximum Q sequence length.
    int64_t kvCacheBlockSize = 0;  // Block size for PA.
    uint64_t paCmpBlockSize = 0;
    uint64_t paOriBlockSize = 0;
    int64_t orikvCacheBlockSize = 0;
    int64_t cmpkvCacheBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;  // Maximum number of blocks per batch for PA.
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t splitKVNum = 0U;  // Number of S2 partitions across cores.
    SAS_LAYOUT outputLayout;   // Transpose format of the output.
    uint32_t oriMaskMode = 0;
    uint32_t cmpMaskMode = 0;
    uint32_t oriKvStride = 0;
    uint32_t cmpKvStride = 0;
    bool needInit = false;
    uint32_t templateMode = 0;

    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U;  // Maximum number of inter-core S2 partitions for FlashDecoding.
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint32_t actualLenDimsQ = 0U;   // Dimension of query actualSeqLength.
    uint32_t actualLenDimsKV = 0U;  // Dimension of KV actualSeqLength.

    // TND
    uint32_t s2Start = 0U;  // S2 start position for TND.
    uint32_t s2End = 0U;    // Upper S2 loop-index bound for a single core in TND.

    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;

    uint32_t tndFDCoreArrLen = 0U;      // Length of the TND FlashDecoding core-partition information array.
    uint32_t coreStartKVSplitPos = 0U;  // KV start position for TND FlashDecoding.

    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    // sparse attr
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0;
    bool hasOriSparseIndices = false;
    uint32_t oriSparseIndexWidth = 0;

    // cmp attr
    int64_t cmpRatio = 0;

    // win
    int32_t oriWinRight = 0;
    int32_t oriWinLeft = 128;

    // Whether to return SoftmaxLse.
    bool returnSoftmaxLse = false;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};
}  // namespace SASKernel
#endif  // SPARSE_ATTN_SHAREDKV_COMMON_H

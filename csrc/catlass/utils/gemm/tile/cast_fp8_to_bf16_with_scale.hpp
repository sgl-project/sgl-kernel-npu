/**

This program is free software, you can redistribute it and/or modify.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the
software repository for the full text of the License.
*/
#ifndef CATLASS_GEMM_TILE_CAST_FP8_TO_BF16_WITH_SCALE_HPP
#define CATLASS_GEMM_TILE_CAST_FP8_TO_BF16_WITH_SCALE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tile_traits_with_scale.hpp"

namespace Catlass::Gemm::Tile {
template <class ArchTag, class SrcType_, class DstType_, class ScaleType_, uint32_t COMPUTE_LENGTH>
struct TileCastFp8ToBf16WithScaleDequant {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;
    using LayoutRowMajor = Catlass::layout::RowMajor;

    using ElementScale = typename ScaleType_::Element;
    using LayoutScale = typename ScaleType_::Layout;

    static_assert(std::is_same_v<LayoutSrc, layout::RowMajor> || std::is_same_v<LayoutSrc, layout::ColumnMajor>,
                  "Unsupported layout, only can be Row/Column Major.");
    static_assert(std::is_same_v<LayoutDst, LayoutSrc>, "layout src and dst must be the same.");

    static const uint32_t Alignment = 256;

    struct Params {
        half scalar;
        half zeroPoint;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(half scalar_, half zeroPoint_)
        {
            scalar = scalar_;
            zeroPoint = zeroPoint_;
        }
    };

    CATLASS_DEVICE
    TileCastFp8ToBf16WithScaleDequant() {}

    CATLASS_DEVICE
    TileCastFp8ToBf16WithScaleDequant(Arch::Resource<ArchTag> const &resource, Params const &params_) : params(params_)
    {
        bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc));
            bufferOffset += COMPUTE_LENGTH;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            outputBuffer[i] = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                                  .template ReinterpretCast<bfloat16_t>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            workspace[i] = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                               .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            tmpFp32[i] = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                             .template ReinterpretCast<float>();
            bufferOffset += COMPUTE_LENGTH * 4;
        }

        InitLocalMaskVec(resource, bufferOffset);

        scaleCacheBuffer = (resource.ubBuf.template GetBufferByByte<int8_t>(bufferOffset * sizeof(int8_t)))
                               .template ReinterpretCast<float>();
        for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[i]);
        }
    }

    CATLASS_DEVICE
    ~TileCastFp8ToBf16WithScaleDequant()
    {
        for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> gmDst, LayoutDst const &layoutDst,
                    AscendC::GlobalTensor<ElementSrc> gmSrc, LayoutSrc const &layoutSrc,
                    LayoutScale const &layoutScalePerGroup, uint32_t const &groupSize, uint32_t const OffsetK)
    {
        uint32_t scaleTileRows = layoutScalePerGroup.shape(0);
        uint32_t scaleTileCols = layoutScalePerGroup.shape(1);
        uint32_t scaleStride = layoutScalePerGroup.stride(0);
        uint32_t scaleTileColsUB = RoundUp<32 / sizeof(ElementScale), uint32_t>(scaleTileCols);

        uint32_t tilesNum, tileLen, srcStride, dstStride;
        if constexpr (std::is_same_v<LayoutSrc, layout::RowMajor>) {
            tilesNum = layoutSrc.shape(0);
            tileLen = layoutSrc.shape(1);
            srcStride = layoutSrc.stride(0);
            dstStride = layoutDst.stride(0);
        } else if constexpr (std::is_same_v<LayoutSrc, layout::ColumnMajor>) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            srcStride = layoutSrc.stride(1);
            dstStride = layoutDst.stride(1);
        }

        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        uint32_t tilesRemain = tilesNum % AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < tilesRemain) {
            tilesPerAiv++;
        }

        uint64_t taskSrcOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * srcStride;
        uint64_t taskDstOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * dstStride;
        if (AscendC::GetSubBlockIdx() >= tilesRemain) {
            taskSrcOffset += tilesRemain * srcStride;
            taskDstOffset += tilesRemain * dstStride;
        }

        uint32_t baseTileStart = OffsetK + AscendC::GetSubBlockIdx() * tilesPerAiv;
        if (AscendC::GetSubBlockIdx() >= tilesRemain) {
            baseTileStart += tilesRemain;
        }

        uint32_t totalLoops = 0;
        uint32_t loopsPerTile, tilesInALoop;
        uint32_t tileLenRoundFp8 = RoundUp<Alignment, uint32_t>(tileLen);
        if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
            // One signle tile length is bigger than COMPUTE_LENGTH, which should be clipped.
            loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH);
            totalLoops = tilesPerAiv * loopsPerTile;
        } else if (tileLenRoundFp8 != 0) {
            // COMPUTE_LENGTH is bigger than tile length, such that more than one tiles can be arranged together.
            tilesInALoop = COMPUTE_LENGTH / tileLenRoundFp8;
            totalLoops = CeilDiv(tilesPerAiv, tilesInALoop);
        }  // tileLenRoundFp8 == 0 --> totalLoops = 0

        uint32_t tileTailLen = tileLen % COMPUTE_LENGTH;
        uint64_t srcProcessOffset, dstProcessOffset;
        uint32_t loadLen = COMPUTE_LENGTH, storeLen, loadRepeat = 1, storeRepeat = 1;

        uint32_t pingpong = 0;
        for (int ldx = 0; ldx < static_cast<int>(totalLoops); ldx++) {
            uint32_t validLen;
            if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
                uint32_t fullTileRounds = ldx / loopsPerTile;
                uint32_t residueTileRounds = ldx % loopsPerTile;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride + residueTileRounds * COMPUTE_LENGTH;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride + residueTileRounds * COMPUTE_LENGTH;

                loadLen = COMPUTE_LENGTH;
                if ((residueTileRounds == loopsPerTile - 1) && (tileTailLen != 0)) {
                    loadLen = tileTailLen;
                }
                loadRepeat = 1;
                storeLen = loadLen;
                storeRepeat = 1;

                validLen = loadLen * loadRepeat;
            } else {
                uint32_t fullTileRounds = ldx * tilesInALoop;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride;

                loadLen = tileLen;
                storeLen = tileLen;
                loadRepeat = tilesInALoop;
                storeRepeat = tilesInALoop;

                if ((ldx == static_cast<int>(totalLoops) - 1) && (tilesPerAiv % tilesInALoop != 0)) {
                    loadRepeat = tilesPerAiv % tilesInALoop;
                    storeRepeat = loadRepeat;
                }

                validLen = tileLenRoundFp8 * loadRepeat;
            }

            // GM -> UB (int8)
            AscendC::DataCopyExtParams dataCopyParamsIn(
                loadRepeat, loadLen * sizeof(ElementSrc), (srcStride - loadLen) * sizeof(ElementSrc),
                (tileLenRoundFp8 - loadLen) * sizeof(ElementSrc) / BYTE_PER_BLK, 0);
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[pingpong]);
            AscendC::DataCopyPad(inputBuffer[pingpong], gmSrc[srcProcessOffset], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[pingpong]);

            AscendC::LocalTensor<half> workspace2 = outputBuffer[pingpong].template ReinterpretCast<half>();

            Dequant(inputBuffer[pingpong], workspace2, value_vector1, value_vector2, workspace[pingpong], validLen);

            // fp16->fp32
            AscendC::Cast<float, half>(tmpFp32[pingpong], workspace2, AscendC::RoundMode::CAST_NONE, validLen);

            uint32_t baseRow = 0;
            uint32_t baseCol = 0;

            if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
                uint32_t fullTileRounds = ldx / loopsPerTile;
                uint32_t residueTileRounds = ldx % loopsPerTile;
                baseRow = baseTileStart + fullTileRounds;
                baseCol = residueTileRounds * COMPUTE_LENGTH;
            } else {
                uint32_t fullTileRounds = ldx * tilesInALoop;
                baseRow = baseTileStart + fullTileRounds;
                baseCol = 0;
            }

            const uint32_t dbBytes = AscendC::GetDataBlockSizeInBytes();
            const uint32_t rowStrideBytes = tileLenRoundFp8 * sizeof(float);
            const uint32_t rowStrideDb = rowStrideBytes / dbBytes;

            const float k256 = 256.0f;
            if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
                uint32_t sRow = baseRow / groupSize;
                uint32_t globalCol = baseCol;
                uint32_t localCol = 0;
                uint32_t remainLen = loadLen;

                while (remainLen > 0) {
                    uint32_t inGroup = globalCol % groupSize;
                    uint32_t colNuminGroup = groupSize - inGroup;
                    if (colNuminGroup > remainLen) {
                        colNuminGroup = remainLen;
                    }

                    uint32_t sCol = globalCol / groupSize;
                    uint32_t scaleIdx = sRow * scaleTileColsUB + sCol;
                    float scale = scaleCacheBuffer.GetValue(scaleIdx) * k256;

                    uint32_t off = localCol;
                    uint32_t r = colNuminGroup;
                    while (r > 0) {
                        uint32_t chunk = (r >= 64) ? 64 : r;
                        AscendC::Muls<float>(tmpFp32[pingpong][off], tmpFp32[pingpong][off], scale, (uint64_t)chunk, 1,
                                             {1, 1, (uint8_t)rowStrideDb, (uint8_t)rowStrideDb});
                        off += chunk;
                        r -= chunk;
                    }

                    globalCol += colNuminGroup;
                    localCol += colNuminGroup;
                    remainLen -= colNuminGroup;
                }
            } else {
                uint32_t rowNuminTurn = loadRepeat;
                uint32_t groupNuminRow = CeilDiv(rowNuminTurn, groupSize);
                uint32_t tailGroupRows = rowNuminTurn % groupSize;
                uint32_t groupNuminCol = tileLenRoundFp8 / groupSize;
                uint32_t curR, curC, rowNuminGroup, colNuminGroup;

                for (uint32_t g_row = 0; g_row < groupNuminRow; g_row++) {
                    curR = baseRow + g_row * groupSize;
                    if (g_row == groupNuminRow - 1 && tailGroupRows > 0) {
                        rowNuminGroup = tailGroupRows;
                    } else {
                        rowNuminGroup = groupSize;
                    }
                    for (uint32_t g_col = 0; g_col < groupNuminCol; g_col++) {
                        curC = g_col * groupSize;
                        colNuminGroup = groupSize;
                        uint32_t sRow = curR / groupSize;
                        uint32_t sCol = curC / groupSize;
                        uint32_t scaleIdx = 0;
                        scaleIdx = sRow * scaleTileColsUB + sCol;
                        float scale = scaleCacheBuffer.GetValue(scaleIdx) * k256;
                        uint32_t fp32Offset = (g_row * groupSize) * tileLenRoundFp8 + curC;
                        uint64_t mask = 64;
                        AscendC::Muls<float>(tmpFp32[pingpong][fp32Offset], tmpFp32[pingpong][fp32Offset], scale, mask,
                                             rowNuminGroup, {1, 1, (uint8_t)rowStrideDb, (uint8_t)rowStrideDb});
                        AscendC::Muls<float>(tmpFp32[pingpong][fp32Offset + 64], tmpFp32[pingpong][fp32Offset + 64],
                                             scale, mask, rowNuminGroup,
                                             {1, 1, (uint8_t)rowStrideDb, (uint8_t)rowStrideDb});
                    }
                }
            }
            pipe_barrier(PIPE_V);

            // fp32->bf16
            outputBuffer[pingpong] = workspace2.ReinterpretCast<bfloat16_t>();
            AscendC::Cast<bfloat16_t, float>(outputBuffer[pingpong], tmpFp32[pingpong], AscendC::RoundMode::CAST_RINT,
                                             validLen);
            pipe_barrier(PIPE_V);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[pingpong]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[pingpong]);

            // UB -> GM
            AscendC::DataCopyExtParams dataCopyParams(storeRepeat, storeLen * sizeof(ElementDst),
                                                      (tileLenRoundFp8 - storeLen) * sizeof(ElementDst) / BYTE_PER_C0,
                                                      (dstStride - storeLen) * sizeof(ElementDst), 0);
            AscendC::DataCopyPad(gmDst[dstProcessOffset], outputBuffer[pingpong], dataCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[pingpong]);

            pingpong = (pingpong + 1) % BUFFER_NUM;
        }
    }

    CATLASS_DEVICE
    void loadAllTileScales(uint32_t scaleGroupNumk, uint32_t scaleGroupNumN, uint32_t scaleStride,
                           AscendC::GlobalTensor<ElementScale> const gmScaleB)
    {
        AscendC::DataCopyParams copyScaleParams;
        copyScaleParams.blockCount = scaleGroupNumk;
        copyScaleParams.blockLen = scaleGroupNumN * sizeof(float);
        copyScaleParams.srcStride = (scaleStride - scaleGroupNumN) * sizeof(float);
        AscendC::DataCopyPadParams padScaleParams;
        AscendC::DataCopyPad(scaleCacheBuffer, gmScaleB, copyScaleParams, padScaleParams);
    }

private:
    CATLASS_DEVICE
    void InitLocalMaskVec(Arch::Resource<ArchTag> const &resource, int64_t &bufferOffset)
    {
        int16_t value_uint = 0x4000;
        value_vector1 = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector1, value_uint, 128);
        AscendC::PipeBarrier<PIPE_V>();
        value_uint = 0x3FFF;
        value_vector2 = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector2, value_uint, 128);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void Dequant(AscendC::LocalTensor<int8_t> &src, AscendC::LocalTensor<half> &dst,
                 AscendC::LocalTensor<int16_t> &value_vector1, AscendC::LocalTensor<int16_t> &value_vector2,
                 AscendC::LocalTensor<half> &workspace, uint32_t validLen)
    {
        pipe_barrier(PIPE_V);
        uint32_t num = validLen;
        num = (num + 128 - 1) / 128 * 128;
        AscendC::Cast<half, uint8_t>(dst.template ReinterpretCast<half>(), src.template ReinterpretCast<uint8_t>(),
                                     AscendC::RoundMode::CAST_NONE, num);
        pipe_barrier(PIPE_V);

        AscendC::Adds<half>(dst, dst, 1024, num);
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(dst.template ReinterpretCast<uint16_t>(), dst.template ReinterpretCast<uint16_t>(),
                                     7, num);
        pipe_barrier(PIPE_V);

        uint64_t mask = 128;
        AscendC::And<int16_t>(workspace.template ReinterpretCast<int16_t>(), dst.template ReinterpretCast<int16_t>(),
                              value_vector1, mask, num / 128, {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::And<int16_t>(dst.template ReinterpretCast<int16_t>(), dst.template ReinterpretCast<int16_t>(),
                              value_vector2, mask, num / 128, {1, 1, 1, 8, 8, 0});

        AscendC::ShiftLeft<uint16_t>(workspace.template ReinterpretCast<uint16_t>(),
                                     workspace.template ReinterpretCast<uint16_t>(), 1, num);
        pipe_barrier(PIPE_V);

        AscendC::Or<int16_t>(dst.template ReinterpretCast<int16_t>(), dst.template ReinterpretCast<int16_t>(),
                             workspace.template ReinterpretCast<int16_t>(), num);
        pipe_barrier(PIPE_V);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<int8_t> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<int16_t> value_vector1;
    AscendC::LocalTensor<int16_t> value_vector2;
    AscendC::LocalTensor<bfloat16_t> outputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<half> workspace[BUFFER_NUM];
    AscendC::LocalTensor<float> tmpFp32[BUFFER_NUM];
    AscendC::LocalTensor<float> scaleCacheBuffer;
    AscendC::TEventID EventIdBuffer[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndexForCast{0};
    int64_t bufferOffset{0};

    Params params;
};

}  // namespace Catlass::Gemm::Tile

#endif  // CATLASS_GEMM_TILE_CAST_FP8_TO_BF16_WITH_SCALE_HPP

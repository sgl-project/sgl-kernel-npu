/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_SOFTMAX_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/epilogue/tile/tile_copy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    class MaskType_>
class BlockEpilogue<
    EpilogueAtlasA2MLASoftmax,
    OutputType_,
    InputType_,
    MaskType_>
{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2MLASoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    using CopyGmToUbMask = Tile::CopyGm2Ub<ArchTag, MaskType_>;
    using CopyUbToGmOutput = Tile::CopyUb2Gm<ArchTag, OutputType_>;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;        // 64 * 128 * 2B
    static constexpr uint32_t UB_LINE_SIZE = 512;          // 128 * 2 * 2B
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;  // 128 * 2
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128; // 128
    static constexpr uint32_t MULTIPLIER = 2;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t T_BLOCK_SIZE = 32 / 2;
    static constexpr uint32_t UB_UINT8_LINE_SIZE = 512;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE_MLA = 16384;
    static constexpr uint32_t HALF_DM_UB_SIZE = 128;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, half tor_, uint32_t kvSplitCoreNum_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 0;
        // constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 6 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 9 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 13 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE_MLA;

        tor = tor_;
        kvSplitCoreNum = kvSplitCoreNum_;
        tvUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        lpUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(LP_UB_TENSOR_OFFSET);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor[0] = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        gmUbTensor[1] = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET + UB_UINT8_LINE_SIZE);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    CATLASS_DEVICE
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    CATLASS_DEVICE
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }
        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    CATLASS_DEVICE
    void ReduceSumRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        uint32_t curRowNum,
        uint32_t kSeqTile,
        uint32_t kSeqTileRound)
    {
        if (kSeqTile <= FLOAT_VECTOR_SIZE) {
            SetMask(kSeqTile);
            AscendC::RepeatReduceSum<float, false>(dst, src, curRowNum, 0, 0, 1, 1, kSeqTileRound / FLOAT_BLOCK_SIZE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            for (uint32_t rowsum_idx = 1; rowsum_idx < kSeqTile / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
                AscendC::Add<float, false>(
                    src,
                    src,
                    src[rowsum_idx * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1, kSeqTileRound / FLOAT_BLOCK_SIZE,
                        kSeqTileRound / FLOAT_BLOCK_SIZE, kSeqTileRound / FLOAT_BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (kSeqTile % FLOAT_VECTOR_SIZE > 0) {
                SetMask(kSeqTile % FLOAT_VECTOR_SIZE);
                AscendC::Add<float, false>(
                    src,
                    src,
                    src[kSeqTile / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1, kSeqTileRound / FLOAT_BLOCK_SIZE,
                        kSeqTileRound / FLOAT_BLOCK_SIZE, kSeqTileRound / FLOAT_BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::RepeatReduceSum<float, false>(dst, src, curRowNum, 0, 0, 1, 1, kSeqTileRound / FLOAT_BLOCK_SIZE);
        }
    }

    CATLASS_DEVICE
    void TensorSubValueRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &MaxTensor,
        const AscendC::LocalTensor<float> &tempMaxTensor,
        uint32_t curRowNum,
        uint32_t subMRound,
        uint32_t kSeqTile,
        uint32_t kSeqTileRound)
    {
        AscendC::Brcb(tempMaxTensor.ReinterpretCast<uint32_t>(), MaxTensor.ReinterpretCast<uint32_t>(),
                      subMRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < kSeqTile / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                dst[subIdx * FLOAT_VECTOR_SIZE],
                src[subIdx * FLOAT_VECTOR_SIZE],
                tempMaxTensor,
                (uint64_t)0,
                curRowNum,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, kSeqTileRound / FLOAT_BLOCK_SIZE, kSeqTileRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (kSeqTile % FLOAT_VECTOR_SIZE > 0) {
            SetMask(kSeqTile % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                dst[kSeqTile / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[kSeqTile / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tempMaxTensor,
                (uint64_t)0,
                curRowNum,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, kSeqTileRound / FLOAT_BLOCK_SIZE, kSeqTileRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
                        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound, 0, 1, 1,
                                                  numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0,
                1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowSumIdx = 1; rowSumIdx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[rowSumIdx * FLOAT_VECTOR_SIZE], numRowsRound, 0,
                                                      1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                  numRowsRound, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask((numElems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE);
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    CATLASS_DEVICE
    void ReduceMaxRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &tempTensor,
        uint32_t curRowNum,
        uint32_t kSeqTile,
        uint32_t kSeqTileRound)
    {
        if (kSeqTile <= FLOAT_VECTOR_SIZE) {
            SetMask(kSeqTile);
            AscendC::WholeReduceMax<float, false>(
                dst, src, (int32_t)0, curRowNum, 1, 1,
                kSeqTileRound / FLOAT_BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            AscendC::DataCopy(tempTensor, src, AscendC::DataCopyParams(curRowNum, HALF_VECTOR_SIZE / BLOCK_SIZE,
                (kSeqTileRound - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE, 0));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t rowmaxIdx = 1; rowmaxIdx < kSeqTile / FLOAT_VECTOR_SIZE; ++rowmaxIdx) {
                AscendC::Max<float, false>(
                    tempTensor,
                    tempTensor,
                    src[rowmaxIdx * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1, 8, 8, kSeqTileRound / FLOAT_BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (kSeqTile % FLOAT_VECTOR_SIZE > 0) {
                SetMask(kSeqTile % FLOAT_VECTOR_SIZE);
                AscendC::Max<float, false>(
                    tempTensor,
                    tempTensor,
                    src[kSeqTile / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1, 8, 8, kSeqTileRound / FLOAT_BLOCK_SIZE));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::WholeReduceMax<float, false>(
                dst, tempTensor, (int32_t)0, curRowNum, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        uint32_t nIdx,
        uint32_t softmaxPingPongFlag,
        uint32_t *glFlag, uint32_t taskPingPongFlag, uint32_t sUbOffset)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t kSeqTile = layoutInput.shape(1);
        uint32_t kSeqTileRound = layoutInput.stride(0);
        uint32_t subMRound = (curRowNum + 16 - 1) / 16 * 16;
        uint32_t sub_m_d64 = (curRowNum + 63) / 64; // up aligned to 128
        uint64_t dmUbOffsetCurCycle = (uint64_t)(softmaxPingPongFlag * HALF_DM_UB_SIZE);
        uint64_t llUbOffsetCurCycle = (uint64_t)(softmaxPingPongFlag * HALF_LL_UB_SIZE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(softmaxPingPongFlag + 6);
        AscendC::DataCopy(lsUbTensor[sUbOffset], gInput,
                          AscendC::DataCopyParams(1, curRowNum * kSeqTileRound / FLOAT_BLOCK_SIZE, 0, 0));

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // muls scale_value
        AscendC::Muls<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], tor, (uint64_t)0,
                                    (curRowNum * kSeqTileRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // *** lm = rowmax(ls)
        ReduceMaxRepeatM(lmUbTensor, lsUbTensor[sUbOffset], tvUbTensor, curRowNum, kSeqTile, kSeqTileRound);

        if (nIdx != 0) {
            AscendC::Max<float, false>(
                hmUbTensor,
                lmUbTensor,
                gmUbTensor[taskPingPongFlag],
                (uint64_t)0,
                sub_m_d64,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[taskPingPongFlag],
                hmUbTensor,
                (uint64_t)0,
                sub_m_d64,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(hmUbTensor, lmUbTensor, AscendC::DataCopyParams(1, subMRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        }
        // *** gm = hm
        AscendC::DataCopy(gmUbTensor[taskPingPongFlag], hmUbTensor, AscendC::DataCopyParams(1, subMRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();

        if (kvSplitCoreNum != 1) {
            if (nIdx == 0) {
                if (glFlag[taskPingPongFlag] == 1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(taskPingPongFlag + 2);
                    glFlag[taskPingPongFlag] = 0;
                }
            }
        }

        // *** ls = ls - hm_block
        TensorSubValueRepeatM(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                              hmUbTensor, tvUbTensor,
                              curRowNum, subMRound, kSeqTile, kSeqTileRound);

        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            (uint64_t)0,
            (curRowNum * kSeqTileRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();
        RowsumTAILTILE(lsUbTensor[sUbOffset], llUbTensor[llUbOffsetCurCycle], tvUbTensor, curRowNum, kSeqTile, kSeqTileRound);

        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                tvUbTensor16[sUbOffset * 2], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                (curRowNum * kSeqTileRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                tvUbTensor16[sUbOffset * 2], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                (curRowNum * kSeqTileRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        uint16_t blockCount = 1;
        uint16_t blockLen = curRowNum * kSeqTileRound / T_BLOCK_SIZE;
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;

        AscendC::DataCopy(gOutput,
                          tvUbTensor16[sUbOffset * 2],
                          AscendC::DataCopyParams(
                              blockCount,
                              blockLen,
                              srcStride,
                              dstStride));

        // *** ll = rowsum(ls32)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(softmaxPingPongFlag + 6);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        GemmCoord actualBlockShape,
        uint32_t nIdx,
        uint32_t curHeadNum,
        uint32_t softmaxPingPongFlag,
        uint32_t *glFlag, uint32_t taskPingPongFlag)
    {
        uint32_t rowActual = actualBlockShape.m();
        uint32_t nActual = actualBlockShape.n();
        uint32_t tokenNumPerHead = rowActual / curHeadNum;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t sUbOffset = softmaxPingPongFlag * 8192;

        uint32_t curHeadSplitSubBlock = curHeadNum / subBlockNum;
        uint32_t curHeadThisSubBlock = (subBlockIdx == 0) ? curHeadSplitSubBlock : (curHeadNum - curHeadSplitSubBlock);

        uint32_t rowActualThisSubBlock = curHeadThisSubBlock * tokenNumPerHead;
        uint32_t rowOffsetSubBlock = subBlockIdx * curHeadSplitSubBlock * tokenNumPerHead;

        if (rowActualThisSubBlock > 0) {
            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetSubBlock, 0));
            auto gInputThisSubBlock = gInput[offsetInput];
            auto layoutInputThisSubBlock = layoutInput.GetTileLayout(MatrixCoord(rowActualThisSubBlock, nActual));
            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetSubBlock, 0));
            auto gOutputThisSubBlock = gOutput[offsetOutput];
            auto layoutOutputThisSubBlock = layoutOutput.GetTileLayout(MatrixCoord(rowActualThisSubBlock, nActual));
            SubCoreCompute(gOutputThisSubBlock, gInputThisSubBlock, layoutOutputThisSubBlock, layoutInputThisSubBlock,
                           nIdx, softmaxPingPongFlag, glFlag, taskPingPongFlag, sUbOffset);
        }
    }

private:
    float tor;
    uint32_t pingpongFlag = 0;
    uint32_t kvSplitCoreNum = 1;
    AscendC::LocalTensor<ElementOutput> tvUbTensor16;
    AscendC::LocalTensor<float> lpUbTensor32;
    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor[2];
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;

    CopyGmToUbInput copyGmToUbInput;
    CopyGmToUbMask copyGmToUbMask;
    CopyUbToGmOutput copyUbToGmOutput;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_SOFTMAX_HPP
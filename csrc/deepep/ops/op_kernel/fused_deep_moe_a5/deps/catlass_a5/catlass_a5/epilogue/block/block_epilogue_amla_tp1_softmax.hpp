/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_AMLA_TP1_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_AMLA_TP1_SOFTMAX_HPP

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
    class UpdateType_,
    class MaskType_>
class BlockEpilogue<
    EpilogueAtlasA2AMLATP1Softmax,
    OutputType_,
    InputType_,
    UpdateType_,
    MaskType_>
{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2AMLATP1Softmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;

    using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    using CopyGmToUbMask = Tile::CopyGm2Ub<ArchTag, MaskType_>;
    using CopyUbToGmOutput = Tile::CopyUb2Gm<ArchTag, OutputType_>;

    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_LINE_SIZE = 512;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE_MLA = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t S_BLOCK_STACK = 4;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;
    static constexpr uint32_t M_SLICE = 16;
    static constexpr uint32_t QK_READY_ID = 1;
    static constexpr uint32_t SOFTMAX_READY_ID = 2;

    static constexpr float LN2 = 0.6931471805599453094172;
    static constexpr float RECIP_OF_LN2 = 1 / LN2;
    static constexpr float FLOAT_E_SCALAR = 8388608;
    static constexpr uint32_t PRE_AUTOADD_READY_ID = 0;
    static constexpr uint32_t SM_AUTOADD_READY_ID = 7;
    static constexpr uint32_t BRCB_SIZE = 128;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, half tor_, uint32_t kvSplitCoreNum_ = 1)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 2 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 4 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 5 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 7 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE_MLA;

        constexpr uint32_t NBRCB_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t HN_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 9 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t DN_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 10 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GN_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 12 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t S32_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 14 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t S16Tmp_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 16 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t S16_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 18 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t HC_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 20 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GC_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 22 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t EPS_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 24 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t N_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 26 * UB_UINT8_LINE_SIZE;

        tor = tor_;
        kvSplitCoreNum = kvSplitCoreNum_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor[0] = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        gmUbTensor[1] = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET + UB_UINT8_LINE_SIZE);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor[0] = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        glUbTensor[1] = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET + UB_UINT8_LINE_SIZE);

        hnUbTensor = resource.ubBuf.template GetBufferByByte<float>(HN_UB_TENSOR_OFFSET);
        dnUbTensor = resource.ubBuf.template GetBufferByByte<float>(DN_UB_TENSOR_OFFSET);
        gnUbTensor = resource.ubBuf.template GetBufferByByte<float>(GN_UB_TENSOR_OFFSET);
        s32UbTensor = resource.ubBuf.template GetBufferByByte<float>(S32_UB_TENSOR_OFFSET);
        s16UbTensorTmp = resource.ubBuf.template GetBufferByByte<ElementOutput>(S16Tmp_UB_TENSOR_OFFSET);
        NBrcbUbTensor = resource.ubBuf.template GetBufferByByte<int32_t>(NBRCB_UB_TENSOR_OFFSET);
        s16UbTensor = resource.ubBuf.template GetBufferByByte<float>(S16_UB_TENSOR_OFFSET);
        hcUbTensor = resource.ubBuf.template GetBufferByByte<float>(HC_UB_TENSOR_OFFSET);
        gcUbTensor = resource.ubBuf.template GetBufferByByte<float>(GC_UB_TENSOR_OFFSET);
        epsUbTensor = resource.ubBuf.template GetBufferByByte<float>(EPS_UB_TENSOR_OFFSET);
        NUbTensor = resource.ubBuf.template GetBufferByByte<int32_t>(N_UB_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

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
    void SetkvSplitCoreNum(uint32_t kvSplitCoreNum_)
    {
        kvSplitCoreNum = kvSplitCoreNum_;
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
    void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
                           const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                           uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                              numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
                                              numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
                           const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                           uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor, numRowsRound, 0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(
            rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
            (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
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
    void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
                           const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                           uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                              numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
                                              numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
                           const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                           uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor, numRowsRound, 0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
            (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    CATLASS_DEVICE
    void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
                        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
                        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound, 0, 1, 1,
                                                  numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0,
                1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[rowmax_idx * FLOAT_VECTOR_SIZE], numRowsRound,
                                                      0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                  numRowsRound, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask((numElems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE);
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(
                    rowmaxUb, tvUbTensor, (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    CATLASS_DEVICE
    void SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
                        uint32_t m, uint32_t nReal, uint32_t nStride, uint32_t pingpongFlag, uint32_t rowOffset,
                        uint32_t sUbOffset, uint32_t nIdx, uint32_t isLastNTile, uint32_t *glFlag,
                        uint32_t taskPingPongFlag, uint32_t gSPingPongFlag, uint32_t isLastM, uint32_t isFirstM)
    {
        uint32_t round_m = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(pingpongFlag);
        // input QK
        AscendC::DataCopy(lsUbTensor[sUbOffset], gInput, AscendC::DataCopyParams(m, nStride / FLOAT_BLOCK_SIZE, 0, 0));

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);

        // *** ls = tor * ls
        AscendC::Muls<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], tor, (uint64_t)0,
                                    (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();


        if (nReal == 512) {
            RowmaxSPECTILE512(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, round_m, nReal, nStride);
        } else if (nReal == 256) {
            RowmaxSPECTILE256(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, round_m, nReal, nStride);
        } else {
            RowmaxTAILTILE(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, round_m, nReal, nStride);
        }


        if (nIdx == 0) {
            AscendC::DataCopy(hmUbTensor[rowOffset], lmUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, round_m / FLOAT_BLOCK_SIZE, 0, 0));

            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(m);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(hmUbTensor[rowOffset], lmUbTensor[rowOffset], gmUbTensor[taskPingPongFlag][rowOffset], (uint64_t)0,
                                       1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(dmUbTensor[rowOffset],
                                       gmUbTensor[taskPingPongFlag][rowOffset], hmUbTensor[rowOffset], (uint64_t)0, 1,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(dmUbTensor[rowOffset],
                                       dmUbTensor[rowOffset],
                                       (uint64_t)0, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }

        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(gmUbTensor[taskPingPongFlag][rowOffset], hmUbTensor[rowOffset],
                          AscendC::DataCopyParams(1, round_m / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();

        // ni = hnUbTensor = mi / -LN2
        SetVecMask(m);
        AscendC::Muls<float, false>(
                hnUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (-1.0f) * RECIP_OF_LN2,
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, float, false>(
            hnUbTensor[rowOffset], hnUbTensor[rowOffset], AscendC::RoundMode::CAST_ROUND, (uint64_t)0,
            1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // dnUbTensor = ni - ni-1
        if (nIdx != 0) {
            AscendC::Sub<float, false>(
                dnUbTensor[rowOffset],
                hnUbTensor[rowOffset],
                gnUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // n1 = n0, dn = 0
            AscendC::Duplicate<float, false>(dnUbTensor[rowOffset], 0.0f, (uint64_t)0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::DataCopy(gnUbTensor[rowOffset], hnUbTensor[rowOffset], AscendC::DataCopyParams(1, round_m / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();

        // mi / LN2
        SetVecMask(m);
        AscendC::Muls<float, false>(
                tvUbTensor,
                hmUbTensor[rowOffset],
                RECIP_OF_LN2,
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // hnUbTensor = ni + mi / LN2
        AscendC::Add<float, false>(
            hnUbTensor[rowOffset],
            hnUbTensor[rowOffset],
            tvUbTensor,
            (uint64_t)0,
            1,
            AscendC::BinaryRepeatParams(
                1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // ln2 * (ni + mi / ln2)
        AscendC::Muls<float, false>(
                hnUbTensor[rowOffset],
                hnUbTensor[rowOffset],
                LN2,
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // s32 = exp(ln2 * (ni + mi / ln2))
        AscendC::Exp<float, false>(
            s32UbTensor[rowOffset],
            hnUbTensor[rowOffset],
            (uint64_t)0,
            1,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // s16 = s32 -> s16
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                s16UbTensorTmp[rowOffset], s32UbTensor[rowOffset], AscendC::RoundMode::CAST_ROUND, (uint64_t)0,
                1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                s16UbTensorTmp[rowOffset], s32UbTensor[rowOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();
        // s16 = s16 -> s16
        AscendC::Cast<float, ElementOutput, false>(
            s16UbTensor[rowOffset], s16UbTensorTmp[rowOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
            1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::PipeBarrier<PIPE_V>();

        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::Brcb(tvUbTensor.template ReinterpretCast<uint32_t>(),
                      hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(), round_m / FLOAT_BLOCK_SIZE,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // *** ls = ls - hm_block
        for (uint32_t subIdx = 0; subIdx < nReal / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE], lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor, (uint64_t)0, m,
                AscendC::BinaryRepeatParams(1, 1, 0, nStride / FLOAT_BLOCK_SIZE, nStride / FLOAT_BLOCK_SIZE, 1));
        }
        if (nReal % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(nReal % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][nReal / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][nReal / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0, m,
                AscendC::BinaryRepeatParams(1, 1, 0, nStride / FLOAT_BLOCK_SIZE, nStride / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // *** ls = exp(ls)
        AscendC::Exp<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], (uint64_t)0,
                                   (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                   AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // *** ll = rowsum(ls32)
        if (nReal == 512) {
            RowsumSPECTILE512(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, round_m, nReal, nStride);
        } else if (nReal == 256) {
            RowsumSPECTILE256(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, round_m, nReal, nStride);
        } else {
            RowsumTAILTILE(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, round_m, nReal, nStride);
        }

        AscendC::Brcb(
            tvUbTensor.ReinterpretCast<uint32_t>(),
            s16UbTensor.ReinterpretCast<uint32_t>()[rowOffset],
            round_m / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // *** p = p * s16
        for (uint32_t subIdx = 0; subIdx < nReal / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Mul<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE], lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor, (uint64_t)0, m,
                AscendC::BinaryRepeatParams(1, 1, 0, nStride / FLOAT_BLOCK_SIZE, nStride / FLOAT_BLOCK_SIZE, 1));

        }
        if (nReal % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(nReal % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(
                lsUbTensor[sUbOffset][nReal / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][nReal / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0, m,
                AscendC::BinaryRepeatParams(1, 1, 0, nStride / FLOAT_BLOCK_SIZE, nStride / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset * 2], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset * 2], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        AscendC::DataCopy(gOutput, lpUbTensor[sUbOffset * 2], AscendC::DataCopyParams(m, nStride * 2 / 32, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(pingpongFlag);

        if (isLastM) {
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
        }
        if (nIdx == 0) {
            // *** gl = ll
            AscendC::DataCopy(glUbTensor[taskPingPongFlag][rowOffset], llUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, round_m / FLOAT_BLOCK_SIZE, 0, 0));

            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(m);
            // // *** gl = dm * gl
            AscendC::Mul<float, false>(
                glUbTensor[taskPingPongFlag][rowOffset], dmUbTensor[rowOffset],
                glUbTensor[taskPingPongFlag][rowOffset], (uint64_t)0, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            // // *** gl = ll + gl
            AscendC::Add<float, false>(glUbTensor[taskPingPongFlag][rowOffset], glUbTensor[taskPingPongFlag][rowOffset], llUbTensor[rowOffset], (uint64_t)0,
                                       1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        // -----------------------------------------
        SetVecMask(m);
        if (isLastNTile) {
            AscendC::Mul<float, false>(
                glUbTensor[taskPingPongFlag][rowOffset],
                glUbTensor[taskPingPongFlag][rowOffset],
                s16UbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (nIdx > 0) {
            AscendC::Div<float, false>(
                hcUbTensor[rowOffset],
                s32UbTensor[rowOffset],
                s16UbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }
        // ci = s32 / s16
        if (nIdx == 0) {
            AscendC::Duplicate<float, false>(gcUbTensor[rowOffset], 1.0, (uint64_t)0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (nIdx > 0) {
            // eps = ci-1 / ci
            AscendC::Div<float, false>(
                epsUbTensor[rowOffset],
                gcUbTensor[rowOffset],
                hcUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::DataCopy(gcUbTensor[rowOffset], hcUbTensor[rowOffset], AscendC::DataCopyParams(1, round_m / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
            // eps = eps - 1.0
            SetVecMask(m);
            AscendC::Adds<float, false>(
                epsUbTensor[rowOffset],
                epsUbTensor[rowOffset],
                (float)(-1.0f),
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // eps = eps * 1.5
            AscendC::Muls<float, false>(
                epsUbTensor[rowOffset],
                epsUbTensor[rowOffset],
                (float)(1.5f),
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // max((ni - ni-1), -30.0)
            AscendC::Maxs<float, false>(
                dnUbTensor[rowOffset],
                dnUbTensor[rowOffset],
                (float)(-30.0f),
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // eps = eps + 10^-6
            AscendC::Adds<float, false>(
                epsUbTensor[rowOffset],
                epsUbTensor[rowOffset],
                (float)(0.000001f),
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // eps + 10^-6 + max((ni - ni-1), -30.0)
            AscendC::Add<float, false>(
                epsUbTensor[rowOffset],
                epsUbTensor[rowOffset],
                dnUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(
                    1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // eps = eps * 2e23
            AscendC::Muls<float, false>(
                epsUbTensor[rowOffset],
                epsUbTensor[rowOffset],
                FLOAT_E_SCALAR,
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // N
            AscendC::Cast<int32_t, float, false>(
                NUbTensor[gSPingPongFlag * UB_FLOAT_LINE_SIZE + rowOffset], epsUbTensor[rowOffset], AscendC::RoundMode::CAST_ROUND, (uint64_t)0,
                1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    CATLASS_DEVICE
    void AmlaAutoadd(AscendC::GlobalTensor<int32_t> gOTmpInt32,
                        uint32_t subM, uint32_t embed, uint32_t gSPingPongFlag
                    )
    {
        // update OTmp
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        for (uint32_t i = 0; i < BRCB_SIZE / FLOAT_BLOCK_SIZE; i++){
            AscendC::Brcb(
                NBrcbUbTensor.ReinterpretCast<uint32_t>()[i * FLOAT_BLOCK_SIZE],
                NUbTensor.ReinterpretCast<uint32_t>()[gSPingPongFlag * UB_FLOAT_LINE_SIZE],
                subM / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(BRCB_SIZE / FLOAT_BLOCK_SIZE, BRCB_SIZE));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);

        Arch::CrossCoreWaitFlag(preAutoAddReady);

        AscendC::SetAtomicAdd<int32_t>();

        for (uint32_t i = 0; i < embed / BRCB_SIZE; i++) {
            AscendC::DataCopy(
                gOTmpInt32[i * BRCB_SIZE],
                NBrcbUbTensor,
                AscendC::DataCopyParams(
                    subM, BRCB_SIZE / FLOAT_BLOCK_SIZE, 0, (embed - BRCB_SIZE) / FLOAT_BLOCK_SIZE));
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        AscendC::SetAtomicNone();
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
                    AscendC::GlobalTensor<int32_t> gOTmpInt32,
                    const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
                    const LayoutUpdate &layoutOTmp,
                    GemmCoord actualBlockShape,
                    uint32_t nIdx, uint32_t isLastNTile, uint32_t *glFlag, uint32_t taskPingPongFlag, uint32_t gSPingPongFlag)
    {
        uint32_t cur_head_num = actualBlockShape.m();
        uint32_t qkN = actualBlockShape.n();
        uint32_t embed = actualBlockShape.k();
        uint32_t qkRoundN = layoutInput.stride(0);

        uint32_t pingpongFlag = 0;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t subM = (subBlockIdx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2;

        uint32_t mEnd = (subM + M_SLICE - 1) / M_SLICE;

        for (uint32_t mInd = 0; mInd < mEnd; mInd++) {
            uint32_t rowOffset = mInd * M_SLICE;
            uint32_t currM = mInd == mEnd - 1 ? subM - rowOffset : M_SLICE;
            uint32_t isLastM = mInd == mEnd - 1 ? 1 : 0;
            uint32_t isFirstM = mInd == 0 ? 1 : 0;
            uint32_t sUbOffset = pingpongFlag * 8192;
            int64_t offsetOutput = rowOffset * qkRoundN;
            auto gOutputThisSubBlock = gOutput[offsetOutput];
            int64_t offsetInput = rowOffset * qkRoundN;
            auto gInputThisSubBlock = gInput[offsetInput];
            if (mInd == 0) {
                Arch::CrossCoreWaitFlag(qkReady);
            }
            if (currM == 0) {
                continue;
            }
            SubCoreCompute(gOutputThisSubBlock, gInputThisSubBlock, currM, qkN, qkRoundN, pingpongFlag, rowOffset,
                           sUbOffset, nIdx, isLastNTile, glFlag, taskPingPongFlag, gSPingPongFlag, isLastM, isFirstM);
            pingpongFlag = 1 - pingpongFlag;
        }
        if (nIdx > 0) {
            int64_t gmOffsetOTmp = layoutOTmp.GetOffset(MatrixCoord(subM, 0));
            auto gOTmpInt32ThisSubBlock = gOTmpInt32[gmOffsetOTmp];
            AmlaAutoadd(gOTmpInt32, subM, embed, gSPingPongFlag);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxAutoAddReady);
    }

private:
    float tor;
    uint32_t pingpongFlag = 0;
    uint32_t kvSplitCoreNum = 1;

    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor[2];
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> glUbTensor[2];

    // new add
    AscendC::LocalTensor<float> hnUbTensor;
    AscendC::LocalTensor<float> dnUbTensor;
    AscendC::LocalTensor<float> gnUbTensor;
    AscendC::LocalTensor<float> s32UbTensor;
    AscendC::LocalTensor<ElementOutput> s16UbTensorTmp;
    AscendC::LocalTensor<float> s16UbTensor;
    AscendC::LocalTensor<float> hcUbTensor;
    AscendC::LocalTensor<float> gcUbTensor;
    AscendC::LocalTensor<float> epsUbTensor;
    AscendC::LocalTensor<int32_t> NUbTensor;
    AscendC::LocalTensor<int32_t> NBrcbUbTensor;
    Arch::CrossCoreFlag preAutoAddReady{PRE_AUTOADD_READY_ID};
    Arch::CrossCoreFlag softmaxAutoAddReady{SM_AUTOADD_READY_ID};

    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};

};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_AMLA_TP1_SOFTMAX_HPP
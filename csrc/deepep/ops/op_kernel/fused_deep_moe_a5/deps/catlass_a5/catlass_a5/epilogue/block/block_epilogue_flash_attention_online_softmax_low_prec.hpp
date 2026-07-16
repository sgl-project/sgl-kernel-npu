/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_LOW_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_LOW_PREC_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class LayoutS_,
    class MaskType_,
    class TileCopy_>
class BlockEpilogue<
    EpilogueFAOnlineSoftmax,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>,
    MaskType_,
    TileCopy_>
{
public:
    using DispatchPolicy = EpilogueFAOnlineSoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = half;
    using ElementMask = typename MaskType_::Element;
    using TileCopy = TileCopy_;
    
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;
    using LayoutMask = typename MaskType_::Layout;

    using CopyGmToUbMask = typename TileCopy::CopyGmToUbMask;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr ElementInput MIN_VALUE = -65504.0f;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx();

        scaleValue = static_cast<ElementInput>(scaleValue_);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LL_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline
    void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }
    
    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag)
    {
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ ElementInput*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ ElementInput*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
        ComputeScaleAndMax<ElementInput>(
            sAddr, nowMaxAddr, m, nLoops, tailN, nPadding, scaleValue, nRound);
        if (!isFirstKvSTile) {
            UpdateMax<ElementInput>(nowMaxAddr, lastMaxAddr, mLoops, tailM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        ComputeExpSubSum16<ElementOutput, ElementInput>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, nLoops, tailN, blockStride, nRound);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
        mLoops = AscendC::CeilDivision(m, 64) - 1;
        tailM = (m - 1) % 64 + 1;
        if (isFirstKvSTile) {
            CastExpSumAndExpMax<ElementInput>(
                lastSumAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        } else {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag, bool enableDn)
    {
        //待补充
        return;
    }

    template <class TensorP, class TensorMask>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, TensorMask &gmMaskTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag,
         uint32_t triUp,  uint32_t triDown, uint32_t globalWindowSize,  uint32_t localWindowSize,
         uint32_t kvSStartIdx, uint32_t kvSEndIdx, uint32_t maskType)
    {   
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        // calc mask shift in gm
        uint32_t gmOffsetMaskRow = 0;
        uint32_t gmOffsetMaskColumn = 0;
        uint32_t maskColumn = 0;
        uint32_t addMaskUbOffset = 0;
        if (maskType == 1) {
            if (triUp >= kvSStartIdx) {
                gmOffsetMaskRow = triUp - kvSStartIdx;
                gmOffsetMaskColumn = 0;
                maskColumn = kvSEndIdx - kvSStartIdx;
            } else {
                gmOffsetMaskRow = 0;
                gmOffsetMaskColumn = kvSStartIdx - triUp;
                maskColumn = n;
                addMaskUbOffset = 0;
            }
        } else {
            if (maskType == 2) {
                gmOffsetMaskRow = triDown - actualBlockShape.m();
            } else if (maskType == 3) {
                gmOffsetMaskRow = globalWindowSize + localWindowSize - 1;
            }
            gmOffsetMaskColumn = kvSStartIdx;
            maskColumn = kvSEndIdx - kvSStartIdx;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, 128);
        auto gMaskTensorTlaTile = GetTile(gmMaskTensorTla,
                        tla::MakeCoord(gmOffsetMaskRow + subBlockIdx_ * mCopyOffset, gmOffsetMaskColumn), tla::MakeShape(m, maskColumnRound));
        CopyGmToUbMask copyGmToUbMask;
        auto ubMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(m, maskColumnRound);
        auto ubMaskTensorTla = tla::MakeTensor(maskUbTensor, ubMaskLayoutTla, Arch::PositionUB{});
        auto ubMaskTensorTlaTile = GetTile(ubMaskTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, maskColumnRound));

        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ ElementInput*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ ElementInput*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementMask *maskUbAddr = (__ubuf__ ElementMask *)maskUbTensor.GetPhyAddr();
        
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        copyGmToUbMask(ubMaskTensorTlaTile, gMaskTensorTlaTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        ComputeScaleAndMaxMask<ElementInput>(
            sAddr, nowMaxAddr, maskUbAddr, m, nLoops, tailN, nPadding, scaleValue, nRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        if (!isFirstKvSTile) {
            UpdateMax<ElementInput>(nowMaxAddr, lastMaxAddr, mLoops, tailM);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        ComputeExpSubSum16<ElementOutput, ElementInput>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, nLoops, tailN, blockStride, nRound);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
        mLoops = AscendC::CeilDivision(m, 64) - 1;
        tailM = (m - 1) % 64 + 1;
        if (isFirstKvSTile) {
            CastExpSumAndExpMax<ElementInput>(
                lastSumAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        } else {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<ElementMask> maskUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    template <typename ElementS>
    __simd_vf__ inline void ComputeScaleAndMax(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementS> minVreg;
        RegTensor<ElementS> srcVreg;
        RegTensor<ElementS> maxSrcVreg;
        RegTensor<ElementS> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementS>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            Duplicate(maxSrcVreg, MIN_VALUE);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(srcVreg, srcUb + i * S2BaseSize + j * HALF_REP_SIZE);
                Muls(srcVreg, srcVreg, dScale, pregFull);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(
                    srcUb + i * S2BaseSize + j * HALF_REP_SIZE, srcVreg, pregFull);
                Max(maxSrcVreg, maxSrcVreg, srcVreg, pregFull);
            }
            LoadAlign(srcVreg, srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE);
            Muls(srcVreg, srcVreg, dScale, pregFull);

            
            Select(srcVreg, srcVreg, minVreg, pregTailN);
            StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE, srcVreg, pregTailN);
            Max(maxSrcVreg, maxSrcVreg, srcVreg, pregFull);

            ReduceMax(maxTmpVreg, maxSrcVreg, pregFull);
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <typename ElementS>
    __simd_vf__ inline void ComputeScaleAndMaxMask(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementMask *maskUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementS> minVreg;
        RegTensor<ElementS> srcVreg;
        RegTensor<ElementS> maxSrcVreg;
        RegTensor<ElementS> maxTmpVreg;
        RegTensor<uint8_t> maskVreg;
        RegTensor<half> maskVregb16;
        RegTensor<ElementS> srcVreg_mask;

        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementS>(tailN);
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            Duplicate(maxSrcVreg, MIN_VALUE);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(srcVreg, srcUb + i * S2BaseSize + j * HALF_REP_SIZE);
                LoadAlign<ElementMask, LoadDist::DIST_UNPACK_B8>(maskVreg, maskUb + i * 128 + j * HALF_REP_SIZE);
                Muls(srcVreg, srcVreg, dScale, pregFull);
                Cast<half, ElementMask, castTraitZero>(maskVregb16, maskVreg, preg_all_b16);
                Compares(pregCompare, maskVregb16, static_cast<half>(0), preg_all_b16);
                Select(srcVreg_mask, srcVreg, minVreg, pregCompare);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(
                    srcUb + i * S2BaseSize + j * HALF_REP_SIZE, srcVreg_mask, pregFull);
                Max(maxSrcVreg, maxSrcVreg, srcVreg, pregFull);
            }
            LoadAlign(srcVreg, srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            LoadAlign<ElementMask, LoadDist::DIST_UNPACK_B8>(maskVreg, maskUb + i * 128 + nLoops * HALF_REP_SIZE);
            Cast<half, ElementMask, castTraitZero>(maskVregb16, maskVreg, preg_all_b16);
            Compares(pregCompare, maskVregb16, static_cast<half>(0), preg_all_b16);
            Select(srcVreg_mask, srcVreg, minVreg, pregCompare);
            
            Select(srcVreg, srcVreg_mask, minVreg, pregTailN);
            StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE, srcVreg, pregTailN);
            Max(maxSrcVreg, maxSrcVreg, srcVreg, pregFull);

            ReduceMax(maxTmpVreg, maxSrcVreg, pregFull);
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }


    template <typename ElementS>
    __simd_vf__ inline void UpdateMax(
        __ubuf__ ElementS *nowMaxUb, __ubuf__ float *lastMaxUb, uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementS> nowMaxVreg;
        RegTensor<float> lastMaxFloatVreg0;
        RegTensor<float> lastMaxFloatVreg1;
        RegTensor<ElementS> maxVreg;
        RegTensor<ElementS> lastMaxVreg;
        RegTensor<ElementS> lastMaxVreg0;
        RegTensor<ElementS> lastMaxVreg1;

        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<ElementS>(tailM);
        for (uint16_t i = 0; i < mLoops; ++i) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    lastMaxFloatVreg0, lastMaxFloatVreg1, lastMaxUb + i * HALF_REP_SIZE);
            Cast<ElementS, float, castTraitZero>(lastMaxVreg0, lastMaxFloatVreg0, pregFloatFull);
            Cast<ElementS, float, castTraitOne>(lastMaxVreg1, lastMaxFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)lastMaxVreg,
                (RegTensor<uint16_t>&)lastMaxVreg0, (RegTensor<uint16_t>&)lastMaxVreg1,
                pregFull);
            LoadAlign(nowMaxVreg, nowMaxUb + i * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg, pregFull);
            StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(nowMaxUb + i * HALF_REP_SIZE, maxVreg, pregFull);
        }
        if (tailM > 64) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    lastMaxFloatVreg0, lastMaxFloatVreg1, lastMaxUb + mLoops * HALF_REP_SIZE);
            Cast<ElementS, float, castTraitZero>(lastMaxVreg0, lastMaxFloatVreg0, pregFloatFull);
            Cast<ElementS, float, castTraitOne>(lastMaxVreg1, lastMaxFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)lastMaxVreg,
               (RegTensor<uint16_t>&)lastMaxVreg0, (RegTensor<uint16_t>&)lastMaxVreg1,
               pregFull);
            LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg, pregTailM);
            StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(nowMaxUb + mLoops * HALF_REP_SIZE, maxVreg, pregTailM);
        } else {
            LoadAlign(lastMaxFloatVreg0, lastMaxUb + mLoops * HALF_REP_SIZE);
            Cast<ElementS, float, castTraitZero>(lastMaxVreg, lastMaxFloatVreg0, pregFloatFull);
            DeInterleave(lastMaxVreg0, lastMaxVreg1, lastMaxVreg, lastMaxVreg);
            LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg0, pregTailM);
            StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(nowMaxUb + mLoops * HALF_REP_SIZE, maxVreg, pregTailM);
        }
    }

    template <typename ElementP, typename ElementS>
    __simd_vf__ inline void ComputeExpSubSum16(__ubuf__ ElementP *expUb, __ubuf__ ElementS *srcUb,
        __ubuf__ ElementS *nowMaxUb, __ubuf__ ElementS *expSumUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementS> expVreg;
        RegTensor<ElementS> expSumVreg;
        RegTensor<ElementS> maxVreg;

        RegTensor<ElementP> expDstVreg;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementS>(tailN);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementS, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(expVreg, srcUb + i * S2BaseSize + j * HALF_REP_SIZE);
                Sub(expDstVreg, expVreg, maxVreg, pregFull);
                Exp(expDstVreg, expDstVreg, pregFull);
                Add(expSumVreg, expSumVreg, expDstVreg, pregFull);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY>(
                    expUb + i * ELE_NUM_PER_C0 + j * blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE,
                    expDstVreg, blockStride, pregFull);
            }
            
            LoadAlign(expVreg, srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE);
            Sub(expDstVreg, expVreg, maxVreg, pregTailN);
            Exp(expDstVreg, expDstVreg, pregTailN);
            Add<ElementP, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstVreg, pregTailN);
            StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0 + nLoops * blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE,
                expDstVreg, blockStride, pregTailN);

            ReduceSum(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <typename ElementS>
    __simd_vf__ inline void CastExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *maxUb,
        __ubuf__ ElementS *expSumUb, __ubuf__ ElementS *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementS> nowMaxVreg0;
        RegTensor<ElementS> nowMaxVreg1;
        RegTensor<ElementS> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<ElementS> brcExpSumVreg0;
        RegTensor<ElementS> brcExpSumVreg1;
        RegTensor<ElementS> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementS, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementS, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementS, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregFull);
        }
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementS, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementS, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregTailM);
    }

    template <typename ElementS>
    __simd_vf__ inline void UpdateExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *expMaxUb,
        __ubuf__ float *maxUb, __ubuf__ ElementS *expSumUb, __ubuf__ ElementS *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementS> nowMaxVreg0;
        RegTensor<ElementS> nowMaxVreg1;
        RegTensor<ElementS> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<ElementS> brcExpSumVreg0;
        RegTensor<ElementS> brcExpSumVreg1;
        RegTensor<ElementS> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        RegTensor<float> updateExpSumVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementS, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(lastMaxVreg, maxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementS, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + i * FLOAT_REP_SIZE, expMaxVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(lastExpSumVreg, sumUb + i * FLOAT_REP_SIZE);
            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementS, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, updateExpSumVreg, pregFull);
        }
        LoadAlign(lastMaxVreg, maxUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementS, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + mLoops * FLOAT_REP_SIZE, expMaxVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementS, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregTailM);
        Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, updateExpSumVreg, pregTailM);
    }
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_LOW_PREC_HPP
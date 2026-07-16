/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_SWIGLU_AND_MX_QUANT_HPP
#define CATLASS_EPILOGUE_TILE_TILE_SWIGLU_AND_MX_QUANT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Tile {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using namespace AscendC::MicroAPI;
#endif
template <
    class ArchTag_,
    class ActType_,
    class GateType_,
    class CalcType_,
    class GluResType_,
    class QuantOutType_,
    class QuantScaleType_,
    uint16_t EMAX_
>
struct TileSwigluAndMxQuant {
    using ArchTag = ArchTag_;
    using ActType = ActType_;
    using GateType = GateType_;
    using CalcType = CalcType_;
    using GluResType = GluResType_;
    using QuantOutType = QuantOutType_;
    using QuantScaleType = QuantScaleType_;
    static constexpr uint16_t EMAX = EMAX_;
    static constexpr uint32_t MX_BLOCK_SIZE = 32;
    static constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    static constexpr int16_t SHR_NUM_FOR_BF16 = 7;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    static constexpr uint32_t VF_BLOCK_BYTES = 256;
    static constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    CATLASS_DEVICE
    __simd_vf__ void operator()(
        __ubuf__ int8_t* quantOut,
        __ubuf__ QuantScaleType* quantScaleOut,
        __ubuf__ GluResType* gluRes,
        __ubuf__ uint16_t* maxExp,
        __ubuf__ uint16_t* halfScale,
        __ubuf__ ActType* actData,
        __ubuf__ GateType* gateData,
        uint16_t mSize,
        uint32_t nSize,
        uint32_t nAligned)
    {
        constexpr uint32_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(CalcType);
        constexpr uint32_t eleNumPerVfGluRes = VF_BLOCK_BYTES / sizeof(GluResType);
        uint32_t OneRowRepeatTimes = CeilDiv(nSize, sizePerRepeat);
        uint32_t nLoopCntGluRes = CeilDiv(nSize, eleNumPerVfGluRes);
        uint32_t nScaleLoopCnt = CeilDiv(nSize, MX_BLOCK_SIZE);
        static constexpr DivSpecificMode DIV_MODE = {
            MaskMergeMode::ZEROING,
            true,
        };

        constexpr static CastTrait ctCalcToGluRes = {
            RegLayout::ZERO, SatMode::NO_SAT,
            MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        float float_one = 1.0;
        uint32_t nSrcUbAligned = RoundUp(nSize, AscendC::ONE_BLK_SIZE / sizeof(CalcType));
        uint32_t nDstUbAligned = RoundUp(nSize, AscendC::ONE_BLK_SIZE);
        //swiglu
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            for (uint32_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                uint32_t elementNum = Min(sizePerRepeat, nSize - vfBlockIdx * sizePerRepeat);
                MaskReg maskN = UpdateMask<CalcType>(elementNum);

                RegTensor<float> actCalcReg;
                RegTensor<float> gateCalcReg;
                RegTensor<float> negActReg;
                RegTensor<float> expNegActReg;
                RegTensor<float> onePlusExpReg;
                RegTensor<float> swishOutputReg;
                RegTensor<float> gluCalcReg;
                RegTensor<GluResType> gluGluResReg;

                uint32_t srcOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
                uint32_t dstOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;

                DataCopy(actCalcReg, actData + srcOffset);

                Muls(negActReg, actCalcReg, -float_one, maskN);
                Exp(expNegActReg, negActReg, maskN);
                Adds(onePlusExpReg, expNegActReg, float_one, maskN);
                Div<float, &DIV_MODE>(swishOutputReg, actCalcReg, onePlusExpReg, maskN);
                DataCopy(gateCalcReg, gateData + srcOffset);
                Mul(gluCalcReg, swishOutputReg, gateCalcReg, maskN);

                Cast<GluResType, CalcType, ctCalcToGluRes>(gluGluResReg, gluCalcReg, maskN);
                DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gluRes + dstOffset, gluGluResReg, maskN);
            }
        }


        uint32_t totalDataCount = mSize * nAligned;
        uint32_t totalScaleCount = mSize * nScaleLoopCnt;
        uint16_t vlForHalfNumber = eleNumPerVfGluRes;
        uint16_t elementAfterReduce = VF_BLOCK_BYTES / MX_BLOCK_SIZE;
        uint16_t loopDataNum = CeilDiv(totalDataCount, static_cast<uint32_t>(vlForHalfNumber) * 2);
        uint16_t loopScaleNum = CeilDiv(totalScaleCount, static_cast<uint32_t>(vlForHalfNumber));
        
        ComputeMaxExp(gluRes, maxExp, totalDataCount, loopDataNum, vlForHalfNumber);

        __ubuf__ uint16_t* scaleWriteAddr = reinterpret_cast<__ubuf__ uint16_t*>(quantScaleOut);
        ComputeScale(maxExp, scaleWriteAddr, halfScale, totalScaleCount, loopScaleNum, vlForHalfNumber);

        QuantToFp8(gluRes, halfScale, quantOut, totalDataCount, loopDataNum, vlForHalfNumber);
    }

    CATLASS_DEVICE
    __simd_callee__ void ComputeMaxExp( __ubuf__ bfloat16_t *gluRes, __ubuf__ uint16_t *maxExp, uint32_t totalDataCount, uint16_t loopDataNum, uint16_t vlForHalfNumber)
    {
        __VEC_SCOPE__
        {
            uint16_t elementAfterReduce = VF_BLOCK_BYTES / MX_BLOCK_SIZE;
            RegTensor<bfloat16_t> vdExp0;
            RegTensor<bfloat16_t> vdExp1;
            RegTensor<bfloat16_t> vdExp0BF16;
            RegTensor<bfloat16_t> vdExp1BF16;
            RegTensor<uint16_t> vdExpExtract0;
            RegTensor<uint16_t> vdExpExtract1;

            RegTensor<uint16_t> expMaskBF16;
            Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

            RegTensor<uint16_t> vdMaxExp;
            MaskReg scaleMask1;
            MaskReg scaleMask2;
            UnalignReg uRegMaxExp;
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
            for (uint16_t loopIdx = 0; loopIdx < loopDataNum; loopIdx++) {
                scaleMask1 = UpdateMask<bfloat16_t>(totalDataCount);
                scaleMask2 = UpdateMask<bfloat16_t>(totalDataCount);

                DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE,
                                            LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, gluRes, vlForHalfNumber * 2);

                And(vdExpExtract0, (RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                And(vdExpExtract1, (RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);

                Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
                ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

                DataCopyUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    maxExp, vdMaxExp, uRegMaxExp, elementAfterReduce);
            }
            DataCopyUnAlignPost(maxExp, uRegMaxExp, 0);
        }
        return;
    }

    CATLASS_DEVICE
    __simd_callee__ void ComputeScale(__ubuf__ uint16_t *maxExp, __ubuf__ uint16_t *scaleWriteAddr, __ubuf__ uint16_t *halfScale,
    uint32_t totalScaleCount, uint16_t loopScaleNum, uint16_t vlForHalfNumber)
    {
        __VEC_SCOPE__
        {
            RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScaleVal, fp8NanReg;
            Duplicate(expMask, MAX_EXP_FOR_BF16);
            RegTensor<uint16_t> vdMaxExp;


            MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;

            RegTensor<uint16_t> maxExpValue, zeroReg, nanReg, specialExpReg;
            
            Duplicate(maxExpValue, EMAX);
            Duplicate(scaleBias, BF16_EXP_BIAS);
            Duplicate(fp8NanReg, MAX_EXP_FOR_FP8);
            Duplicate(zeroReg, static_cast<uint16_t>(0));
            Duplicate(nanReg, NAN_CUSTOMIZATION);
            Duplicate(specialExpReg, SPECIAL_EXP_THRESHOLD);

            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < loopScaleNum; i++) {
                preMaskScale = UpdateMask<uint16_t>(totalScaleCount);

                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    vdMaxExp, maxExp, vlForHalfNumber);

                Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
                Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroReg, preMaskScale);
                Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
                Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

                Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
                ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
                Select<uint16_t>(scaleValue, scaleValue, fp8NanReg, cmpResult);
                Select<uint16_t>(scaleValue, scaleValue, zeroReg, zeroMask);

                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE,
                                            StoreDist::DIST_PACK_B16>(
                    scaleWriteAddr, scaleValue, vlForHalfNumber >> 1, preMaskScale);

                Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
                Sub(halfScaleVal, scaleBias, sharedExp, preMaskScale);
                Select<uint16_t>(halfScaleVal, halfScaleVal, nanReg, cmpResult);
                Select<uint16_t>(halfScaleVal, halfScaleVal, zeroReg, zeroMask);
                Select<uint16_t>(halfScaleVal, specialExpReg, halfScaleVal, specialDataMask);

                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    halfScale, halfScaleVal, vlForHalfNumber, preMaskScale);
            }
        }
        return;
    }

    CATLASS_DEVICE
    __simd_callee__ void QuantToFp8(__ubuf__ bfloat16_t *gluRes, __ubuf__ uint16_t *halfScale, __ubuf__ int8_t *quantOut,
    uint32_t totalDataCount, uint16_t loopDataNum, uint16_t vlForHalfNumber)
    {
        uint32_t totalDataCount2 = totalDataCount * 2;
        uint16_t elementAfterReduce = VF_BLOCK_BYTES / MX_BLOCK_SIZE;
        __VEC_SCOPE__
        {
            MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
            MaskReg maskAll =
                CreateMask<uint16_t, MaskPattern::ALL>();
            RegTensor<uint16_t> halfScaleForMul;
            RegTensor<float> floatScaleForMul;
            RegTensor<bfloat16_t> vdExp0, vdExp1, vdExp0Convert, vdExp1Convert;
            RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
            RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
            RegTensor<QuantOutType> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

            static constexpr CastTrait castTraitZero = {
                RegLayout::ZERO, SatMode::UNKNOWN,
                MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr CastTrait castTraitOne = {
                RegLayout::ONE, SatMode::UNKNOWN,
                MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr CastTrait castTrait32to8 = {
                RegLayout::ZERO, SatMode::SAT,
                MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < loopDataNum; i++) {
                dataMask1 = UpdateMask<bfloat16_t>(totalDataCount);
                dataMask2 = UpdateMask<bfloat16_t>(totalDataCount);
                dataMask3 = UpdateMask<bfloat16_t>(totalDataCount2);
                dataMask4 = UpdateMask<bfloat16_t>(totalDataCount2);
                DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE,
                                            LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, gluRes,
                    vlForHalfNumber * 2);
                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE,
                                            LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScale,
                                                                                    elementAfterReduce);

                Mul(vdExp0, vdExp0, (RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                Mul(vdExp1, vdExp1, (RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Cast<float, bfloat16_t, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                Cast<float, bfloat16_t, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                Cast<QuantOutType, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                Cast<QuantOutType, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                Cast<float, bfloat16_t, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                Cast<float, bfloat16_t, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                Cast<QuantOutType, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                Cast<QuantOutType, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
                DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE,
                                            StoreDist::DIST_PACK4_B32>(
                    quantOut, (RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
                DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE,
                                            StoreDist::DIST_PACK4_B32>(
                    quantOut, (RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
                DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE,
                                            StoreDist::DIST_PACK4_B32>(
                    quantOut, (RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
                DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE,
                                            StoreDist::DIST_PACK4_B32>(
                    quantOut, (RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
            }
        }
        return;
    }
#endif
};

}  // namespace Catlass::Epilogue::Tile

#endif  // CATLASS_EPILOGUE_TILE_TILE_SWIGLU_AND_MX_QUANT_HPP
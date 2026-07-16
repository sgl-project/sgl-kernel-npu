/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_PERBLOCK_QUANT_HPP
#define CATLASS_EPILOGUE_TILE_TILE_PERBLOCK_QUANT_HPP

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
    class ElementSrc_,
    class ElementDst_,
    class ElementScale_
>
struct TilePerBlockQuant {
    using ArchTag = ArchTag_;
    using ElementSrc = ElementSrc_;
    using ElementDst = ElementDst_;
    using ElementScale = ElementScale_;

    static_assert(
        std::is_same_v<ArchTag, Arch::Ascend950> && (std::is_same_v<ElementSrc, bfloat16_t>) &&
        (std::is_same_v<ElementDst, float8_e4m3_t>) &&
        (std::is_same_v<ElementScale, float>),
        "The element type template parameters of TilePerBlockQuant are wrong"
    );

    CATLASS_DEVICE
    TilePerBlockQuant() {}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    static constexpr int32_t VL_B16 = static_cast<int32_t>(BYTE_PER_VECTOR_FRACTAL / sizeof(uint16_t));
    static constexpr int32_t VL_B32 = static_cast<int32_t>(BYTE_PER_VECTOR_FRACTAL / sizeof(uint32_t));
    static constexpr float FP8_E4M3FN_MAX = 448;

    __simd_vf__ void perBlockScaleQuant(
        __ubuf__ ElementSrc* srcUb, __ubuf__ ElementDst* dstUb, __ubuf__ ElementScale* scaleUb, uint16_t nLoops
    )
    {
        static constexpr CastTrait ctFp322Fp8 = {
            RegLayout::ZERO, SatMode::NO_SAT,
            MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT
        };
        static constexpr CastTrait ctFp322Fp16 = {
            RegLayout::ZERO, SatMode::NO_SAT,
            MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT
        };
        static constexpr CastTrait ctHalf2Fp32Zero = {
            RegLayout::ZERO, SatMode::UNKNOWN,
            MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN
        };
        static constexpr CastTrait ctHalf2Fp32One = {
            RegLayout::ONE, SatMode::UNKNOWN,
            MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN
        };
        RegTensor<uint16_t> absMaskReg;
        RegTensor<uint16_t> srcReg;
        RegTensor<float> srcRegZero;
        RegTensor<float> srcRegOne;
        RegTensor<float> srcMaxReg;
        RegTensor<float> maxReg;
        RegTensor<float> mulScaleReg;
        RegTensor<float> scaleReg;
        RegTensor<float8_e4m3_t> outReg;
        MaskReg pregFullB32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg pregFullB16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg pregFullB8 = CreateMask<uint8_t, MaskPattern::ALL>();

        Duplicate(absMaskReg, static_cast<uint16_t>(0x7fff), pregFullB16);
        Duplicate((RegTensor<uint16_t>&)srcMaxReg, static_cast<uint16_t>(0), pregFullB16);
        for (uint16_t i = 0; i < nLoops; i++) {
            LoadAlign(srcReg, (__ubuf__ uint16_t*)srcUb + i * VL_B16);
            And(srcReg, srcReg, absMaskReg, pregFullB16);
            Max((RegTensor<uint16_t>&)srcMaxReg, (RegTensor<uint16_t>&)srcMaxReg, (RegTensor<uint16_t>&)srcReg, pregFullB16);
        }
        Reduce<ReduceType::MAX>((RegTensor<uint16_t>&)srcMaxReg, (RegTensor<uint16_t>&)srcMaxReg, pregFullB16);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(srcMaxReg, (RegTensor<bfloat16_t>&)srcMaxReg, pregFullB16);
        Duplicate(srcMaxReg, srcMaxReg, pregFullB32);
        Duplicate(maxReg, FP8_E4M3FN_MAX, pregFullB32);
        Div(mulScaleReg, maxReg, srcMaxReg, pregFullB32);
        Div(scaleReg, srcMaxReg, maxReg, pregFullB32);
        for (uint16_t i = 0; i < nLoops; i++) {
            LoadAlign(srcReg, (__ubuf__ uint16_t*)srcUb + i * VL_B16);
            Cast<float, bfloat16_t, ctHalf2Fp32Zero>(srcRegZero, (RegTensor<bfloat16_t>&)srcReg, pregFullB16);
            Cast<float, bfloat16_t, ctHalf2Fp32One>(srcRegOne, (RegTensor<bfloat16_t>&)srcReg, pregFullB16);
            Mul(srcRegZero, srcRegZero, mulScaleReg, pregFullB32);
            Mul(srcRegOne, srcRegOne, mulScaleReg, pregFullB32);
            Interleave(srcRegZero, srcRegOne, srcRegZero, srcRegOne);
            Cast<float8_e4m3_t, float, ctFp322Fp8>((RegTensor<float8_e4m3_t>&)srcRegZero, srcRegZero, pregFullB32);
            Cast<float8_e4m3_t, float, ctFp322Fp8>((RegTensor<float8_e4m3_t>&)srcRegOne, srcRegOne, pregFullB32);
            StoreAlign<uint8_t, StoreDist::DIST_PACK4_B32>(reinterpret_cast<__ubuf__ uint8_t*>(dstUb) + i * VL_B16, (RegTensor<uint8_t>&)srcRegZero, pregFullB8);
            StoreAlign<uint8_t, StoreDist::DIST_PACK4_B32>(reinterpret_cast<__ubuf__ uint8_t*>(dstUb) + i * VL_B16 + VL_B32, (RegTensor<uint8_t>&)srcRegOne, pregFullB8);
        }
        StoreAlign<uint32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(reinterpret_cast<__ubuf__ uint32_t*>(scaleUb), (RegTensor<uint32_t>&)scaleReg, pregFullB32);
    }

    template <class TensorSrc, class TensorDst, class TensorScale>
    CATLASS_DEVICE
    void operator()(
        TensorSrc const &ubIn,
        TensorDst const &ubOut,
        TensorScale const &ubScale,
        uint16_t nLoops
    )
    {
        __ubuf__ ElementSrc* srcUbAddr = (__ubuf__ ElementSrc*)ubIn.data().GetPhyAddr();
        __ubuf__ ElementDst* dstUbAddr = (__ubuf__ ElementDst*)ubOut.data().GetPhyAddr();
        __ubuf__ ElementScale* scaleUbAddr = (__ubuf__ ElementScale*)ubScale.data().GetPhyAddr();
        perBlockScaleQuant(srcUbAddr, dstUbAddr, scaleUbAddr, nLoops);
    }
#endif
};

} // namespace Catlass::Epilogue::Tile

#endif //CATLASS_EPILOGUE_TILE_TILE_PERBLOCK_QUANT_HPP

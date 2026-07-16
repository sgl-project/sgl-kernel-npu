/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_PERTOKEN_DEQUANT_HPP
#define CATLASS_EPILOGUE_TILE_TILE_PERTOKEN_DEQUANT_HPP

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
    class ElementScale_,
    class ElementPerToken_,
    class ElementDst_,
    class TileShape_
>
struct TilePerTokenDequant {
    using ArchTag = ArchTag_;
    using ElementSrc = ElementSrc_;
    using ElementScale = ElementScale_;
    using ElementPerToken = ElementPerToken_;
    using ElementDst = ElementDst_;
    using TileShape = TileShape_;

    static_assert(
        std::is_same_v<ArchTag, Arch::Ascend950> && std::is_same_v<ElementSrc, int32_t> && 
        (std::is_same_v<ElementDst, half> || std::is_same_v<ElementDst, bfloat16_t> || std::is_same_v<ElementDst, float>) &&
        (std::is_same_v<ElementScale, half> || std::is_same_v<ElementScale, bfloat16_t> || std::is_same_v<ElementScale, float>) &&
        (std::is_same_v<ElementPerToken, half> || std::is_same_v<ElementPerToken, bfloat16_t> || std::is_same_v<ElementPerToken, float>),
        "The element type template parameters of TilePerTokenDequant are wrong"
    );

    CATLASS_DEVICE
    TilePerTokenDequant() {}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    static constexpr int16_t N_BASE_SIZE = static_cast<int16_t>(TileShape::COLUMN);
    static constexpr int16_t FLOAT_REP_SIZE = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(float));

    static constexpr CastTrait ctInt322Fp32 = {
        RegLayout::UNKNOWN, SatMode::UNKNOWN,
        MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT
    };
    static constexpr CastTrait ctFp322Half = {
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

    template <class T, LoadDist Dist = LoadDist::DIST_NORM>
    __simd_callee__ inline void LoadIn(RegTensor<float>& dstVreg, __ubuf__ T*  srcUb, MaskReg& mask, MaskReg& maskB16){
        if constexpr (AscendC::IsSameType<T, float>::value) {
            LoadAlign<T, Dist>(dstVreg, srcUb);
        } else if constexpr(AscendC::IsSameType<T, int32_t>::value) {
            RegTensor<T> srcVreg;
            LoadAlign<T, Dist>(srcVreg, srcUb);
            Cast<float, T, ctInt322Fp32>(dstVreg, srcVreg, mask);
        } else {
            RegTensor<T> srcVreg;
            RegTensor<float> srcVreg0, srcVreg1;
            LoadAlign<T, Dist>(srcVreg, srcUb);
            Cast<float, T, ctHalf2Fp32Zero>(srcVreg0, srcVreg, mask);
            Cast<float, T, ctHalf2Fp32One>(srcVreg1, srcVreg, maskB16);
            Interleave(dstVreg, srcVreg1, srcVreg0, srcVreg1);
        }
    }

    template <class T>
    __simd_callee__ inline void StoreOut(__ubuf__ T* dstUb, RegTensor<float>& srcVreg, MaskReg& mask){
        if constexpr (AscendC::IsSameType<T, float>::value) {
            StoreAlign<T, StoreDist::DIST_NORM_B32>(dstUb, srcVreg, mask);
        } else {
            RegTensor<T> dstVreg;
            Cast<T, float, ctFp322Half>(dstVreg, srcVreg, mask);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dstUb, dstVreg, mask);
        }
    }

    __simd_vf__ void perTokenScaleDequant(
        __ubuf__ ElementDst* dstUb, __ubuf__ ElementSrc* srcUb, __ubuf__ ElementScale* scaleUb,
        __ubuf__ ElementPerToken* perTokenUb, uint16_t m, uint16_t nLoops, uint32_t tailN
    )
    {
        RegTensor<float> perTokenVreg;
        RegTensor<float> scaleVreg;
        RegTensor<float> srcVreg;
        RegTensor<float> mulScaleVreg;
        RegTensor<float> dstVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFullB16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
        uint32_t tailNB16 = tailN;
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg pregTailNB16 = UpdateMask<bfloat16_t>(tailNB16);
        
        for (uint16_t i = 0; i < m; i++) {
            if constexpr (AscendC::IsSameType<ElementPerToken, float>::value) {
                LoadIn<ElementPerToken, LoadDist::DIST_BRC_B32>(perTokenVreg, perTokenUb + i, pregFull, pregFullB16);
            } else {
                LoadIn<ElementPerToken, LoadDist::DIST_BRC_B16>(perTokenVreg, perTokenUb + i, pregFull, pregFullB16);
            }
            for (uint16_t j = 0; j < nLoops; j++) {
                LoadIn<ElementSrc>(srcVreg, srcUb + i * N_BASE_SIZE + j * FLOAT_REP_SIZE, pregFull, pregFullB16);
                LoadIn<ElementScale>(scaleVreg, scaleUb + j * FLOAT_REP_SIZE, pregFull, pregFullB16);
                Mul(mulScaleVreg, srcVreg, scaleVreg, pregFull);
                Mul(dstVreg, mulScaleVreg, perTokenVreg, pregFull);
                StoreOut<ElementDst>(dstUb + i * N_BASE_SIZE + j * FLOAT_REP_SIZE, dstVreg, pregFull);
            }
            LoadIn<ElementSrc>(srcVreg, srcUb + i * N_BASE_SIZE + nLoops * FLOAT_REP_SIZE, pregTailN, pregTailNB16);
            LoadIn<ElementScale>(scaleVreg, scaleUb + nLoops * FLOAT_REP_SIZE, pregTailN, pregTailNB16);
            Mul(mulScaleVreg, srcVreg, scaleVreg, pregTailN);
            Mul(dstVreg, mulScaleVreg, perTokenVreg, pregTailN);
            StoreOut<ElementDst>(dstUb + i * N_BASE_SIZE + nLoops * FLOAT_REP_SIZE, dstVreg, pregTailN);
        }
    }

    template <class TensorDst, class TensorSrc, class TensorScale, class TensorPerToken>
    CATLASS_DEVICE
    void operator()(
        TensorDst const &ubOut,
        TensorSrc const &ubIn,
        TensorScale const &ubScale,
        TensorPerToken const &ubPerToken
    )
    {
        static_assert(tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                      tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                      tla::detail::isVector<typename TensorScale::Layout>::value &&
                      tla::detail::isVector<typename TensorPerToken::Layout>::value &&
                      TensorSrc::position == AscendC::TPosition::VECCALC &&
                      TensorDst::position == AscendC::TPosition::VECCALC &&
                      TensorScale::position == AscendC::TPosition::VECCALC &&
                      TensorPerToken::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be UB and RowMajor, "
            "TensorDst must be UB and RowMajor, TensorScale must be UB and VectorLayout, "
            "TensorPerToken must be UB and VectorLayout."
        );
        uint32_t m = tla::get<0>(ubIn.shape());
        uint32_t n = tla::get<1>(ubIn.shape());
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementSrc));
        uint16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint16_t tailN = n - nLoops * vlSize;
        __ubuf__ ElementDst* dstUbAddr = (__ubuf__ ElementDst*)ubOut.data().GetPhyAddr();
        __ubuf__ ElementSrc* srcUbAddr = (__ubuf__ ElementSrc*)ubIn.data().GetPhyAddr();
        __ubuf__ ElementScale* scaleUbAddr = (__ubuf__ ElementScale*)ubScale.data().GetPhyAddr();
        __ubuf__ ElementPerToken* perTokenUbAddr = (__ubuf__ ElementPerToken*)ubPerToken.data().GetPhyAddr();
        perTokenScaleDequant(dstUbAddr, srcUbAddr, scaleUbAddr, perTokenUbAddr, m, nLoops, tailN);
    }
#endif
};

} // namespace Catlass::Epilogue::Tile

#endif //CATLASS_EPILOGUE_TILE_TILE_PERTOKEN_DEQUANT_HPP

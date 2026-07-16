/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_ATLASA2_COPY_L0C_TO_GM_HPP
#define CATLASS_CONV_TILE_ATLASA2_COPY_L0C_TO_GM_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, half, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, bfloat16_t, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::AtlasA2,
    ElementAccumulator_,
    Gemm::GemmType<ElementDst_, layout::NC1HWC0>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::NC1HWC0;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr uint16_t C0 = BYTE_PER_C0 / sizeof(ElementDst);

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout,
        uint8_t unitFlag = 0
    ) // (Batch, Cout1, Ho, Wo, C0)
    {
        // compute sizes
        uint32_t cout1Actual = dstLayout.shape(1);
        uint32_t coutRound = cout1Actual * C0;
        uint32_t hoActual = dstLayout.shape(2);
        uint32_t woActual = dstLayout.shape(3);
        uint32_t howoActual = hoActual * woActual;
        uint32_t howoRound = RoundUp<C0>(howoActual);
        // compute dstStride
        uint32_t strideHo = dstLayout.stride(2);   // Wo * C0
        uint32_t strideHoWo = dstLayout.stride(1); // Ho * Wo * C0
        uint32_t HoWo = strideHoWo / C0;

        for (int hoIdx = 0; hoIdx < hoActual; hoIdx++) {
            size_t gmOffset = hoIdx * strideHo;
            size_t l0Offset = hoIdx * woActual * C0;
            AscendC::FixpipeParamsV220 fixPipeParams(
                coutRound, // nSize
                woActual,  // mSize
                howoRound, // srcStride
                HoWo,      // dstStride
                reluEn
            );
            fixPipeParams.quantPre = quantPre;
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst[gmOffset], src[l0Offset], fixPipeParams);
        }
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <class TensorSrc_, class TensorDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::AtlasA2,
    TensorSrc_,
    TensorDst_,
    ScaleGranularity::NO_QUANT,
    ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = typename TensorDst_::Element;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementDst>::value;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE 
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        // compute sizes
        uint32_t coutRound = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        uint32_t howoRound = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        uint32_t hoActual = tla::get<2>(dstTensor.shape());
        uint32_t woActual = tla::get<3>(dstTensor.shape());
  
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = coutRound; 
        intriParams.mSize = woActual;
        intriParams.srcStride = howoRound;
        intriParams.dstStride = tla::get<1>(dstTensor.stride()) / ELE_NUM_PER_C0;

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto srcOffsetInner = woActual * ELE_NUM_PER_C0;

        // Call AscendC Fixpipe
        for (int hoIdx = 0; hoIdx < hoActual; hoIdx++) {
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(
                dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
            dstOffset += tla::get<2>(dstTensor.stride());
            srcOffset += srcOffsetInner;
        }
    }
};

} // namespace Catlass::Conv::Tile

#endif // CATLASS_CONV_TILE_ATLASA2_COPY_L0C_TO_GM_HPP

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_ASCEND950_COPY_L0C_TO_GM_HPP
#define CATLASS_CONV_TILE_ASCEND950_COPY_L0C_TO_GM_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/conv/tile/ascend950/copy_l0c_to_dst.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

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
    Catlass::Arch::Ascend950,
    TensorSrc_,
    TensorDst_,
    ScaleGranularity::NO_QUANT,
    ReluEnable_> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = typename TensorDst_::Element;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
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
  
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> intriParams;

        // Fixpipe layout information
        intriParams.nSize = coutRound; 
        intriParams.mSize = woActual;
        intriParams.srcStride = howoRound;
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

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

#endif // CATLASS_CONV_TILE_ASCEND950_COPY_L0C_TO_GM_HPP

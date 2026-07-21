/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ASCEND950_COPY_L1_TO_BT_HPP
#define CATLASS_GEMM_TILE_ASCEND950_COPY_L1_TO_BT_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/tile/tile_copy_tla.hpp"
#include "catlass_a5/numeric_size.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyL1ToBT, Ascend950, VectorLayout in and VectorLayout out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::C2>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isVector<typename TensorSrc::Layout>::value &&
                          tla::detail::isVector<typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::A1 &&
                          TensorDst::position == AscendC::TPosition::C2,
                      "The input parameters do not match. TensorSrc must be L1 and Vector, "
                      "while TensorDst must be BT and Vector");

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.originShape()), ELE_NUM_PER_C0);
        if (sizeof(ElementSrc) == 4) {
            // the burst length should be even when B32
            intriParams.blockLen = RoundUp(intriParams.blockLen, 2);
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

////////////////////////////////////TileCopy(No-TLA, Ascend950)////////////////////////////////////////////////

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToBT {
    static_assert(DEPENDENT_FALSE<ArchTag>,
                  "Unsupported copy l1 to biasTable buffer, can not find the specialization.");
};

template <class ElementSrc, class ElementDst>
struct CopyL1ToBT<Arch::Ascend950, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
                  Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    CATLASS_DEVICE
    CopyL1ToBT() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementDst> dstTensor, AscendC::LocalTensor<ElementSrc> srcTensor,
                    LayoutDst layoutDst, LayoutSrc layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(layoutDst.shape(0), ELE_NUM_PER_C0);
        if (sizeof(ElementSrc) == 4) {
            // the burst length should be even when B32
            intriParams.blockLen = RoundUp(intriParams.blockLen, 2);
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Tile

#endif  // CATLASS_GEMM_TILE_ASCEND950_COPY_L1_TO_BT_HPP

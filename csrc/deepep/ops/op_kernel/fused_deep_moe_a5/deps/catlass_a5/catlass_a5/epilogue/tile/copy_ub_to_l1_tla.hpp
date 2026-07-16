/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_COPY_UB_TO_L1_TLA_HPP
#define CATLASS_EPILOGUE_TILE_COPY_UB_TO_L1_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Tile {

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct CopyUb2L1Tla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyUb2L1Tla, can not find the specialization.");
};

/// Partial specialization for CopyUbToL1, Ascend950, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be VECCALC and RowMajor, while TensorDst must be L1 and "
            "zN");

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstNzNStride = tla::get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        const uint32_t dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        uint32_t rows = nValue;
        uint32_t cols = RoundUp<BYTE_PER_BLK>(dValue * sizeof(ElementSrc)) / BYTE_PER_BLK;
        if (cols > rows) {
            for (uint32_t i = 0; i < rows; i++) {
                uint32_t dst = i * dstNzNStride * BYTE_PER_BLK / sizeof(ElementDst);
                uint32_t src = i * srcDValue;
                AscendC::DataCopyParams dataCopyParams(cols, 1, 0, dstNzC0Stride - 1);
                AscendC::DataCopy(dstTensor.data()[dstOffset + dst], srcTensor.data()[srcOffset + src], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < cols; i++) {
                uint32_t dst = i * dstNzC0Stride * BYTE_PER_BLK / sizeof(ElementDst);
                uint32_t src = i * BYTE_PER_BLK / sizeof(ElementSrc);
                AscendC::DataCopyParams dataCopyParams(
                    rows, 1, srcDValue * sizeof(ElementSrc) / BYTE_PER_BLK - 1, dstNzNStride - 1);
                AscendC::DataCopy(dstTensor.data()[dstOffset + dst], srcTensor.data()[srcOffset + src], dataCopyParams);
            }
        }
    }
};

/// Partial specialization for Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value &&
                     tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::iszNUnAlign<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                      tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendC::TPosition::VECCALC &&
                      TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be UB and zN, "
            "while TensorDst must be L1 and zN");

        int64_t srcShape = tla::get<0, 0>(srcTensor.shape());
        AscendC::DataCopyParams dataCopyParams(
            tla::get<1, 1>(srcTensor.shape()),
            srcShape,
            (tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - srcShape),
            (tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - srcShape)
        );
        
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    }
};

}  // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_COPY_UB_TO_L1_TLA_HPP


/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ASCEND950_COPY_GM_TO_L1_HPP
#define CATLASS_GEMM_TILE_ASCEND950_COPY_GM_TO_L1_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/tile/tile_copy_tla.hpp"
#include "catlass_a5/numeric_size.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyGmToL1, Ascend950, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t ndNum = 1,
                                   uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN");

        const uint32_t nValue = tla::get<0>(srcTensor.originShape());
        const uint32_t dValue = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        // AscendC Nd2Nz for float4: pass dValue/srcDValue as int8 length (aligned with Ascend950 copy_gm_to_l1).
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
        }
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and zN, while TensorDst must be L1 and zN");

        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcTensor.originShape()));
        uint32_t blockLen = tla::get<0>(srcTensor.originShape());

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopyParams repeatParams;
        if (srcOuterStrideCol / ELE_NUM_PER_C0 <= STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = dstOuterStrideCol / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                AscendC::DataCopy(dstTensor.data()[dstOffset + i * dstOuterStrideCol],
                                  srcTensor.data()[srcOffset + i * srcOuterStrideCol], repeatParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t ndNum = 1,
                                   uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(tla::detail::isColumnMajor<typename TensorSrc::Layout>::value &&
                          tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
                      "while TensorDst must be L1 and nZ");

        const uint32_t nValue = tla::get<1>(srcTensor.originShape());
        const uint32_t dValue = tla::get<0>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
        }
        intriParams.dstNzC0Stride = dstOuterStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                          tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and nZ, "
                      "while TensorDst must be L1 and nZ");

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcTensor.originShape()));
        uint32_t blockLen = tla::get<1>(srcTensor.originShape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, VectorLayout in and VectorLayout out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
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
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and Vector, "
                      "while TensorDst must be L1 and Vector");

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.originShape()), ELE_NUM_PER_C0);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1 TLA, Ascend950, MX fp8 e8m0, MxScaleA RowMajor in and zZ out.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, LayoutSrc>::value &&
                                    tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {
    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleForzZ<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and RowMajor, while TensorDst must be "
            "fp8_e8m0_t, L1 and zZ");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Dn2NzParams intriParams;

        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Partial specialization for CopyGmToL1 TLA, Ascend950, MX fp8 e8m0, MxScaleA ColumnMajor in and zZ out.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, LayoutSrc>::value &&
                                    tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {
    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleForzZ<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and ColumnMajor, while TensorDst must "
            "be fp8_e8m0_t, L1 and zZ");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Partial specialization for CopyGmToL1 TLA, Ascend950, MX fp8 e8m0, B RowMajor in and nN out.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, LayoutSrc>::value &&
                                    tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {
    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleFornN<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and RowMajor, while TensorDst must be "
            "fp8_e8m0_t, L1 and nN");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(rows);
        intriParams.dValue = cols;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideCol / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Partial specialization for CopyGmToL1 TLA, Ascend950, MX fp8 e8m0, B ColumnMajor in and nN out.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, LayoutSrc>::value &&
                                    tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {
    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleFornN<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and ColumnMajor, while TensorDst must "
            "be fp8_e8m0_t, L1 and nN");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Dn2NzParams intriParams;

        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(rows);
        intriParams.dValue = cols;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideCol / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

////////////////////////////////////CopyGmToL1(No-TLA, Ascend950)////////////////////////////////////////////////
template <class ArchTag,
          /// GemmType for matrix operand
          class GmType, class L1Type = void, class Enable = void>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <class ArchTag,
          /// GemmType for matrix operand
          class GmType, class L1Type = void>
struct CopyGmToL1IntervalDataCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <class ArchTag,
          /// GemmType for matrix operand
          class GmType, class L1Type = void>
struct CopyGmToL1GMMPTD {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <class ArchTag,
          /// GemmType for matrix operand
          class GmType, class L1Type = void>
struct CopyGmToL1DynamicOptimized {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods
    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t dstInnerStrideRow = layoutDst.stride(0);
        const uint32_t dstOuterStrideCol = layoutDst.stride(3);

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        // AscendC Nd2Nz for float4: pass dValue/srcDValue as int8 length (aligned with Ascend950 copy_gm_to_l1).
        if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv<2>(intriParams.dValue);
            intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
        }

        // strideColsByFractal -->
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, uint32_t ndNum, uint32_t srcNdMatrixStride,
                    uint32_t dstNzNStride, uint32_t dstNzMatrixStride, uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);

        // Manually set the copy stride
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for CopyGmToL1(Ascend950 no-tla), ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods
    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = layoutSrc.shape(1);
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcDValue = layoutSrc.stride(1);
        if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv<2>(intriParams.dValue);
            intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
        }

        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;  // Outer stride -- col
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for Ascend950, ColumnMajor in and nZ B1 out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
                  Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = layoutSrc.shape(1);
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcDValue = layoutSrc.stride(1);
        if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv<2>(intriParams.dValue);
            intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
        }

        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for Ascend950, ColumnMajor in and nZ A1 out, with triple template.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
                  Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};

/// Partial specialization for CopyGmToL1(Ascend950 no-tla), zN in and zN out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    // Note: AscendC::int4b_t no longer supported on Ascend950 platform

    // Methods
    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t blockLen = layoutSrc.shape(0) * layoutSrc.shape(1);
        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = layoutSrc.shape(3);
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(3); i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1(Ascend950 no-tla), nZ in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    // Note: AscendC::int4b_t no longer supported on Ascend950 platform

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t blockLen = layoutSrc.shape(2) * layoutSrc.shape(3);
        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = layoutSrc.shape(1);
        repeatParams.blockLen = blockLen;

        repeatParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = layoutDst.stride(1) / ELE_NUM_PER_C0 - blockLen;

        AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
    }
};

/// Partial specialization for CopyGmToL1(no-tla), Ascend950, fp8_e8m0_t, MxScaleA RowMajor in and zZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
                  Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>,
                  std::enable_if_t<std::is_same_v<Element, AscendC::fp8_e8m0_t>>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = 2;

    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        if (layoutSrc.shape(2) != ELE_NUM_PER_C0) {
            return;
        }

        AscendC::Dn2NzParams intriParams;

        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(layoutSrc.shape(2) * layoutSrc.shape(3));
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcDnMatrixStride = 0;

        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(layoutSrc.stride(0));
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for Ascend950, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(0);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        // AscendC Nd2Nz for float4: pass dValue/srcDValue as int8 length (aligned with Ascend950 copy_gm_to_l1).
        if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv<2>(intriParams.dValue);
            intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
        }
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for Ascend950, PaddingColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(1);
        intriParams.srcDValue = layoutSrc.stride(2);
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

        if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv<2>(intriParams.dValue);
            intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
        }
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for Ascend950, RowMajor in and zN A1 out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
                  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

/// Partial specialization for Ascend950, RowMajor in and zN B1 out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
                  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

/// Partial specialization for Ascend950, RowMajor in and zZ B1 out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
                  Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(0);
        uint32_t ndNum = layoutSrc.shape(0) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(0) % C0_NUM_PER_FRACTAL;

        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(0);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    intriParams.dValue = CeilDiv<2>(intriParams.dValue);
                    intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
                }

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = layoutDst.stride(1);
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(0);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    tailParams.dValue = CeilDiv<2>(tailParams.dValue);
                    tailParams.srcDValue = CeilDiv<2>(tailParams.srcDValue);
                }
                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(0);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    intriParams.dValue = CeilDiv<2>(intriParams.dValue);
                    intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
                }

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(0);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    tailParams.dValue = CeilDiv<2>(tailParams.dValue);
                    tailParams.srcDValue = CeilDiv<2>(tailParams.srcDValue);
                }

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(1);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    intriParams.dValue = CeilDiv<2>(intriParams.dValue);
                }
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;
                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(0);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

/// Partial specialization for Ascend950, RowMajor in and RowMajor A1 out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
                  Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride);
                DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                         srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};

/// Partial specialization for Ascend950, ColumnMajor in and nN out.
template <class Element, AscendC::TPosition Position>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
                  Gemm::GemmType<Element, layout::nN, Position>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;

        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    intriParams.dValue = CeilDiv<2>(intriParams.dValue);
                    intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
                }
                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = layoutDst.stride(3);
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    tailParams.dValue = CeilDiv<2>(tailParams.dValue);
                    tailParams.srcDValue = CeilDiv<2>(tailParams.srcDValue);
                }

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    intriParams.dValue = CeilDiv<2>(intriParams.dValue);
                    intriParams.srcDValue = CeilDiv<2>(intriParams.srcDValue);
                }
                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    tailParams.dValue = CeilDiv<2>(tailParams.dValue);
                    tailParams.srcDValue = CeilDiv<2>(tailParams.srcDValue);
                }
                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                if constexpr (AscendC::Std::is_one_of_v<Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    intriParams.dValue = CeilDiv<2>(intriParams.dValue);
                }
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;
                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

/// Partial specialization for Ascend950, VectorLayout in and zN out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout>,
                  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.nValue = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = layoutSrc.shape(0);
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for Ascend950, VectorLayout in and VectorLayout out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
                  Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

//////////////////////////// CopyGmToL1DynamicOptimized(Ascend950, No TLA) ////////////////////////////

/// Partial specialization for Ascend950, zN in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {};

/// Partial specialization for Ascend950, nZ in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {};

/// Partial specialization for Ascend950, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods
    CATLASS_DEVICE
    CopyGmToL1DynamicOptimized() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        if (layoutSrc.shape(0) <= 16) {
            // If the number of matrix row is very small, call the regular interval-based data-copy
            for (int i = 0; i < layoutSrc.shape(0); ++i) {
                AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                                                       layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                                                       (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
                AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)],
                                  dataCopyParams);
            }
        } else {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.dValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;

            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        }
    }
};

/// Partial specialization for Ascend950, ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods
    CATLASS_DEVICE
    CopyGmToL1DynamicOptimized() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        if (layoutSrc.shape(1) <= 16) {
            // If the number of matrix cols is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            for (int i = 0; i < layoutSrc.shape(1); ++i) {
                AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)),
                                                       layoutDst.shape(0) / ELE_NUM_PER_C0, 0,
                                                       (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
                AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)],
                                  dataCopyParams);
            }
        } else {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.dValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(1);

            intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;  // Outer stride -- col
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;

            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        }
    }
};

/// Partial specialization for Ascend950, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};

//////////////////////////// CopyGmToL1GMMPTD(Ascend950, No TLA) ////////////////////////////
/// Partial specialization for Ascend950, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1GMMPTD<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1GMMPTD() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(0) == 1) {
            // If the number of matrix rows is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                                                   layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                                                   (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor, srcTensor, dataCopyParams);
        } else {
            if (layoutSrc.shape(1) != ELE_NUM_PER_C0 || layoutSrc.stride(0) != ELE_NUM_PER_C0) {
                intriParams.nValue = layoutSrc.shape(0);
                intriParams.srcDValue = layoutSrc.stride(0);
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            } else {
                // If the matrix has ELE_NUM_PER_C0 columns and a stride of ELE_NUM_PER_C0, it follows a row-major
                // layout in L1, allowing the use of the standard contiguous DataCopy interface for more efficient
                // transfers.
                AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, uint32_t ndNum, uint32_t srcNdMatrixStride,
                    uint32_t dstNzNStride, uint32_t dstNzMatrixStride, uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        intriParams.ndNum = ndNum;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

//////////////////////////// CopyGmToL1IntervalDataCopy(Ascend950, No TLA) ////////////////////////////
/// Target: use `DataCopy` interface can acquire better performance compared to `ND2NZ`,
/// fitting shape is 1. Unaligned; 2. "short and wide"; 3. "tall and narrow"

/// Partial specialization for Ascend950, half, RowMajor in and zN out.
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                                                   layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                                                   (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for Ascend950, half, ColumnMajor in and nZ out.
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)),
                                                   layoutDst.shape(0) / ELE_NUM_PER_C0, 0,
                                                   (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
        }
    }
};

/// Partial specialization for Ascend950, half, PaddingRowMajor in and zN out.
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.orgShape(1), layoutDst.shape(2)),
                                                   layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                                                   (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for Ascend950, half, PaddingColumnMajor in and nZ out.
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(CeilDiv(layoutSrc.orgShape(0), layoutDst.shape(0)),
                                                   layoutDst.shape(0) / ELE_NUM_PER_C0, 0,
                                                   (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(2)], dataCopyParams);
        }
    }
};

}  // namespace Catlass::Gemm::Tile

#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_KDA_ASCEND950_COPY_GM_TO_L1_HPP
#define CATLASS_KDA_ASCEND950_COPY_GM_TO_L1_HPP

#include "catlass/arch/kda_arch.hpp"
#include "catlass/kda_catlass.hpp"
#include "catlass/gemm/tile/kda_tile_copy_tla.hpp"
#include "catlass/kda_numeric_size.hpp"
#include "catlass/layout/kda_layout.hpp"
#include "catlass/gemm/kda_gemm_type.hpp"
#include "tla/kda_tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyGmToL1, Ascend950, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
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
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
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
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and zN, while TensorDst must be L1 and zN");

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcTensor.originShape()));
        uint32_t blockLen = tla::get<0>(srcTensor.originShape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorSrc::Layout>::value && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
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
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
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
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
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
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isVector<typename TensorSrc::Layout>::value && tla::detail::isVector<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
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

////////////////////////////////////CopyGmToL1(No-TLA, Ascend950)////////////////////////////////////////////////
template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void,
    class Enable = void>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void>
struct CopyGmToL1GMMPTD {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void>
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
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t dstInnerStrideRow = layoutDst.stride(0);
        const uint32_t dstOuterStrideCol = layoutDst.stride(3);

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);

        // strideColsByFractal -->
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
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

/// Partial specialization for CopyGmToL1(AtlasA5 no-tla), ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    // Note: AscendC::int4b_t no longer supported on A5-platform

    // Methods
    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        // <dstNzC0Stride>dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride()) -->
        // tag.stride(1) --> strideRowsByFractal[语义]
        // <dstNzNStride>dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride()) -->
        // tag.stride(2) --> strideColsInFractal[语义]

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = layoutSrc.shape(1);
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcDValue = layoutSrc.stride(1);

        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0; // Outer stride -- col
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for CopyGmToL1(AtlasA5 no-tla), zN in and zN out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    // Note: AscendC::int4b_t no longer supported on A5-platform

    // Methods
    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t blockLen = layoutSrc.shape(0) * layoutSrc.shape(1);
        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = layoutSrc.shape(3);
        repeatParams.blockLen = blockLen;

        repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen;

        AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
    }
};

/// Partial specialization for CopyGmToL1(AtlasA5 no-tla), nZ in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    // Note: AscendC::int4b_t no longer supported on A5-platform

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
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

/// Partial specialization for CopyGmToL1(no-tla), AtlasA5, fp8_e8m0_t, MxScaleA RowMajor in and zZ out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950,
                  Gemm::GemmType<Element, layout::RowMajor>,
                  Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>,
                  std::enable_if_t<std::is_same_v<Element, AscendC::fp8_e8m0_t>>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = 2;

    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        if (layoutSrc.shape(2) != ELE_NUM_PER_C0) {
            // std::cerr << "layoutSrc.shape(2) != 2" << std::endl;
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

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
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
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <
    class Element>
struct CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
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
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

//////////////////////////// CopyGmToL1DynamicOptimized(Ascend950, No TLA) ////////////////////////////

/// Partial specialization for Ascend950, zN in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {};

/// Partial specialization for Ascend950, nZ in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {};

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
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        if (layoutSrc.shape(0) <= 16) {
            // If the number of matrix row is very small, call the regular interval-based data-copy
            for (int i = 0; i < layoutSrc.shape(0); ++i) {
                AscendC::DataCopyParams dataCopyParams(
                    CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                    layoutDst.shape(2) / ELE_NUM_PER_C0,
                    0,
                    (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
                AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
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
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        if (layoutSrc.shape(1) <= 16) {
            // If the number of matrix cols is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            for (int i = 0; i < layoutSrc.shape(1); ++i) {
                AscendC::DataCopyParams dataCopyParams(
                    CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)),
                    layoutDst.shape(0) / ELE_NUM_PER_C0,
                    0,
                    (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
                AscendC::DataCopy(
                    dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
            }
        } else {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.dValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(1);

            intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0; // Outer stride -- col
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;

            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        }
    }
};

//////////////////////////// CopyGmToL1GMMPTD(Ascend950, No TLA) ////////////////////////////
/// Partial specialization for Ascend950, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1GMMPTD<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1GMMPTD() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
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
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0,
                0,
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
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif

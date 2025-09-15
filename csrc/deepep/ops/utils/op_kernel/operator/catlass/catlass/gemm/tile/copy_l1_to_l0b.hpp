/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP
#define CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////////////
/// new add gemm
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for(uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){  // K N
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<float, layout::zZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<float, layout::nZ, AscendC::TPosition::B2>>{
    using Element = float;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1))) - 1;
        for(uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){ // K N
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};


template<class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>>{
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};
/////////////////////////////////////////////

////////////////////////////////////////////
/// new add gemv
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(3); i++)
        {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(3)], srcTensor[i * layoutSrc.stride(3)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>>
{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = layoutDst.shape(1) * layoutDst.shape(3);
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    };
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<float, layout::nN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<float, layout::zN, AscendC::TPosition::B2>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nN;
    using Element = float;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)) - 1;

        for (uint32_t i = 0; i < CeilDiv<2 * ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
        {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3) * 2],
                srcTensor[i * layoutSrc.stride(3)],
                loadDataParams);
        }
    };
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B2>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nZ;
    using Element = int8_t;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
        {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3)],
                srcTensor[i * layoutSrc.stride(3) * 2],
                loadDataParams);
        }
    }
};
////////////////////////////////////////////

/// Partial specialization for int8_t, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

/// Partial specialization for float, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<float, layout::zN, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1K = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1N = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0K = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0N = layoutDst.shape(2) * layoutDst.shape(3);
        // K, N need to be 16 aligned for f32
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l1NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1N);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        uint16_t l0NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0N);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_RIGHT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0NAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.channelSize = l1NAlign;
        loadDataParams.fMatrixCtrl = true;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

/// Partial specialization for zN in and nZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for nZ in and nZ out. (Transpose B)
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        if (layoutSrc.shape(3) == layoutDst.shape(3)) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }

    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyL1ToL0B, AtlasA2, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                      tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendC::TPosition::A1 &&
                      TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ");

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                              srcTensor.data()[srcOffset + i * srcOuterStrideRow],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                      tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendC::TPosition::A1 &&
                      TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0B and nZ");

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                              srcTensor.data()[srcOffset + i * srcOuterStrideRow],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, int8_t, zN in and nZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::iszN<int8_t, LayoutSrc>::value && tla::detail::isnZ<int8_t, LayoutDst>::value>> {
    using Element = int8_t;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(std::is_same_v<int8_t, typename TensorSrc::Element> &&
                      std::is_same_v<int8_t, typename TensorDst::Element> &&
                      tla::detail::iszN<int8_t, typename TensorSrc::Layout>::value &&
                      tla::detail::isnZ<int8_t, typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendC::TPosition::A1 &&
                      TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be int8_t, L1 and zN, "
            "while TensorDst must be int8_t, L0B and nZ");

        const uint32_t srcOuterShapeCol = tla::get<1, 1>(srcTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = srcOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadDataWithTranspose(dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                                           srcTensor.data()[srcOffset + i * srcOuterStrideRow * 2],
                                           loadDataParams);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP
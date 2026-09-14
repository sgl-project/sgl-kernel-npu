/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_KDA_ASCEND950_COPY_L1_TO_L0A_HPP
#define CATLASS_KDA_ASCEND950_COPY_L1_TO_L0A_HPP

#include "catlass/arch/kda_arch.hpp"
#include "catlass/kda_catlass.hpp"
#include "catlass/kda_numeric_size.hpp"
#include "catlass/layout/kda_layout.hpp"
#include "catlass/gemm/kda_gemm_type.hpp"
#include "catlass/gemm/tile/kda_tile_copy_tla.hpp"
#include "tla/kda_tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyL1ToL0A, Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0A and zN");

        const uint32_t dstOuterShapeRow = CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol = CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
        loadDataParams.mStep = dstOuterShapeRow;
        loadDataParams.kStep = dstOuterShapeCol;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
        loadDataParams.ifTranspose = false;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0A and zN");

        const uint32_t dstOuterShapeRow = CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol = CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = dstOuterShapeRow;
        loadDataParams.kStep = dstOuterShapeCol * l0Batch;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
        loadDataParams.ifTranspose = false;

        AscendC::LoadData(dstTensor.data(), srcTensor.data(), loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, Ascend950, not B8, nZ in and zN out. (Transpose A)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        !AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            !AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                !AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0A and zN");

        const uint32_t L0M = tla::get<0>(dstTensor.originShape());
        const uint32_t L0K = tla::get<1>(dstTensor.originShape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcCoord));
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0M);
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float, uint32_t, int32_t>) {
            loadDataParams.kStep = RoundUp<2>(loadDataParams.kStep); // for b32 data types, ensure kStep is even
        }
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
        loadDataParams.ifTranspose = true;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch)
    {
        static_assert(
            !AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                !AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0A and zN");

        const uint32_t L1M = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L1K = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t L0M = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0K = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t L0MOrigin = tla::get<0>(dstTensor.originShape());
        const uint32_t L0KOrigin = tla::get<1>(dstTensor.originShape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0KOrigin);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0MOrigin);
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float, uint32_t, int32_t>) {
            loadDataParams.kStep = RoundUp<2>(loadDataParams.kStep); // for b32 data types, ensure kStep is even
        }
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
        loadDataParams.ifTranspose = true;

        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendC::LoadData(
                dstTensor.data()[l0BatchIdx * L0M * L0K], srcTensor.data()[l0BatchIdx * L1M * L1K], loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0A, Ascend950, B8, nZ in and zN out. (Transpose A)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0A and zN");

        const uint32_t L0MPadded = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0MOrigin = tla::get<0>(dstTensor.originShape());
        const uint32_t L0KOrigin = tla::get<1>(dstTensor.originShape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendC::LoadData2DParamsV2 loadDataParams;
        if (RoundUp<C0_NUM_PER_FRACTAL>(L0MOrigin) % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(srcCoord));
            loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcCoord));
            loadDataParams.mStep = RoundUp<2>(CeilDiv<C0_NUM_PER_FRACTAL>(L0KOrigin));
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0MOrigin);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
            loadDataParams.ifTranspose = true;

            auto dstOffset = dstTensor.layout()(dstTensor.coord());
            AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
        } else {
            for (uint32_t kIdx = 0; kIdx < CeilDiv<ELE_NUM_PER_C0>(L0KOrigin); kIdx++) {
                loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(srcCoord)) + kIdx * 2;
                loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcCoord));
                loadDataParams.mStep = 2;
                loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0MOrigin);
                loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
                loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
                loadDataParams.ifTranspose = true;

                auto dstOffset = dstTensor.layout()(dstTensor.coord());
                AscendC::LoadData(
                    dstTensor.data()[dstOffset + kIdx * L0MPadded * ELE_NUM_PER_C0], srcTensor.data(), loadDataParams);
            }
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch)
    {
        static_assert(
            AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
                tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0A and zN");

        const uint32_t L1M = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L1K = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t L0M = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0K = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t L0MPadded = L0M;
        const uint32_t L0KPadded = L0K;
        const uint32_t L0MOrigin = tla::get<0>(dstTensor.originShape());
        const uint32_t L0KOrigin = tla::get<1>(dstTensor.originShape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::LoadData2DParamsV2 loadDataParams;
        if (RoundUp<C0_NUM_PER_FRACTAL>(L0MOrigin) % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = RoundUp<2>(CeilDiv<C0_NUM_PER_FRACTAL>(L0KOrigin));
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0MOrigin);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
            loadDataParams.ifTranspose = true;

            for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
                AscendC::LoadData(
                    dstTensor.data()[l0BatchIdx * L0MPadded * L0KPadded], srcTensor.data()[l0BatchIdx * L1M * L1K], loadDataParams);
            }
        } else {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = 2;
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0MOrigin);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
            loadDataParams.ifTranspose = true;
            for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
                for (uint32_t kIdx = 0; kIdx < CeilDiv<ELE_NUM_PER_C0>(L0KOrigin); kIdx++) {
                    AscendC::LoadData(
                        dstTensor.data()[l0BatchIdx * L0MPadded * L0KPadded + kIdx * L0MPadded * ELE_NUM_PER_C0],
                        srcTensor.data()[l0BatchIdx * L1M * L1K + kIdx * ELE_NUM_PER_FRACTAL * 2], loadDataParams);
                }
            }
        }
    }
};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        uint16_t aL1M = tla::get<0, 0>(srcTensor.stride());
        uint16_t madM = tla::get<1, 1>(dstTensor.stride());
        uint16_t madK = CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(madM);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(madK);
        loadDataParams.srcStride = CeilDiv<C0_NUM_PER_FRACTAL>(aL1M);
        loadDataParams.dstStride = CeilDiv<C0_NUM_PER_FRACTAL>(madM);

        loadDataParams.ifTranspose = false;
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::LoadData(dstTensor.data(), srcTensor.data()[srcOffset], loadDataParams);
    }
};

////////////////////////////////////CopyL1ToL0A(No-TLA, Ascend950)////////////////////////////////////////////////
template <
    class ArchTag,
    class L1Type,
    class L0Type = void,
    class Enable = void>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

/// Partial specialization for CopyL1ToL0A, AtlasA5, zN in and zN out.
template <class Element>
struct CopyL1ToL0A<Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = layoutDst.shape(1);
        loadDataParams.kStep = layoutDst.shape(3);
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(layoutSrc.stride(3));
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(layoutDst.stride(3));
        loadDataParams.ifTranspose = false;

        AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA5, not B8 or B4, nZ in and zN out. (Transpose A)
template <class Element>
struct CopyL1ToL0A<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>, void,
                   std::enable_if_t<
                       !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t L0M = layoutDst.shape(0) * layoutDst.shape(1);
        const uint32_t L0K = layoutDst.shape(2) * layoutDst.shape(3);
        const uint32_t srcOuterStrideRow = layoutSrc.stride(1);
        const uint32_t dstOuterStrideCol = layoutDst.stride(3);

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0M);
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
        loadDataParams.ifTranspose = true;

        AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA5, B8 or B4, nZ in and zN out. (Transpose A)
template <class Element>
struct CopyL1ToL0A<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>, void,
                   std::enable_if_t<
                       AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const uint32_t L0M = layoutDst.shape(0) * layoutDst.shape(1);
        const uint32_t L0K = layoutDst.shape(2) * layoutDst.shape(3);
        const uint32_t srcOuterStrideRow = layoutSrc.stride(1);
        const uint32_t dstOuterStrideCol = layoutDst.stride(3);

        AscendC::LoadData2DParamsV2 loadDataParams;
        if (L0M % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0M);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
            loadDataParams.ifTranspose = true;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            for (uint32_t kIdx = 0; kIdx < L0K / ELE_NUM_PER_C0; kIdx++) {
                loadDataParams.mStartPosition = kIdx * 2;
                loadDataParams.kStartPosition = 0;
                loadDataParams.mStep = 2;
                loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0M);
                loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
                loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideCol);
                loadDataParams.ifTranspose = true;

                AscendC::LoadData(
                    dstTensor[kIdx * L0M * ELE_NUM_PER_C0], srcTensor, loadDataParams);
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_KDA_ASCEND950_COPY_L1_TO_L0A_HPP

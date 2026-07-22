/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_COPY_HPP
#define CATLASS_EPILOGUE_TILE_TILE_COPY_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/detail/tag_to_layout.hpp"
#include "catlass_a5/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass_a5/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass_a5/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass_a5/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "tla_a5/tensor.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass_a5/epilogue/tile/copy_ub_to_l1_tla.hpp"
#endif

namespace Catlass::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag, class... Args>
struct TileCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <class ArchTag,
          /// GemmType for C matrix operand
          class CType,
          /// GemmType for D matrix operand
          class DType>
struct TileCopy<ArchTag, CType, DType> {
    using ElementC = typename CType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <class ArchTag,
          /// GemmType for C matrix operand
          class CType,
          /// GemmType for X matrix operand
          class XType,
          /// GemmType for D matrix operand
          class DType>
struct TileCopy<ArchTag, CType, XType, DType> {
    using ElementC = typename CType::Element;
    using ElementX = typename XType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, XType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <class ArchTag, class CType, class XType, class YType, class DType>
struct TileCopy<ArchTag, CType, XType, YType, DType> {
    using ElementC = typename CType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, XType>;
    using CopyGmToUbY = CopyGm2Ub<ArchTag, YType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <class ArchTag, class CType, class XType, class YType, class DType>
struct TileCopyBf16 {
    using ElementC = typename CType::Element;
    using ElementX = bfloat16_t;
    using ElementY = bfloat16_t;
    using ElementD = bfloat16_t;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, Gemm::GemmType<bfloat16_t, typename XType::Layout>>;
    using CopyGmToUbY = CopyGm2Ub<ArchTag, Gemm::GemmType<bfloat16_t, typename YType::Layout>>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, Gemm::GemmType<bfloat16_t, typename DType::Layout>>;
};

template <class ArchTag, class CType, class ScaleType, class PerTokenScaleType, class DType>
struct TileCopyPerTokenDequant {
    using ElementC = typename CType::Element;
    using ElementScale = typename ScaleType::Element;
    using ElementPerTokenScale = typename PerTokenScaleType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbScale = CopyGm2Ub<ArchTag, ScaleType>;
    using CopyGmToUbPerTokenScale = CopyPerTokenScale2Ub<ArchTag, PerTokenScaleType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <class ArchTag, class CType, class PerTokenScaleType, class DType>
struct TileCopyW4A4Gemm {
    using ElementC = typename CType::Element;
    using ElementPerTokenScale = typename PerTokenScaleType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbPerTokenScale = CopyGm2Ub<ArchTag, PerTokenScaleType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <class ArchTag,
          /// GemmType for C matrix operand
          class ElementC_, class LayoutTagC_,
          /// GemmType for X matrix operand
          class ElementX_, class LayoutTagX_,
          /// GemmType for Y matrix operand
          class ElementY_, class LayoutTagY_,
          /// GemmType for D matrix operand
          class ElementD_, class LayoutTagD_>
struct TileCopyDequantTla {
    using ElementC = ElementC_;
    using LayoutTagC = LayoutTagC_;
    using LayoutC = detail::TagToLayout_t<ElementC, LayoutTagC>;
    using TensorUbC =
        tla::Tensor<AscendC::LocalTensor<ElementC>, LayoutC, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;

    using ElementX = ElementX_;
    using LayoutTagX = LayoutTagX_;
    using LayoutX = detail::TagToLayout_t<ElementX, LayoutTagX>;
    using TensorUbX =
        tla::Tensor<AscendC::LocalTensor<ElementX>, LayoutX, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;

    using ElementY = ElementY_;
    using LayoutTagY = LayoutTagY_;
    using LayoutY = detail::TagToLayout_t<ElementY, LayoutTagY>;
    using TensorUbY =
        tla::Tensor<AscendC::LocalTensor<ElementY>, LayoutY, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;

    using ElementD = ElementD_;
    using LayoutTagD = LayoutTagD_;
    using LayoutD = detail::TagToLayout_t<ElementD, LayoutTagD>;
    using TensorUbD =
        tla::Tensor<AscendC::LocalTensor<ElementD>, LayoutD, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;

    template <class TensorC>
    using CopyGmToUbC = CopyGm2UbTla<ArchTag, TensorC, TensorUbC>;

    template <class TensorX>
    using CopyGmToUbX = CopyGm2UbTla<ArchTag, TensorX, TensorUbX>;

    template <class TensorY>
    using CopyGmToUbY = CopyGm2UbTla<ArchTag, TensorY, TensorUbY>;

    template <class TensorD>
    using CopyUbToGmD = CopyUb2GmTla<ArchTag, TensorUbD, TensorD>;
};

template <class ArchTag, class ElementMask_, class ElementP_, class LayoutTagMask_, class LayoutTagP_>
struct TileCopySoftmax {
    using ElementMask = ElementMask_;
    using ElementP = ElementP_;

    using LayoutTagMask = LayoutTagMask_;
    using LayoutTagP = LayoutTagP_;

    using LayoutMask = detail::TagToLayout_t<ElementMask, LayoutTagMask>;
    using LayoutP = detail::TagToLayout_t<ElementP, LayoutTagP>;

    using TensorGmMask = tla::Tensor<AscendC::GlobalTensor<ElementMask>, LayoutMask, tla::Coord<tla::_0, tla::_0>,
                                     AscendC::TPosition::GM>;
    using TensorUbMask = tla::Tensor<AscendC::LocalTensor<ElementMask>, LayoutMask, tla::Coord<tla::_0, tla::_0>,
                                     AscendC::TPosition::VECCALC>;

    using CopyGmToUbMask = Tile::CopyGm2UbTla<ArchTag, TensorGmMask, TensorUbMask>;
};

template <class ArchTag, class ElementO_, class LayoutTagO_, class LayoutTagOTmp_>
struct TileCopyRescaleO {
    using ElementO = ElementO_;
    using LayoutTagO = LayoutTagO_;
    using LayoutTagOTmp = LayoutTagOTmp_;
    using LayoutO = detail::TagToLayout_t<ElementO, LayoutTagO>;

    using TensorUbO =
        tla::Tensor<AscendC::LocalTensor<ElementO>, LayoutO, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorGmO =
        tla::Tensor<AscendC::GlobalTensor<ElementO>, LayoutO, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyUbToGmO = Tile::CopyUb2GmTla<ArchTag, TensorUbO, TensorGmO>;
};

}  // namespace Catlass::Epilogue::Tile

#endif  // CATLASS_EPILOGUE_TILE_TILE_COPY_HPP

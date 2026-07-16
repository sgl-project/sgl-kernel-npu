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

#ifndef CATLASS_CONV_TILE_TILE_COPY_HPP
#define CATLASS_CONV_TILE_TILE_COPY_HPP

#include <type_traits>

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/detail/tag_to_layout.hpp"
#include "catlass_a5/conv/tile/copy_gm_to_l1.hpp"
#include "catlass_a5/conv/tile/copy_l0c_to_gm.hpp"
#include "catlass_a5/conv/tile/copy_l1_to_l0a.hpp"
#include "catlass_a5/conv/tile/copy_l1_to_l0b.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
template <
    /// Tag indicating architecture
    class ArchTag,
    /// ConvType for Fmap operand
    class FmapType,
    /// ConvType type for Filter operand
    class FilterType,
    /// ConvType type for Output operand
    class OutputType,
    /// ConvType type for Bias operand
    class BiasType = void>
struct TileCopy {
    using ElementFmap = typename FmapType::Element;
    using ElementFilter = typename FilterType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;

    using CopyGmToL1A = Conv::Tile::CopyGmToL1<ArchTag, FmapType>;
    using CopyGmToL1B = Conv::Tile::CopyGmToL1<ArchTag, FilterType>;
    using CopyL1ToL0A = Conv::Tile::CopyL1ToL0A<ArchTag, typename Gemm::helper::L1ATypeSelector<FmapType>::L1AType>;
    using CopyL1ToL0B = Conv::Tile::CopyL1ToL0B<ArchTag, typename Gemm::helper::L1BTypeSelector<FilterType>::L1BType>;
    using CopyL0CToGm = Conv::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, OutputType>;
};
#endif

template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementFmap_,
    class LayoutTagFmap_,
    class ElementFilter_,
    class LayoutTagFilter_,
    class ElementOutput_,
    class LayoutTagOutput_,
    class ElementBias = void,
    bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY_ = ScaleGranularity::NO_QUANT>
struct PackedTileCopyTla {
    using ElementFmap = ElementFmap_;
    using ElementFilter = ElementFilter_;
    using ElementOutput = ElementOutput_;
    using LayoutTagFmap = LayoutTagFmap_;
    using LayoutTagFilter = LayoutTagFilter_;
    using LayoutTagOutput = LayoutTagOutput_;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;
    static constexpr bool ReluEnable = ReluEnable_;
    static constexpr ScaleGranularity DEQUANT_GRANULARITY = DEQUANT_GRANULARITY_;

    using LayoutTagL1A = typename Gemm::helper::L1ATypeSelector<Gemm::GemmType<ElementFmap, LayoutTagFmap>>::L1AType::Layout;
    using LayoutTagL1B = typename Gemm::helper::L1BTypeSelector<Gemm::GemmType<ElementFilter, LayoutTagFilter>>::L1BType::Layout;
    using LayoutTagL0A = typename Gemm::helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = layout::nZ;

    using LayoutFmap = detail::TagToLayout_t<ElementFmap, LayoutTagFmap>;
    using LayoutFilter = detail::TagToLayout_t<ElementFilter, LayoutTagFilter>;
    using LayoutOutput = detail::TagToLayout_t<ElementOutput, LayoutTagOutput>;

    using LayoutL1A = detail::TagToLayout_t<ElementFmap, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementFilter, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementFmap, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementFilter, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL0C = 
        tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementFmap, LayoutTagFmap>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementFilter, LayoutTagFilter>;

    using CopyGmToL1A = Conv::Tile::CopyGmToL1ATla<ElementFmap>;
    using CopyGmToL1B = Conv::Tile::CopyGmToL1BTla<ElementFilter>;
    using CopyL1ToL0A = Conv::Tile::CopyL1ToL0ATla<ElementFmap>;
    using CopyL1ToL0B = Conv::Tile::CopyL1ToL0BTla<ElementFilter>;

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    template <class TensorOut>
    using CopyL0CToDst = Conv::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorOut, DEQUANT_GRANULARITY, ReluEnable>;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
    template <class TensorOut>
    using CopyL0CToGm = Conv::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorOut, DEQUANT_GRANULARITY, ReluEnable>;
#endif
};

} // namespace Catlass::Conv::Tile

#endif // CATLASS_CONV_TILE_TILE_COPY_HPP

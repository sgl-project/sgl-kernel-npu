/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_COPY_EXPAND_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_EXPAND_HPP

#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

#include "copy_l0c_to_ub.hpp"

namespace Catlass::Gemm::Tile {
template <
    /// Tag indicating architecture
    class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_, class ElementMxScaleA_,
    class LayoutMxScaleA_, class ElementMxScaleB_, class LayoutMxScaleB_, class ElementC_, class LayoutTagC_,
    class ElementBias = void, uint32_t TargetUBIdx = 0, CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,
    bool ReluEnable = false, ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct PackedMxTileCopyTlaToTargetUB : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA_, ElementB_, LayoutTagB_,
                                                                ElementC_, LayoutTagC_, ElementBias> {
    using ElementMxScaleA = ElementMxScaleA_;
    using ElementMxScaleB = ElementMxScaleB_;

    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using LayoutMxScaleA = LayoutMxScaleA_;
    using LayoutMxScaleB = LayoutMxScaleB_;

    using LayoutTagL1MxScaleA = layout::zZ;
    using LayoutTagL1MxScaleB = layout::nN;
    using LayoutL1MxScaleA = detail::TagToLayout_t<ElementMxScaleA, LayoutTagL1MxScaleA>;
    using LayoutL1MxScaleB = detail::TagToLayout_t<ElementMxScaleB, LayoutTagL1MxScaleB>;

    using TensorL1MxScaleA = tla::Tensor<AscendC::LocalTensor<ElementMxScaleA>, LayoutL1MxScaleA,
                                         tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1MxScaleB = tla::Tensor<AscendC::LocalTensor<ElementMxScaleB>, LayoutL1MxScaleB,
                                         tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    template <class TensorMxScaleA>
    using CopyGmToL1MxScaleA = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;

    template <class TensorMxScaleB>
    using CopyGmToL1MxScaleB = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;

    static constexpr CopyL0CToUBMode CopyMode = CopyMode_;
    // 重写 CopyL0CToDst
    using TensorL0C = typename PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB, ElementC_,
                                                 LayoutTagC, ElementBias>::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst = Gemm::Tile::CopyL0CToTargetUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY,
                                                          TargetUBIdx, ReluEnable>;
};
}  // namespace Catlass::Gemm::Tile

#endif

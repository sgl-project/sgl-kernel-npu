/**

This program is free software, you can redistribute it and/or modify.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CATLASS_GEMM_PROLOGUE_TRATIS_WITH_SCALE_HPP
#define CATLASS_GEMM_PROLOGUE_TRATIS_WITH_SCALE_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_traits.hpp"
#include <type_traits>

namespace Catlass::Gemm::Tile {

template <class Prologue>
struct PrologueTraitsWithScale : public Prologue {
    using Prologue::Prologue;

    using TensorSrc = AscendC::GlobalTensor<typename Prologue::ElementSrc>;
    using TensorDst = AscendC::GlobalTensor<typename Prologue::ElementDst>;

    using ElementScale = typename Prologue::ElementScale;
    using LayoutScale  = typename Prologue::LayoutScale;
    using TensorScale  = AscendC::GlobalTensor<typename Prologue::ElementScale>;
};

template <>
struct PrologueTraitsWithScale<void> {
    using ElementSrc = EmptyType;
    using LayoutSrc = EmptyType;
    using ElementDst = EmptyType;
    using LayoutDst = EmptyType;

    using TensorSrc = EmptyType;
    using TensorDst = EmptyType;

    using ElementScale = EmptyType;
    using LayoutScale  = EmptyType;
    using TensorScale  = EmptyType;

    using Params = EmptyType;

    template <class... Args>
    CATLASS_DEVICE
    PrologueTraitsWithScale(Args...) {}
};

} //namespace Catlass::Gemm::Tile

#endif
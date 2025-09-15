/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_COPY_TLA_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_TLA_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTla, can not find the specialization.");
};

// Extended template for TileCopyTla that supports manually specifying LayoutTagSrc and LayoutTagDst.
// Users can specialize the copy class by LayoutTagSrc and LayoutTagDst.
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class LayoutTagSrc,
    class LayoutTagDst
>
struct TileCopyTlaExt {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTlaExt, can not find the specialization.");
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_COPY_TLA_HPP

/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_KDA_BLOCK_MMAD_HPP
#define CATLASS_KDA_BLOCK_MMAD_HPP

#include "catlass/kda_catlass.hpp"
#include "catlass/gemm/tile/kda_gemm_tile_copy.hpp"
#include "catlass/gemm/tile/kda_tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::RowMajor,
                                                   ElementB, layout::RowMajor, ElementC, layout::RowMajor, ElementBias>,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A> >
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Gemm::Block

// Only the pingpong-TLA implementation is instantiated by chunk_kda_fwd
// (Gemm::MmadPingpong).  The MmadPingpongTlaMulti / MmadPingpongTlaPreloadAL1B
// specializations live under kernel_utils/block/ and are included by their users.
#include "catlass/gemm/block/block_mmad_pingpong_tla.hpp"

#endif // CATLASS_KDA_BLOCK_MMAD_HPP

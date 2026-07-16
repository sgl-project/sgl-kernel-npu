/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>,
    class Enable = void
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 2201)
/// new add for the reason that i am using the dispatchpolicy which is same as the policy of the optimized_matmul
// so i add a new one class to avoid the conflict
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopyGemm<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,  // change the name
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockGemm {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class AType,
    class BType,
    class CType,
    class BiasType,
    class TileCopy,
    class TileMmad
>
struct BlockMmadAiv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadAiv is not implemented for this DispatchPolicy");
};


template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::SparseTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::RowMajor,
        ElementB, layout::ColumnMajor, ElementC, layout::RowMajor>
>
struct BlockMmadSparseTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSparseTla is not implemented for this DispatchPolicy");
};

#endif

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
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A>
>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementPrologueB,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::zN,
        ElementB, layout::zN, ElementC, layout::zN, ElementBias>,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A>
>
struct BlockMmadA8W4Mx {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadA8W4Mx is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class PrologueSrcType,
    class PrologueDstType,
    class L1TileShape,
    class TileCopy
>
struct BlockPrologue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockPrologue is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Gemm::Block

#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 2201)
#include "catlass_a5/gemm/block/block_mmad_fa_qk.hpp"
#include "catlass_a5/gemm/block/block_mmad_fa_pv.hpp"
#include "catlass_a5/gemm/block/block_mmad_mla_qk.hpp"
#include "catlass_a5/gemm/block/block_mmad_mla_pv.hpp"
#include "catlass_a5/gemm/block/block_mmad_mla_qk_tp1_spec.hpp"
#include "catlass_a5/gemm/block/block_mmad_mla_pv_tp1_spec.hpp"

#include "catlass_a5/gemm/block/block_mmad_amla_pv_tp1_spec.hpp"

#include "catlass_a5/gemm/block/block_mmad_preload.hpp"
#include "catlass_a5/gemm/block/block_mmad_preload_async.hpp"
#include "catlass_a5/gemm/block/block_mmad_preload_async_with_callback.hpp"
#include "catlass_a5/gemm/block/block_mmad_gemm.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_bias.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_qk_head_tail.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_qk_normal.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_pv_head_tail.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_pv_normal.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_full_loadA.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_full_loadA_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_with_prologue.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_slice_k_with_prologue.hpp"
#include "catlass_a5/gemm/block/block_mmad_dynamic_common.hpp"
#include "catlass_a5/gemm/block/block_mmad_dynamic_small.hpp"
#include "catlass_a5/gemm/block/block_mmad_dynamic_streamk.hpp"
#include "catlass_a5/gemm/block/block_mmad_dynamic_single_core_splitk.hpp"
#include "catlass_a5/gemm/block/block_mmad_dynamic_preload_async_with_callback.hpp"
#include "catlass_a5/gemm/block/block_mmad_small.hpp"
#include "catlass_a5/gemm/block/block_mmad_single_core_splitk.hpp"
#include "catlass_a5/gemm/block/block_mmad_dynamic_aiv.hpp"
#include "catlass_a5/gemm/block/block_mmad_streamk.hpp"
#include "catlass_a5/gemm/block/block_mmad_w4a4_per_token_per_channel_dequant.hpp"
#include "catlass_a5/gemm/block/block_mmad_sparse_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_qk_head_tail_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_qk_normal_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_pv_head_tail_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_pv_normal_tla.hpp"
#endif

/// Compactible block utility
#include "catlass_a5/gemm/block/block_mmad_pingpong.hpp"

/// TLA block utility
#include "catlass_a5/gemm/block/block_mmad_pingpong_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_dequant_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_tla_v2.hpp"
#include "catlass_a5/gemm/block/block_mmad_preload_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_preload_async_with_callback_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_pv_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_qk_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_per_group_per_block_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_mx_tla.hpp"
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass_a5/gemm/block/block_mmad_pingpong_mutex_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_pv_mx_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_fai_qk_mx_tla.hpp"
#include "catlass_a5/gemm/block/block_mx_a8w4_prologue.hpp"
#include "catlass_a5/gemm/block/block_mmad_mx_a8w4.hpp"
#include "catlass_a5/gemm/block/block_mmad_pingpong_full_loadA_ascend950_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_svd_quant_tla.hpp"
#include "catlass_a5/gemm/block/block_mmad_flash_attention_pv.hpp"
#include "catlass_a5/gemm/block/block_mmad_flash_attention_qk.hpp"
#include "catlass_a5/gemm/block/block_mmad_flash_attention_qk_DN.hpp"
#endif
#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP

// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* include file of ascendc */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

/* include file of catlass */
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "../utils/gemm/kernel/grouped_matmul_slice_m_fp8_w8a16.hpp"
#include "../utils/gemm/tile/cast_fp8_to_bf16_with_scale.hpp"
#include "../utils/gemm/block/block_mmad_pingpong_with_prologue_fp8_w8a16.hpp"
#include "../utils/gemm/dispatch_policy_ext.hpp"

using namespace Catlass;
using namespace matmul;

extern "C" __global__ __aicore__ void catlass_fp8w8a16_gmm_bfloat16_t(GM_ADDR deviceA, GM_ADDR deviceB,
                                                                      GM_ADDR deviceScale, GM_ADDR deviceGroupList,
                                                                      GM_ADDR deviceC, GM_ADDR deviceWorkspace,
                                                                      uint32_t g, uint32_t m, uint32_t n, uint32_t k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GemmCoord problemShape{m, n, k};
    uint32_t problemCount = g;
    uint32_t groupSize = 128;
    using ElementA = bfloat16_t;
    using ElementPrologueB = int8_t;
    using ElementB = bfloat16_t;
    using ElementC = bfloat16_t;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutPrologueB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutPrologueB layoutPrologueB{k, n};
    LayoutC layoutC{m, n};

    uint32_t scale_rows = CeilDiv(static_cast<size_t>(k), groupSize);
    uint32_t scale_cols = CeilDiv(static_cast<size_t>(n), groupSize);

    using ElementScale = float;
    using LayoutScale = layout::RowMajor;
    LayoutScale layoutScale{scale_rows, scale_cols};

    half deqScalar = 1.0;
    half deqZeroPoint = 0.0;
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
    using L0TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;

    using PrologueSrcType = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
    using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;
    using PrologueScaleType = Gemm::GemmType<ElementScale, LayoutScale>;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = PrologueDstType;
    using CType = Gemm::GemmType<ElementC, LayoutC>;

    using ArchTag = Arch::AtlasA2;
    constexpr bool ENABLE_UNIT_FLAG = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PingPongWithPrologueBFP8<ENABLE_UNIT_FLAG>;
    using PrologueA = void;
    constexpr uint32_t computeLen = 8 * 1024;
    using PrologueB = Gemm::Tile::TileCastFp8ToBf16WithScaleDequant<ArchTag, PrologueSrcType, PrologueDstType,
                                                                    PrologueScaleType, computeLen>;
    using TileCopy = Gemm::Tile::TileCopyWithPrologue<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;

    if (k > n) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMFP8W8A16<BlockMmadOpt, void, BlockScheduler, int64_t>;
        typename MatmulKernel::Params params_{
            problemShape, problemCount,    deviceGroupList, deviceA,        layoutA,
            deviceB,      layoutPrologueB, deviceC,         layoutC,        {{}, {deqScalar, deqZeroPoint}, {}},
            deviceScale,  layoutScale,     groupSize,       deviceWorkspace};
        MatmulKernel matmul_kernel;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        matmul_kernel(params_);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMFP8W8A16<BlockMmadOpt, void, BlockScheduler, int64_t>;
        typename MatmulKernel::Params params_{
            problemShape, problemCount,    deviceGroupList, deviceA,        layoutA,
            deviceB,      layoutPrologueB, deviceC,         layoutC,        {{}, {deqScalar, deqZeroPoint}, {}},
            deviceScale,  layoutScale,     groupSize,       deviceWorkspace};
        MatmulKernel matmul_kernel;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        matmul_kernel(params_);
    }
}

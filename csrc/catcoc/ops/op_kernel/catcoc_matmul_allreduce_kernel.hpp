// Licensed under the BSD 3-Clause License  (the "License");
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
/* include file of catlass */
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

/* include file of catcoc */
#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.h"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.h"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.h"
#include "catcoc/detail/remote_copy_type.h"
#include "catcoc/dgemm/kernel/matmul_allreduce.h"

// shmem_device
#include "shmem_api.h"

#include "catcoc_host_tiling.h"

using namespace AscendC;
using namespace Catcoc;
using namespace sglang::npu_kernel;

template <DataFormatMode dMode, WeightFormatMode wMode>
class CatCocMatmulAllreduce
{
public:
    uint64_t fftsAddr_;
    int32_t teamIdx_;

    __aicore__ inline void Init(uint64_t fftsAddr, uint64_t teamIdx)
    {
        fftsAddr_ = fftsAddr;
        teamIdx_ = (int32_t)teamIdx;
    }

    __aicore__ inline void Process(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric, GM_ADDR gmTiling)
    {
        // Set FFTS address
        AscendC::SetSyncBaseAddr(fftsAddr_);
        // Set shmem config
        int32_t newTeamIdx = 0;
        // int32_t newTeamIdx = teamIdx_;
        uint32_t rankIdx = shmem_team_my_pe(newTeamIdx);
        uint32_t rankSize = shmem_team_n_pes(newTeamIdx);

        // Define ArchTag
        using ArchTag = Catlass::Arch::AtlasA2;

        // unzip cocTiling
        auto tiling_data = reinterpret_cast<__gm__ sglang::npu_kernel::KernelCATCOCHostTilingData *>(gmTiling);
        uint32_t m = tiling_data->m;
        uint32_t n = tiling_data->n;
        uint32_t k = tiling_data->k;
        uint32_t m0 = tiling_data->m0;
        uint32_t k0 = tiling_data->k0;
        uint32_t n0 = tiling_data->n0;

        // switch cases
        using ElementA =
            typename std::conditional_t<dMode == DataFormatMode::FP16, half,
                                        typename std::conditional_t<dMode == DataFormatMode::BF16, __bf16, float>>;
        using ElementB = ElementA;
        using ElementC = ElementA;
        using ElementD = ElementA;

        using LayoutA = Catlass::layout::RowMajor;
        using LayoutB = typename std::conditional_t<wMode == WeightFormatMode::WEIGHT_ND, Catlass::layout::RowMajor,
                                                    Catlass::layout::zN>;
        using LayoutC = Catlass::layout::RowMajor;
        using LayoutD = Catlass::layout::RowMajor;

        using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
        using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
        using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
        using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

        LayoutA layoutA{m, k};
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);  // adapted for both NZ and ND
        LayoutD layoutD{m, n};
        Catlass::GemmCoord problemShape{m, n, k};

        /* init catcoc instance and run */
        constexpr bool enableUnitFlag = true;
        using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
        using L1TileShape = Catlass::GemmShape<128, 256, 256>;
        using L0TileShape = Catlass::GemmShape<128, 256, 64>;
        using BlockMmad =
            Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

        constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
        constexpr uint32_t SWIZZLE_DIRECTION = 1;
        using BlockMmadScheduler =
            Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
        using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

        using RemoteSrcType = CType;
        using RemoteDstType = DType;
        using CopyDirect = Catcoc::detail::CopyDirect;
        using TileRemoteCopy =
            CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
        using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

        constexpr uint32_t COMM_BLOCK_ROWS = 8;
        constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
        constexpr uint32_t CORE_SPLIT_ROWS = 16;
        constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
        using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
        using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

        constexpr uint32_t UB_STAGES = 2;
        constexpr uint32_t SCATTER_TILE_ROWS = 4;
        constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
        using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
        using EpilogueReduceScatterDispatch =
            CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES, Catcoc::detail::CopyMode::Scatter>;
        using BlockEpilogueReduceScatter =
            CommEpilogue::Block::CommBlockEpilogue<EpilogueReduceScatterDispatch, RemoteSrcType, RemoteDstType,
                                                   CommCoreSplit, CommBlockShape, EpilogueReduceScatterTileShape,
                                                   TileRemoteCopy, TileScheduler>;

        constexpr uint32_t ALLGATHER_TILE_ROWS = 4;
        constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
        using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
        using EpilogueAllGatherDispatch =
            CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES, Catcoc::detail::CopyMode::Gather>;
        using BlockEpilogueAllGather =
            CommEpilogue::Block::CommBlockEpilogue<EpilogueAllGatherDispatch, RemoteSrcType, RemoteDstType,
                                                   CommCoreSplit, CommBlockShape, EpilogueAllGatherTileShape,
                                                   TileRemoteCopy, TileScheduler>;

        constexpr uint32_t WORKSPACE_STAGES = 2;
        constexpr uint32_t COMM_INTERVAL = 4;
        using MatmulAllReduceKernel =
            DGemm::Kernel::MatmulAllReduce<BlockMmad, BlockEpilogueReduceScatter, BlockEpilogueAllGather,
                                           BlockMmadScheduler, BlockEpilogueScheduler, WORKSPACE_STAGES>;

        typename BlockEpilogueReduceScatter::Params reduceScatterParams{};
        typename BlockEpilogueAllGather::Params allGatherParams{};

        typename MatmulAllReduceKernel::Params params{problemShape,
                                                      rankIdx,
                                                      rankSize,
                                                      COMM_INTERVAL,
                                                      gmA,
                                                      layoutA,
                                                      gmB,
                                                      layoutB,
                                                      gmD,
                                                      layoutD,
                                                      gmSymmetric,
                                                      reduceScatterParams,
                                                      allGatherParams};

        MatmulAllReduceKernel matmulAllReduceKernel;
        matmulAllReduceKernel(params);
    }
};

template <DataFormatMode dMode, WeightFormatMode wMode>
__aicore__ void catcoc_matmul_allreduce_impl(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
                                             GM_ADDR gmSymmetric, GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    CatCocMatmulAllreduce<dMode, wMode> op;
    op.Init(fftsAddr, teamIdx);
    op.Process(gmA, gmB, gmD, gmSymmetric, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_fp16(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                   GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
                                                                   GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_matmul_allreduce_impl<DataFormatMode::FP16, WeightFormatMode::WEIGHT_ND>(fftsAddr, teamIdx, gmA, gmB, gmD,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_fp16_wnz(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                       GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
                                                                       GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_matmul_allreduce_impl<DataFormatMode::FP16, WeightFormatMode::WEIGHT_NZ>(fftsAddr, teamIdx, gmA, gmB, gmD,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_bf16(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                   GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
                                                                   GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_matmul_allreduce_impl<DataFormatMode::BF16, WeightFormatMode::WEIGHT_ND>(fftsAddr, teamIdx, gmA, gmB, gmD,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_bf16_wnz(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                       GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
                                                                       GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_matmul_allreduce_impl<DataFormatMode::BF16, WeightFormatMode::WEIGHT_NZ>(fftsAddr, teamIdx, gmA, gmB, gmD,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

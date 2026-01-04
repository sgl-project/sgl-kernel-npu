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
#include "catcoc/dgemm/block/block_swizzle_allgather.h"
#include "catcoc/dgemm/kernel/allgather_matmul.h"

// shmem_device
#include "shmem_api.h"

#include "catcoc_host_tiling.h"

using namespace AscendC;
using namespace Catcoc;
using namespace sglang::npu_kernel;

template <DataFormatMode dMode, WeightFormatMode wMode>
class CatCocAllgatherMatmul
{
public:
    uint64_t fftsAddr_;
    int32_t teamIdx_;

    __aicore__ inline void Init(uint64_t fftsAddr, uint64_t teamIdx)
    {
        fftsAddr_ = fftsAddr;
        teamIdx_ = (int32_t)teamIdx;
    }

    __aicore__ inline void Process(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric, GM_ADDR gmTiling)
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

        /*
        if(rankIdx == 0) {
            AscendC::printf("m is: %u ;", tiling_data->m);
            AscendC::printf("n is: %u ;", tiling_data->n);
            AscendC::printf("k is: %u ;\n", k);
            AscendC::printf("rankIdx is %u ; rankSize is %u ; teamIdx is: %d ;\n", rankIdx, rankSize, newTeamIdx);

            AscendC::printf("[dev] tiling_ptr on device is %lu \n", (uint64_t) tiling_data);
            AscendC::printf("[dev] ipt_a_ptr is %ld, ipt_b_ptr is %ld, opt_c_ptr is %ld\n", gmA, gmB, gmC);
            printf("[dev] fftsAddr is %lu, symm_ptr is %lu\n", fftsAddr, (uint64_t) gmSymmetric);
        }
        */

        /*
        uint32_t swizzleOffset = tiling_data->swizzleOffset;
        uint32_t swizzleDirect = tiling_data->swizzleDirect;
        uint32_t pValue = tiling_data->pValue;
        uint32_t commDataSplit = tiling_data->commDataSplit;
        uint32_t commNpuSplit = tiling_data->commNpuSplit;
        uint32_t ubMoveNum = tiling_data->ubMoveNum;
        uint32_t lenPerLoop = tiling_data->lenPerLoop;
        */

        // switch cases
        using ElementA =
            typename std::conditional_t<dMode == DataFormatMode::FP16, half,
                                        typename std::conditional_t<dMode == DataFormatMode::BF16, __bf16, float>>;
        using ElementB = ElementA;
        using ElementC = ElementA;

        using LayoutA = Catlass::layout::RowMajor;
        using LayoutB = typename std::conditional_t<wMode == WeightFormatMode::WEIGHT_ND, Catlass::layout::RowMajor,
                                                    Catlass::layout::zN>;
        using LayoutC = Catlass::layout::RowMajor;

        using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
        using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
        using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

        LayoutA layoutA{m, k};
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);  // adapted for both NZ and ND
        LayoutC layoutC{m * rankSize, n};
        Catlass::GemmCoord problemShape{m, n, k};

        /* init catcoc instance and run */
        constexpr bool enableUnitFlag = true;
        using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
        using L1TileShape = Catlass::GemmShape<128, 256, 256>;
        using L0TileShape = Catlass::GemmShape<128, 256, 64>;
        using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
        using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
        using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
        using BlockMmad =
            Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

        using BlockSchedulerForAllgather = Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
        using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;

        using RemoteSrcType = AType;
        using RemoteDstType = AType;
        using CopyDirect = Catcoc::detail::CopyDirect;
        using TileRemoteCopy =
            CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
        using TileSchedulerForAllgather = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

        using CommBlockShape = Catlass::MatrixShape<64, UINT_MAX / 2>;
        using CommCoreSplit = Catlass::MatrixShape<20, 1>;

        constexpr uint32_t UB_STAGES = 2;
        using AllGatherTileShape = Catlass::MatrixShape<32, 256>;
        using AllGatherDispatch =
            CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES, Catcoc::detail::CopyMode::Gather>;
        using BlockEpilogueAllGather =
            CommEpilogue::Block::CommBlockEpilogue<AllGatherDispatch, RemoteSrcType, RemoteDstType, CommCoreSplit,
                                                   CommBlockShape, AllGatherTileShape, TileRemoteCopy,
                                                   TileSchedulerForAllgather>;

        constexpr uint32_t WORKSPACE_STAGES = 2;
        constexpr uint32_t COMM_INTERVAL = 3;
        using AllGatherMatmulKernel =
            DGemm::Kernel::AllGatherMatmul<BlockMmad, BlockEpilogueAllGather, BlockSchedulerForAllgather,
                                           CommBlockScheduler, WORKSPACE_STAGES>;

        typename BlockEpilogueAllGather::Params allGatherParams{};

        // Prepare params
        typename AllGatherMatmulKernel::Params params{problemShape,  rankIdx,     rankSize,  // newTeamIdx,
                                                      COMM_INTERVAL, gmA,         layoutA,        gmB, layoutB, gmC,
                                                      layoutC,       gmSymmetric, allGatherParams};

        // Call kernel
        AllGatherMatmulKernel matmulCommKernel;
        matmulCommKernel(params);
    }
};

template <DataFormatMode dMode, WeightFormatMode wMode>
__aicore__ void catcoc_allgather_matmul_impl(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
                                             GM_ADDR gmSymmetric, GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    CatCocAllgatherMatmul<dMode, wMode> op;
    op.Init(fftsAddr, teamIdx);
    op.Process(gmA, gmB, gmC, gmSymmetric, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_allgather_matmul_fp16(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                   GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
                                                                   GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_allgather_matmul_impl<DataFormatMode::FP16, WeightFormatMode::WEIGHT_ND>(fftsAddr, teamIdx, gmA, gmB, gmC,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_allgather_matmul_fp16_wnz(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                       GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
                                                                       GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_allgather_matmul_impl<DataFormatMode::FP16, WeightFormatMode::WEIGHT_NZ>(fftsAddr, teamIdx, gmA, gmB, gmC,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_allgather_matmul_bf16(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                   GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
                                                                   GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_allgather_matmul_impl<DataFormatMode::BF16, WeightFormatMode::WEIGHT_ND>(fftsAddr, teamIdx, gmA, gmB, gmC,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

extern "C" __global__ __aicore__ void catcoc_allgather_matmul_bf16_wnz(uint64_t fftsAddr, uint64_t teamIdx, GM_ADDR gmA,
                                                                       GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
                                                                       GM_ADDR gmWorkspace, GM_ADDR gmTiling)
{
    catcoc_allgather_matmul_impl<DataFormatMode::BF16, WeightFormatMode::WEIGHT_NZ>(fftsAddr, teamIdx, gmA, gmB, gmC,
                                                                                    gmSymmetric, gmWorkspace, gmTiling);
}

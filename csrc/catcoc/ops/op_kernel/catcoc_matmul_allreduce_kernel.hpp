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

extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_fp16(uint64_t fftsAddr, uint64_t teamIdx,
                                                                   GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
                                                                   GM_ADDR gmSymmetric, GM_ADDR gmWorkspace,
                                                                   GM_ADDR gmTiling)
{
    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    // Set shmem config
    int32_t newTeamIdx = 0;
    // int32_t newTeamIdx = (int32_t)teamIdx;
    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using ElementD = half;

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

    Catlass::layout::RowMajor layoutA{m, k};
    Catlass::layout::RowMajor layoutB{k, n};
    Catlass::layout::RowMajor layoutD{m, n};

    Catlass::GemmCoord blockShape{m0, n0, k0};
    Catlass::GemmCoord problemShape{m, n, k};

    /* init catcoc instance and run */
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<ElementB, Catlass::layout::RowMajor>;
    using CType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using DType = Catlass::Gemm::GemmType<ElementD, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
            MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t SCATTER_TILE_ROWS = 32;
    constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueReduceScatterDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueAllGatherDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using MatmulAllReduceKernel = DGemm::Kernel::MatmulAllReduce<
            BlockMmad,
            BlockEpilogueReduceScatter,
            BlockEpilogueAllGather,
            BlockMmadScheduler,
            BlockEpilogueScheduler,
            WORKSPACE_STAGES
    >;

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};
    typename BlockEpilogueAllGather::Params allGatherParams{};

    typename MatmulAllReduceKernel::Params params{
            problemShape, rankIdx, rankSize,
            COMM_INTERVAL,
            gmA, layoutA,
            gmB, layoutB,
            gmD, layoutD,
            gmSymmetric,
            reduceScatterParams,
            allGatherParams
    };

    MatmulAllReduceKernel matmulAllReduceKernel;
    matmulAllReduceKernel(params);
}

extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_fp16_wnz(uint64_t fftsAddr, uint64_t teamIdx,
                                                                       GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
                                                                       GM_ADDR gmSymmetric, GM_ADDR gmWorkspace,
                                                                       GM_ADDR gmTiling)
{
    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    // Set shmem config
    int32_t newTeamIdx = 0;
    // int32_t newTeamIdx = (int32_t)teamIdx;
    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using ElementD = half;

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

    Catlass::layout::RowMajor layoutA{m, k};
    Catlass::layout::zN layoutB = Catlass::layout::zN::MakeLayout<ElementB>(k, n);
    Catlass::layout::RowMajor layoutD{m, n};

    Catlass::GemmCoord blockShape{m0, n0, k0};
    Catlass::GemmCoord problemShape{m, n, k};

    /* init catcoc instance and run */
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<ElementB, Catlass::layout::zN>;
    using CType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using DType = Catlass::Gemm::GemmType<ElementD, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
            MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t SCATTER_TILE_ROWS = 32;
    constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueReduceScatterDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueAllGatherDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using MatmulAllReduceKernel = DGemm::Kernel::MatmulAllReduce<
            BlockMmad,
            BlockEpilogueReduceScatter,
            BlockEpilogueAllGather,
            BlockMmadScheduler,
            BlockEpilogueScheduler,
            WORKSPACE_STAGES
    >;

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};
    typename BlockEpilogueAllGather::Params allGatherParams{};

    typename MatmulAllReduceKernel::Params params{
            problemShape, rankIdx, rankSize,
            COMM_INTERVAL,
            gmA, layoutA,
            gmB, layoutB,
            gmD, layoutD,
            gmSymmetric,
            reduceScatterParams,
            allGatherParams
    };

    MatmulAllReduceKernel matmulAllReduceKernel;
    matmulAllReduceKernel(params);
}


extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_bf16(uint64_t fftsAddr, uint64_t teamIdx,
                                                                   GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
                                                                   GM_ADDR gmSymmetric, GM_ADDR gmWorkspace,
                                                                   GM_ADDR gmTiling)
{
    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    // Set shmem config
    int32_t newTeamIdx = 0;
    // int32_t newTeamIdx = (int32_t)teamIdx;
    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    using ElementA = __bf16;
    using ElementB = __bf16;
    using ElementC = __bf16;
    using ElementD = __bf16;

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

    Catlass::layout::RowMajor layoutA{m, k};
    Catlass::layout::RowMajor layoutB{k, n};
    Catlass::layout::RowMajor layoutD{m, n};

    Catlass::GemmCoord blockShape{m0, n0, k0};
    Catlass::GemmCoord problemShape{m, n, k};

    /* init catcoc instance and run */
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<ElementB, Catlass::layout::RowMajor>;
    using CType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using DType = Catlass::Gemm::GemmType<ElementD, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
            MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t SCATTER_TILE_ROWS = 32;
    constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueReduceScatterDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueAllGatherDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using MatmulAllReduceKernel = DGemm::Kernel::MatmulAllReduce<
            BlockMmad,
            BlockEpilogueReduceScatter,
            BlockEpilogueAllGather,
            BlockMmadScheduler,
            BlockEpilogueScheduler,
            WORKSPACE_STAGES
    >;

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};
    typename BlockEpilogueAllGather::Params allGatherParams{};

    typename MatmulAllReduceKernel::Params params{
            problemShape, rankIdx, rankSize,
            COMM_INTERVAL,
            gmA, layoutA,
            gmB, layoutB,
            gmD, layoutD,
            gmSymmetric,
            reduceScatterParams,
            allGatherParams
    };

    MatmulAllReduceKernel matmulAllReduceKernel;
    matmulAllReduceKernel(params);
}


extern "C" __global__ __aicore__ void catcoc_matmul_allreduce_bf16_wnz(uint64_t fftsAddr, uint64_t teamIdx,
                                                                       GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
                                                                       GM_ADDR gmSymmetric, GM_ADDR gmWorkspace,
                                                                       GM_ADDR gmTiling)
{
    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    // Set shmem config
    int32_t newTeamIdx = 0;
    // int32_t newTeamIdx = (int32_t)teamIdx;
    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    using ElementA = __bf16;
    using ElementB = __bf16;
    using ElementC = __bf16;
    using ElementD = __bf16;

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

    Catlass::layout::RowMajor layoutA{m, k};
    Catlass::layout::zN layoutB = Catlass::layout::zN::MakeLayout<ElementB>(k, n);
    Catlass::layout::RowMajor layoutD{m, n};

    Catlass::GemmCoord blockShape{m0, n0, k0};
    Catlass::GemmCoord problemShape{m, n, k};

    /* init catcoc instance and run */
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, Catlass::layout::RowMajor>;
    using BType = Catlass::Gemm::GemmType<ElementB, Catlass::layout::zN>;
    using CType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using DType = Catlass::Gemm::GemmType<ElementD, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
            MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t SCATTER_TILE_ROWS = 32;
    constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueReduceScatterDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
            EpilogueAllGatherDispatch,
            RemoteSrcType, RemoteDstType,
            CommCoreSplit,
            CommBlockShape,
            EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using MatmulAllReduceKernel = DGemm::Kernel::MatmulAllReduce<
            BlockMmad,
            BlockEpilogueReduceScatter,
            BlockEpilogueAllGather,
            BlockMmadScheduler,
            BlockEpilogueScheduler,
            WORKSPACE_STAGES
    >;

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};
    typename BlockEpilogueAllGather::Params allGatherParams{};

    typename MatmulAllReduceKernel::Params params{
            problemShape, rankIdx, rankSize,
            COMM_INTERVAL,
            gmA, layoutA,
            gmB, layoutB,
            gmD, layoutD,
            gmSymmetric,
            reduceScatterParams,
            allGatherParams
    };

    MatmulAllReduceKernel matmulAllReduceKernel;
    matmulAllReduceKernel(params);
}


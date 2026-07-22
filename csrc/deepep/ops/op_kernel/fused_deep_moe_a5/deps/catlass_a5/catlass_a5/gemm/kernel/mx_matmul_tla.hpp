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

#ifndef CATLASS_GEMM_KERNEL_MX_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_MX_MATMUL_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm/block/block_scheduler_aswt.hpp"
#include "catlass_a5/gemm/block/block_swizzle.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MxMatmulTlaBase
{
public:
    using BlockMmad = BlockMmad_;
    using MmadArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        GM_ADDR ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrMxScaleA_, LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_,
               LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrBias_ = nullptr)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrMxScaleA(ptrMxScaleA_),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(ptrMxScaleB_),
              layoutMxScaleB(layoutMxScaleB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrBias(ptrBias_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t *ptrA;
        LayoutA layoutA;
        uint8_t *ptrB;
        LayoutB layoutB;
        uint8_t *ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        uint8_t *ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        uint8_t *ptrC;
        LayoutC layoutC;
        uint8_t *ptrBias{nullptr};
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.problemShape,   args.ptrA,        args.layoutA,        args.ptrB,
                      args.layoutB,        args.ptrMxScaleA, args.layoutMxScaleA, args.ptrMxScaleB,
                      args.layoutMxScaleB, args.ptrC,        args.layoutC,        args.ptrBias};
        return params;
    }
};

// Primary template is an explicit fallback. Supported schedulers are implemented via partial specializations.
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MxMatmulTla
{
    static_assert(DEPENDENT_FALSE<BlockScheduler_>,
                  "Unsupported BlockScheduler for MxMatmulTla. "
                  "Please use Gemm::Block::GemmIdentityBlockSwizzle<...> or "
                  "Gemm::Block::BlockSchedulerAswt<...>.");
};

template <class BlockMmad_, class BlockEpilogue_, class L1TileShape_, class L0TileShape_, bool isGmm_>
class MxMatmulTla<BlockMmad_, BlockEpilogue_, Gemm::Block::BlockSchedulerAswt<L1TileShape_, L0TileShape_, isGmm_>>
    : public MxMatmulTlaBase<BlockMmad_, BlockEpilogue_,
                             Gemm::Block::BlockSchedulerAswt<L1TileShape_, L0TileShape_, isGmm_>>
{
    using Base = MxMatmulTlaBase<BlockMmad_, BlockEpilogue_,
                                 Gemm::Block::BlockSchedulerAswt<L1TileShape_, L0TileShape_, isGmm_>>;

public:
    using Params = typename Base::Params;
    using Arguments = typename Base::Arguments;
    using BlockMmad = typename Base::BlockMmad;
    using MmadArchTag = typename Base::MmadArchTag;
    using L1TileShape = typename Base::L1TileShape;
    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementMxScaleA = typename Base::ElementMxScaleA;
    using LayoutMxScaleA = typename Base::LayoutMxScaleA;
    using ElementMxScaleB = typename Base::ElementMxScaleB;
    using LayoutMxScaleB = typename Base::LayoutMxScaleB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;
    using ElementBias = typename Base::ElementBias;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using BlockScheduler = typename Base::BlockScheduler;

    CATLASS_DEVICE
    MxMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        BlockScheduler matmulBlockScheduler(curBlockIdx, blockNum, params.problemShape);

        constexpr bool isMxFp8 = std::is_same_v<ElementA, ElementB> &&
                                 (std::is_same_v<ElementA, float8_e4m3_t> || std::is_same_v<ElementA, float8_e5m2_t>);
        if constexpr (isMxFp8) {
            // enable tail wave workload balance only when ElementA and ElementB are both float8_e4m3_t or
            // float8_e5m2_t.
            if (matmulBlockScheduler.endBlockIdx_ + 1 <= blockNum / 2) {
                matmulBlockScheduler.UpdateTailTile();
            }
        }

        Arch::Resource<MmadArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        if (CeilDiv(params.problemShape.m(), Base::L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), Base::L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        uint32_t coreLoops = matmulBlockScheduler.round_;
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            bool isLastLoop = (loopIdx == coreLoops - 1 && curBlockIdx <= matmulBlockScheduler.endBlockIdx_);
            matmulBlockScheduler.UpdateMNTileIdx(loopIdx, isLastLoop);
            matmulBlockScheduler.UpdateBlockShape(loopIdx, isLastLoop);

            auto blockShape = matmulBlockScheduler.GetBlockShape();
            auto blkElemCoord = matmulBlockScheduler.GetBlockCoordByElement();

            if (blockShape.m() == 0 || blockShape.n() == 0) {
                continue;
            }

            uint32_t mOffset = blkElemCoord.m();
            uint32_t nOffset = blkElemCoord.n();

            auto tensorBlockA =
                GetTile(tensorA, tla::MakeCoord(mOffset, 0), tla::MakeShape(blockShape.m(), blockShape.k()));
            auto tensorBlockB =
                GetTile(tensorB, tla::MakeCoord(0, nOffset), tla::MakeShape(blockShape.k(), blockShape.n()));
            auto tensorBlockMxScaleA =
                GetTile(tensorMxScaleA, tla::MakeCoord(mOffset, 0),
                        tla::MakeShape(blockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(blockShape.k())));
            auto tensorBlockMxScaleB =
                GetTile(tensorMxScaleB, tla::MakeCoord(0, nOffset),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(blockShape.k()), blockShape.n()));
            auto tensorBlockC =
                GetTile(tensorC, tla::MakeCoord(mOffset, nOffset), tla::MakeShape(blockShape.m(), blockShape.n()));

            if constexpr (std::is_void_v<ElementBias>) {
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, blockShape, tensorBlockMxScaleA,
                          tensorBlockMxScaleB);
            } else {
                auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(nOffset), tla::MakeShape(blockShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, blockShape, tensorBlockMxScaleA,
                          tensorBlockMxScaleB, tensorBlockBias);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {}
};

template <class BlockMmad_, class BlockEpilogue_, uint32_t SwizzleOffset, uint32_t SwizzleDirection>
class MxMatmulTla<BlockMmad_, BlockEpilogue_, Gemm::Block::GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>>
    : public MxMatmulTlaBase<BlockMmad_, BlockEpilogue_,
                             Gemm::Block::GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>>
{
    using Base = MxMatmulTlaBase<BlockMmad_, BlockEpilogue_,
                                 Gemm::Block::GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>>;

public:
    using Params = typename Base::Params;
    using Arguments = typename Base::Arguments;
    using BlockMmad = typename Base::BlockMmad;
    using MmadArchTag = typename Base::MmadArchTag;
    using L1TileShape = typename Base::L1TileShape;
    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementMxScaleA = typename Base::ElementMxScaleA;
    using LayoutMxScaleA = typename Base::LayoutMxScaleA;
    using ElementMxScaleB = typename Base::ElementMxScaleB;
    using LayoutMxScaleB = typename Base::LayoutMxScaleB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;
    using ElementBias = typename Base::ElementBias;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using BlockScheduler = typename Base::BlockScheduler;

    static constexpr uint32_t L1_TILE_M = Base::L1_TILE_M;
    static constexpr uint32_t L1_TILE_N = Base::L1_TILE_N;
    static constexpr uint32_t L1_TILE_K = Base::L1_TILE_K;

    CATLASS_DEVICE
    MxMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<MmadArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockMxScaleA =
                GetTile(tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
            auto tensorBlockMxScaleB =
                GetTile(tensorMxScaleB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            if constexpr (std::is_void_v<ElementBias>) {
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                          tensorBlockMxScaleB);
            } else {
                auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N),
                                               tla::MakeShape(actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                          tensorBlockMxScaleB, tensorBlockBias);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {}
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_MX_MATMUL_TLA_HPP

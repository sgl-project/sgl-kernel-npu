/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_PER_TENSOR_PER_CHANNEL_DEQUANT_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_PER_TENSOR_PER_CHANNEL_DEQUANT_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Template for grouped matmul kernel. Compute grouped C = A * B
template <class BlockMmadTla_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class GroupedMatmulSliceMFixpipeDequantTla
{
public:
    using BlockMmadTla = BlockMmadTla_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = typename BlockMmadTla::ArchTag;
    using L1TileShape = typename BlockMmadTla::L1TileShape;
    using ElementA = typename BlockMmadTla::ElementA;
    using LayoutA = typename BlockMmadTla::LayoutA;
    using ElementB = typename BlockMmadTla::ElementB;
    using LayoutB = typename BlockMmadTla::LayoutB;
    using ElementC = typename BlockMmadTla::ElementC;
    using LayoutC = typename BlockMmadTla::LayoutC;
    using ElementAccumulator = typename BlockMmadTla::ElementAccumulator;
    using ElementGroupList = ElementGroupList_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        float perTensorScale;
        __gm__ uint64_t *ptrPerChannelScale;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrC_, LayoutC const &layoutC_,
               float perTensorScale_, GM_ADDR ptrPerChannelScale_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)),
              layoutC(layoutC_),
              perTensorScale(perTensorScale_),
              ptrPerChannelScale(reinterpret_cast<__gm__ uint64_t *>(ptrPerChannelScale_))
        {}
    };
    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t *ptrGroupList;
        uint8_t *ptrA;
        LayoutA layoutA;
        uint8_t *ptrB;
        LayoutB layoutB;
        uint8_t *ptrC;
        LayoutC layoutC;
        float perTensorScale;
        uint8_t *ptrPerChannelScale;
    };
    static bool CanImplement(const Arguments &args)
    {
        return true;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }
    static Params ToUnderlyingArguments(const Arguments &args, void *workspace)
    {
        Params params{args.problemShape, args.problemCount,   args.ptrGroupList,      args.ptrA,
                      args.layoutA,      args.ptrB,           args.layoutB,           args.ptrC,
                      args.layoutC,      args.perTensorScale, args.ptrPerChannelScale};

        return params;
    }
    // Methods
    CATLASS_HOST_DEVICE
    GroupedMatmulSliceMFixpipeDequantTla() {}
    // Methods
    CATLASS_HOST_DEVICE
    ~GroupedMatmulSliceMFixpipeDequantTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();

        Arch::Resource<ArchTag> resource;
        BlockMmadTla blockMmadTla(resource);
        BlockScheduler blockScheduler(curBlockIdx, blockNum);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);
        AscendC::GlobalTensor<uint64_t> gmScale;
        gmScale.SetGlobalBuffer(params.ptrPerChannelScale);

        int64_t gmGroupOffsetB = 0;
        uint32_t mStart = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto layoutQuant = tla::MakeLayout(params.problemShape.n());
        auto tensorQuant = tla::MakeTensor(gmScale, layoutQuant, Arch::PositionGM{});

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));

            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            blockScheduler.UpdateGroupParams(inGroupProblemShape);
            if (groupIdx == params.problemCount - 1 && blockScheduler.endBlockIdx_ + 1 <= blockNum / 2) {
                blockScheduler.UpdateTailTile();
            }

            // 每个专家的权重不一样，需要设置GlobalTensor
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});

            // Loop through the matmul of each groupIdx
            uint32_t coreLoops = blockScheduler.round_;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                // Compute block information
                bool isLastGroupRound = groupIdx == params.problemCount - 1 && loopIdx == coreLoops - 1 &&
                                        curBlockIdx <= blockScheduler.endBlockIdx_;
                blockScheduler.UpdateMNTileIdx(loopIdx, isLastGroupRound);
                blockScheduler.UpdateBlockShape(loopIdx, isLastGroupRound);
                auto blockShape = blockScheduler.GetBlockShape();
                auto blkElemCoord = blockScheduler.GetBlockCoordByElement();

                uint32_t mCoord = blkElemCoord.m();
                uint32_t nCoord = blkElemCoord.n();
                uint32_t kCoord = blkElemCoord.k();

                uint32_t blockM = blockShape.m();
                uint32_t blockN = blockShape.n();
                uint32_t blockK = blockShape.k();

                // Make tiled views
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(mStart + mCoord, kCoord), tla::MakeShape(blockM, blockK));
                auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(kCoord, nCoord), tla::MakeShape(blockK, blockN));
                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(mStart + mCoord, nCoord), tla::MakeShape(blockM, blockN));
                auto tensorBlockQuant = GetTile(tensorQuant, tla::MakeCoord(nCoord), tla::MakeShape(blockN));

                GemmCoord blockShapeMNK{static_cast<uint32_t>(blockM), static_cast<uint32_t>(blockN),
                                        static_cast<uint32_t>(blockK)};
                // Compute block-scoped matrix multiply-add
                blockMmadTla(tensorBlockA, tensorBlockB, tensorBlockC, blockShape, {}, tensorBlockQuant,
                             params.perTensorScale);
            }
            mStart += inGroupProblemShape.m();
            gmGroupOffsetB += static_cast<int64_t>(inGroupProblemShape.k()) * inGroupProblemShape.n();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {}
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_PER_TENSOR_PER_CHANNEL_DEQUANT_TLA_HPP
